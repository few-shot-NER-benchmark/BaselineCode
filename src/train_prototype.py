import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import torch
device = torch.device("cuda")
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import time

from collections import Counter, defaultdict
import numpy as np
from sample_few_shot import get_label_dict
from meta_model import RobertaNER, BertNER
from eval_util import batch_span_eval
from data import *
from torch.utils.tensorboard import SummaryWriter


def train_func(sub_trains_, epoch, tokenizer, id2labels, label_sentence_dicts, soft_kmeans):

    # Train the model
    train_loss = 0
    total_pred, total_gold, total_crct = 0.0, 0.0, 0.0
    # data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
    #                   collate_fn=generate_batch)  
    # model.train()
    group_num = -(-EPISODE_NUM // 200)
    for g_num in range(group_num):
        data, dataset_chosen = generate_episode(sub_trains_, epoch * group_num + g_num, min(200, EPISODE_NUM - g_num*200), max_seq_len, SUP_CLS_NUM, SUP_PER_CLS, QUR_PER_CLS, id2labels, label_sentence_dicts, 
            gpu_num=gpu_num, uniform = False, all_o_sent = int(QUR_PER_CLS * o_sent_ratio))
        for j, (text, attention_mask, cls, orig_len, support_class, inv_support_classes) in enumerate(data):
            # print(support_class)
            # print(inv_support_classes)
            id2label = id2labels[dataset_chosen[j]]
            optimizer.zero_grad()
            outputs = []
            preds = []
            gold = []
            global_class_map = []

            for cls_i in range(2 * SUP_CLS_NUM + 1):
                global_class_map.append(support_class[cls_i])
            global_class_map = torch.tensor(global_class_map)
            device = torch.device("cuda")
            global_class_map.to(device)
            large_group_num = SUP_CLS_NUM * (SUP_PER_CLS + QUR_PER_CLS)
            # epi_iter_num = torch.tensor(epi_iter_num)
            for cls_i in range(SUP_CLS_NUM):
                sel_idx = []
                
                for g in range(gpu_num):
                    sel_idx.extend([x + large_group_num * g for x in range(SUP_CLS_NUM * SUP_PER_CLS)])
                sel_idx = torch.tensor(sel_idx)
                text_1, attention_mask_1, cls_1 = torch.index_select(text, 0, sel_idx).to(device), torch.index_select(attention_mask, 0, sel_idx).to(device), torch.index_select(cls, 0, sel_idx).to(device)
                # epi_iter_num1 = torch.index_select(epi_iter_num,0,sel_idx)
                # print(f"sel_idx {sel_idx}")

                sel_idx = []
                for g in range(gpu_num):
                    sel_idx.extend([x + large_group_num * g for x in range(SUP_CLS_NUM * SUP_PER_CLS + cls_i * QUR_PER_CLS, SUP_CLS_NUM * SUP_PER_CLS + (cls_i + 1) * QUR_PER_CLS)])
                sel_idx = torch.tensor(sel_idx)
                # print(f"sel_idx {sel_idx}")
                # if use this code, should notice that cls_1 might change (cls won't)
                # for sent in cls_1:
                #     for k,cl in enumerate(sent):
                #         if cl == 0 and np.random.random_sample() > 0.33:
                #             sent[k]=-100
                text_2, attention_mask_2, cls_2 = torch.index_select(text, 0, sel_idx).to(device), torch.index_select(attention_mask, 0, sel_idx).to(device), torch.index_select(cls, 0, sel_idx).to(device)   
                # epi_iter_num2 = torch.index_select(epi_iter_num,0,sel_idx)
                # prototypes, embed_class_len = model.compute_prototypes(text_1, support_class_num = SUP_CLS_NUM, attention_mask=attention_mask_1, labels=cls_1)

                loss, output = model(text_1, text_2, cls_1, cls_2,\
                    attention_mask_1, attention_mask_2, \
                    SUP_CLS_NUM, use_global = False, \
                    dataset_chosen = dataset_chosen[j], global_class_map = global_class_map.to(device), \
                    soft_kmeans = soft_kmeans, \
                    metric = args.metric, norm = args.norm, \
                    class_metric=args.class_metric, \
                    instance_metric=args.instance_metric
                    )
                # print(f"output shape outside model: {output.shape}")
                loss.mean().backward()
                train_loss += loss.mean().item()
                # outputs.extend(output)

                new_len = []
                for g in range(gpu_num):
                    new_len.extend([orig_len[i + g * large_group_num + SUP_CLS_NUM * SUP_PER_CLS + QUR_PER_CLS * cls_i] for i in range(int(len(output)/gpu_num))])
                # print(new_len)
                pred = [[id2label[support_class[int(x)]] for j,x in enumerate(y[1:new_len[i]-1]) if int(cls_2[i][j + 1]) != -100] for i,y in enumerate(output)]
                gt = [[id2label[support_class[int(x)]] for x in y[1:new_len[i]-1] if int(x) != -100] for i,y in enumerate(cls_2)]         
                preds.extend(pred)
                gold.extend(gt)
                if 'shotxxx' in args.train_text:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            if 'shotxxx' not in args.train_text:
                optimizer.step()
                scheduler.step()
            
            print(f"batch: {(j+1) * gpu_num + g_num * 200}/{EPISODE_NUM} lr: {optimizer.param_groups[0]['lr']:.9f} loss: {loss.mean().item()/QUR_PER_CLS/SUP_CLS_NUM:.9f}")
            writer.add_scalar('prototype_euclidean_'+args.dataset+'/'+model_name.split('/')[2]+' lr ',optimizer.param_groups[0]['lr'],epoch*EPISODE_NUM + (j+1) * gpu_num + g_num * 200)
            writer.add_scalar('prototype_euclidean_'+args.dataset+'/'+model_name.split('/')[2]+' loss ',loss.mean().item()/QUR_PER_CLS/SUP_CLS_NUM,epoch*EPISODE_NUM + (j+1) * gpu_num + g_num * 200)
    
            # if ((j+1) * gpu_num + g_num * 200)%16 == 0:
            #     print(f"batch: {(j+1) * gpu_num + g_num * 200}/{EPISODE_NUM} lr: {optimizer.param_groups[0]['lr']:.9f} loss: {loss.mean().item()/QUR_PER_CLS/SUP_CLS_NUM:.9f}")
                # print([tokenizer.convert_ids_to_tokens(int(x)) for x in text[SUP_CLS_NUM * SUP_PER_CLS][1:orig_len[SUP_CLS_NUM * SUP_PER_CLS]-1]])
                # print(preds[0])
                # print(len(preds[0]))
                # print(gold[0])
                # print(len(gold[0]))

            
                
            bpred, bgold, bcrct, _, _, _ = batch_span_eval(preds, gold)
            total_pred += bpred
            total_gold += bgold
            total_crct += bcrct
            # Adjust the learning rate
            # scheduler.step()

            # if ((j+1) * gpu_num )%200 == 64:
            #     print('saving model')
            #     state = {
            #         'epoch': epoch,
            #         'state_dict': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'scheduler': scheduler.state_dict(),
            #         'g_num': g_num,
            #     }
            #     torch.save(state, 'save/prototype/'+ 'checkpoint_'+str((j+1) * gpu_num + g_num * 200)+args.model_name.split('/')[-1]+'.pt')


    microp = total_crct/total_pred if total_pred > 0 else 0
    micror = total_crct/total_gold if total_gold > 0 else 0
    microf1 = 2*microp*micror/(microp + micror) if (microp + micror) > 0 else 0

    return train_loss / train_num_data_point, microp, micror, microf1

def test(data_, epoch, id2labels, label_sentence_dicts, soft_kmeans, test_all_data = False, finetune = False, test_example = None):
    val_loss = 0
    total_pred, total_gold, total_crct = 0.0, 0.0, 0.0
    total_pred_per_type, total_gold_per_type, total_crct_per_type = defaultdict(int), defaultdict(int), defaultdict(int)
    # data = DataLoader(data_, batch_size=256, collate_fn=generate_batch)
    # data = generate_episode(data_, epoch,  EPISODE_NUM, TEST_SUP_CLS_NUM, test_id2label, label_sentence_dict)
    if test_all_data:
        id2label = id2labels[0]
        label_sentence_dict = label_sentence_dicts[0]
        data_ = data_[0]
        if test_example is not None:
            data = generate_testing_episode(data_, epoch+1, 32, max_seq_len, TEST_SUP_CLS_NUM, 0, id2label, label_sentence_dict)
        else:
            data = generate_testing_episode(data_, epoch+1, 32, max_seq_len, TEST_SUP_CLS_NUM, SUP_PER_CLS, id2label, label_sentence_dict)
    else:
        data, dataset_chosen = generate_episode(data_, epoch, 50, max_seq_len, TEST_SUP_CLS_NUM, SUP_PER_CLS, QUR_PER_CLS, id2labels, label_sentence_dicts, 
        gpu_num = gpu_num, uniform = False, all_o_sent = int(QUR_PER_CLS * o_sent_ratio))

    idx = 0
    fine_tuned = False
    for j, (text, attention_mask, cls, orig_len, support_class, inv_support_classes) in enumerate(data): 
        large_group_num = TEST_SUP_CLS_NUM * (SUP_PER_CLS + QUR_PER_CLS)
        global_class_map = []
        for cls_i in range(2 * SUP_CLS_NUM + 1):
            global_class_map.append(support_class[cls_i])
        global_class_map = torch.tensor(global_class_map)
        device = torch.device("cuda")
        global_class_map.to(device)
        if finetune:
            # print(f"learning rate:{optimizer_grouped_parameters[1]['lr']}")
            new_model = torch.load(model_name+str(epoch + 1)+'.pt', map_location=lambda storage, loc: storage.cuda(1))
            new_model.cuda_device = 1
            device = torch.device('cuda:1')
            new_param_optimizer = list(new_model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optim_grouped_parameters = [
                {'params': [p for n, p in new_param_optimizer
                            if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in new_param_optimizer
                            if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
            optim = AdamW(optim_grouped_parameters, lr=5e-5, correct_bias=True)
            # print(f"learning rate: {optim_grouped_parameters[1]['lr']}")
            if test_all_data and fine_tuned:
                pass
            else:
                fine_tuned = True
                for i in range(SUP_PER_CLS):   
                    print(f"fine tune: round {i}")                
                    optim.zero_grad()          
                    sel_idx = []
                    for cls_i in range(TEST_SUP_CLS_NUM):
                        sel_idx.extend([x for x in range(cls_i * SUP_PER_CLS, cls_i * SUP_PER_CLS + i)])
                        sel_idx.extend([x for x in range(cls_i * SUP_PER_CLS + i + 1, (cls_i + 1) * SUP_PER_CLS)])
                    sel_idx = torch.tensor(sel_idx)
                    # print(sel_idx)
                    text_1, attention_mask_1, cls_1 = torch.index_select(text, 0, sel_idx).to(device), torch.index_select(attention_mask, 0, sel_idx).to(device), torch.index_select(cls, 0, sel_idx).to(device)
                    prototypes, embed_class_len = new_model.compute_prototypes(text_1, support_class_num = TEST_SUP_CLS_NUM, attention_mask=attention_mask_1, labels=cls_1)                

                    sel_idx = []          
                    for cls_j in range(TEST_SUP_CLS_NUM):     
                    #     first_time = True                   
                        # for cls_i in range(TEST_SUP_CLS_NUM):
                        #     sel_idx = []
                        #     sel_idx.extend([x for x in range(cls_i * SUP_PER_CLS, cls_i * SUP_PER_CLS + i)])
                        #     sel_idx.extend([x for x in range(cls_i * SUP_PER_CLS + i + 1, (cls_i + 1) * SUP_PER_CLS)])
                        #     sel_idx = torch.tensor(sel_idx)
                        #     # print(sel_idx)
                        #     text_1, attention_mask_1, cls_1 = torch.index_select(text, 0, sel_idx).to(device), torch.index_select(attention_mask, 0, sel_idx).to(device), torch.index_select(cls, 0, sel_idx).to(device)
                        #     if first_time:
                        #         prototypes, embed_class_len = new_model.compute_prototypes(text_1, support_class_num = TEST_SUP_CLS_NUM, attention_mask=attention_mask_1, labels=cls_1)                
                        #         first_time = False
                        #     else:
                        #         prototypes, embed_class_len = new_model.compute_prototypes(text_1, support_class_num = TEST_SUP_CLS_NUM, \
                        #             orig_prototypes = prototypes, orig_embed_class_len = embed_class_len, attention_mask=attention_mask_1, labels=cls_1)                
                             
                        # print(f"line 125 {cls_j}")
                                        
                        sel_idx.extend([cls_j * SUP_PER_CLS + i])
                    sel_idx = torch.tensor(sel_idx)
                    text_1, attention_mask_1, cls_1 = torch.index_select(text, 0, sel_idx).to(device), torch.index_select(attention_mask, 0, sel_idx).to(device), torch.index_select(cls, 0, sel_idx).to(device)
                    loss, output = new_model(text_1, support_class_num = TEST_SUP_CLS_NUM, prototypes = prototypes, attention_mask=attention_mask_1, labels=cls_1)
                    # print(output)
                    # print(cls_1)
                    loss.mean().backward()
                    # val_loss += loss.item()
                    optim.step()
        else:
            # device = torch.device('cuda:0')
            new_model = model
            # new_model = torch.load(model_name+str(epoch + 1)+'.pt', map_location=lambda storage, loc: storage.cuda(0))
            # for m in new_model.modules():
            #     if isinstance(m, nn.Dropout):
            #         m.eval()


        with torch.no_grad():
            outputs = []

            if test_all_data:
                if j%20 == 0:
                    print(f"batch number: {j}")
                if test_example is not None:      
                    if j == 0:
                        if use_multi_prototype:
                            new_test_example = []
                            for c1 in range(int(len(test_example)/TEST_SUP_CLS_NUM)):
                                for c2 in range(TEST_SUP_CLS_NUM):
                                    new_test_example.append(test_example[int(c2 * len(test_example) / TEST_SUP_CLS_NUM) + c1])
                                    # print(int(c2 * len(test_example) / TEST_SUP_CLS_NUM) + c1)
                            # print(f"length of test example: {len(test_example)} length of new test example: {len(new_test_example)}")
                            support_prototypes = []
                            support_group_num = int(len(new_test_example) / TEST_SUP_CLS_NUM / 5)
                            for k in range(support_group_num):
                                support_group = new_test_example[k * int((len(new_test_example) + support_group_num - 1)/support_group_num): min((k+1) * int((len(new_test_example) + support_group_num - 1)/support_group_num), len(new_test_example))]
                                # print(f"length of support group {len(support_group)}")
                                example_data = generate_testing_example(support_group, 32, max_seq_len, TEST_SUP_CLS_NUM, id2label, label_sentence_dict)
                                prototypes = None
                                embed_class_len = None
                                for text_1, attention_mask_1, cls_1, orig_len_1, support_class_1, inv_support_classes_1 in example_data: 
                                    text_1, attention_mask_1, cls_1 = text_1.to(device), attention_mask_1.to(device), cls_1.to(device)
                                    prototypes, embed_class_len = new_model.compute_prototypes(text_1, support_class_num = TEST_SUP_CLS_NUM, \
                                        orig_prototypes = prototypes, orig_embed_class_len = embed_class_len,
                                        attention_mask=attention_mask_1, labels=cls_1)
                                support_prototypes.append(prototypes)
                        else:
                            # print("length of examples")
                            # print(len(test_example))
                            example_data = generate_testing_example(test_example, 32, max_seq_len, TEST_SUP_CLS_NUM, id2label, label_sentence_dict)
                            # print(len(example_data))
                            prototypes = None
                            embed_class_len = None
                            for k, (text_1, attention_mask_1, cls_1, orig_len_1, support_class_1, inv_support_classes_1) in enumerate(example_data): 
                                text_1, attention_mask_1, cls_1 = text_1.to(device), attention_mask_1.to(device), cls_1.to(device)
                                prototypes, embed_class_len = new_model.compute_prototypes(text_1, support_class_num = TEST_SUP_CLS_NUM, \
                                    orig_prototypes = prototypes, orig_embed_class_len = embed_class_len,
                                    attention_mask=attention_mask_1, labels=cls_1)
                                # if k %10 == 0:
                                #     print(embed_class_len)
                            print(embed_class_len)
                    text_1, attention_mask_1, cls_1 = text.to(device), attention_mask.to(device), cls.to(device)
                    if use_multi_prototype:
                        loss, output = new_model.forward_multi_prototype(text_1, support_class_num = TEST_SUP_CLS_NUM, prototype_groups = support_prototypes, soft_kmeans = soft_kmeans, attention_mask=attention_mask_1, labels=cls_1, 
                            metric = args.metric, norm = args.norm, class_metric=args.class_metric, instance_metric=args.instance_metric)
                    else:
                        loss, output = new_model.direct_forward(text_1, support_class_num = TEST_SUP_CLS_NUM, prototypes = prototypes, soft_kmeans = soft_kmeans, attention_mask=attention_mask_1, labels=cls_1, 
                            metric = args.metric, norm = args.norm, class_metric=args.class_metric, instance_metric=args.instance_metric)
                    val_loss += loss.mean().item()
                    outputs.extend(output)
                        
                else:
                    sel_idx = [x for x in range(TEST_SUP_CLS_NUM * SUP_PER_CLS)]
                    sel_idx = torch.tensor(sel_idx)
                    text_1, attention_mask_1, cls_1 = torch.index_select(text, 0, sel_idx).to(device), torch.index_select(attention_mask, 0, sel_idx).to(device), torch.index_select(cls, 0, sel_idx).to(device)
                    prototypes, _ = new_model.compute_prototypes(text_1, support_class_num = TEST_SUP_CLS_NUM, attention_mask=attention_mask_1, labels=cls_1)                

                    sel_idx = [x for x in range(TEST_SUP_CLS_NUM * SUP_PER_CLS, min(TEST_SUP_CLS_NUM * SUP_PER_CLS + 64, len(text)) )]
                    sel_idx = torch.tensor(sel_idx)
                    text_1, attention_mask_1, cls_1 = torch.index_select(text, 0, sel_idx).to(device), torch.index_select(attention_mask, 0, sel_idx).to(device), torch.index_select(cls, 0, sel_idx).to(device)
                    loss, output = new_model.direct_forward(text_1, support_class_num = TEST_SUP_CLS_NUM, prototypes = prototypes, soft_kmeans = soft_kmeans, attention_mask=attention_mask_1, labels=cls_1, 
                        metric = args.metric, norm = args.norm, class_metric=args.class_metric, instance_metric=args.instance_metric)
                    val_loss += loss.mean().item()
                    outputs.extend(output)
            else:  
                device = torch.device("cuda")
                id2label = id2labels[dataset_chosen[j]] 
                sel_idx = []     
                for g in range(gpu_num):
                    sel_idx.extend([x + large_group_num * g for x in range(TEST_SUP_CLS_NUM * SUP_PER_CLS)])
                sel_idx = torch.tensor(sel_idx)            
                text_1, attention_mask_1, cls_1 = torch.index_select(text, 0, sel_idx).to(device), torch.index_select(attention_mask, 0, sel_idx).to(device), torch.index_select(cls, 0, sel_idx).to(device)
                
                all_sel_idx = []
                for g in range(gpu_num):
                    for cls_i in range(SUP_CLS_NUM):
                        sel_idx = [x for x in range(g * large_group_num +TEST_SUP_CLS_NUM * SUP_PER_CLS + cls_i * QUR_PER_CLS, g * large_group_num +TEST_SUP_CLS_NUM * SUP_PER_CLS + (cls_i + 1) * QUR_PER_CLS)]
                        all_sel_idx.extend(sel_idx)
                all_sel_idx = torch.tensor(all_sel_idx)
                text_2, attention_mask_2, cls_2 = torch.index_select(text, 0, all_sel_idx).to(device), torch.index_select(attention_mask, 0, all_sel_idx).to(device), torch.index_select(cls, 0, all_sel_idx).to(device)
                
                # prototypes, embed_class_len = new_model.compute_prototypes(text_1, support_class_num = TEST_SUP_CLS_NUM, attention_mask=attention_mask_1, labels=cls_1)                
                loss, output = new_model(text_1, text_2, cls_1, cls_2,\
                attention_mask_1, attention_mask_2, \
                TEST_SUP_CLS_NUM, use_global = False, \
                dataset_chosen = dataset_chosen[j], global_class_map = global_class_map.to(device), \
                soft_kmeans = soft_kmeans, \
                )
                # loss, output = new_model(text_1, text_2, support_class_num = TEST_SUP_CLS_NUM, prototypes = prototypes, soft_kmeans = soft_kmeans, embed_class_len = embed_class_len, attention_mask=attention_mask_1, labels=cls_1)
                val_loss += loss.mean().item()
                outputs.extend(output)
                # print(len(outputs))
  
            if test_example is not None:
                preds = [[id2label[support_class[int(x)]] for j,x in enumerate(y[1:orig_len[i]-1]) if int(cls[i][j + 1]) != -100] for i,y in enumerate(outputs)]
                gold = [[id2label[support_class[int(x)]] for x in y[1:orig_len[i]-1] if int(x) != -100] for i,y in enumerate(cls)]
            else:
                new_len = []
                for g in range(gpu_num):
                    new_len.extend([orig_len[i + g * large_group_num + TEST_SUP_CLS_NUM * SUP_PER_CLS] for i in range(int(len(outputs)/gpu_num))])
                preds = [[id2label[support_class[int(x)]] for j,x in enumerate(y[1:new_len[i]-1]) if int(cls_2[i][j + 1]) != -100] for i,y in enumerate(outputs)]
                gold = [[id2label[support_class[int(x)]] for x in y[1:new_len[i]-1] if int(x) != -100] for i,y in enumerate(cls_2)]

            # for pred in preds:
            #     for t,token in enumerate(pred):
            #         if len(token.split('I-')) == 2:
            #             if t == 0:
            #                 pred[t] = 'O'
            #                 continue
            #             else:
            #                 tag = token.split('I-')[1]
            #                 if len(pred[t-1]) == 1:
            #                     pred[t] = 'O'
            #                 else:
            #                     if tag != pred[t-1].split('-')[1]:
            #                         pred[t] = 'O'                       
                        
            
            # if j%20 == 0:
            #     print(j)
                # print(len(preds))
                # print(len(gold))
                # print([tokenizer.convert_ids_to_tokens(int(x)) for x in text[TEST_SUP_CLS_NUM * SUP_PER_CLS][1:orig_len[TEST_SUP_CLS_NUM * SUP_PER_CLS]-1]])
                # print(preds[0])
                # print(len(preds[0]))
                # print(gold[0])
                # print(len(gold[0]))
            bpred, bgold, bcrct, pred_span_per_type, gold_span_per_type, crct_span_per_type = batch_span_eval(preds, gold)
            total_pred += bpred
            total_gold += bgold
            total_crct += bcrct
            for x in pred_span_per_type:
                total_pred_per_type[x] += pred_span_per_type[x]
                total_gold_per_type[x] += gold_span_per_type[x]
                total_crct_per_type[x] += crct_span_per_type[x]
    microp = total_crct/total_pred if total_pred > 0 else 0
    micror = total_crct/total_gold if total_gold > 0 else 0
    microf1 = 2*microp*micror/(microp + micror) if (microp + micror) > 0 else 0
    microp_per_type, micror_per_type, microf1_per_type = {}, {}, {}
    for x in total_pred_per_type:
        microp_per_type[x] = total_crct_per_type[x]/total_pred_per_type[x] if total_pred_per_type[x] > 0 else 0
        micror_per_type[x] = total_crct_per_type[x]/total_gold_per_type[x] if total_gold_per_type[x] > 0 else 0
        microf1_per_type[x] = 2*microp_per_type[x]*micror_per_type[x]/(microp_per_type[x]+micror_per_type[x]) if (microp_per_type[x]+micror_per_type[x]) > 0 else 0

    return val_loss / test_num_data_point, microp, micror, microf1, microp_per_type, micror_per_type, microf1_per_type

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def weights_init_custom(model):
    init_layers = [9, 10, 11]
    dense_names = ["query", "key", "value", "dense"]
    layernorm_names = ["LayerNorm"]
    for name, module in model.bert.named_parameters():
        if any(f".{i}." in name for i in init_layers):
            if any(n in name for n in dense_names):
                if "bias" in name:
                    module.data.zero_()
                elif "weight" in name:
                    module.data.normal_(mean=0.0, std=model.config.initializer_range)
            elif any(n in name for n in layernorm_names):
                if "bias" in name:
                    module.data.zero_()
                elif "weight" in name:
                    module.data.fill_(1.0)
    return model



# torch.distributed.init_process_group(backend="nccl",init_method='tcp://localhost:23433',rank=0,world_size=1)


if __name__ == "__main__":

    # os.environ["CUDA_VISIBLE_DEVICES"]="3"
    device = torch.device("cuda")

    parser = argparse.ArgumentParser(description='main',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datapath', default='dataset')
    parser.add_argument('--dataset', default='conll2003')
    parser.add_argument('--train_text', default='train.words')
    parser.add_argument('--train_ner', default='train.ner')
    parser.add_argument('--test_text', default='test.words')
    parser.add_argument('--test_ner', default='test.ner')
    parser.add_argument('--few_shot_sets', default=1, type=int)
    parser.add_argument('--test_example', default='train.words')
    parser.add_argument('--test_example_ner', default='train.ner')
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--base_model', default='roberta', choices=['bert','roberta'])
    parser.add_argument('--train_sup_cls_num', default=4, type=int)
    parser.add_argument('--test_sup_cls_num', default=4, type=int)
    parser.add_argument('--sup_per_cls', default=5, type=int)
    parser.add_argument('--qur_per_cls', default=15, type=int)
    parser.add_argument('--o_sent_ratio', default=0.34, type=float)
    parser.add_argument('--max_seq_len', type=int)
    parser.add_argument('--episode_num', default=150, type=int)
    parser.add_argument('--soft_kmeans', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument('--tensorboard_path',default='./log')
    parser.add_argument('--local_rank',type=int)
    parser.add_argument('--use_gpu',default='0',type=str)
    parser.add_argument('--data_size',default='',type=str)
    parser.add_argument('--metric',default='euc',type=str)
    parser.add_argument('--norm', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--class_metric', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--instance_metric', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_multi_prototype', type=str2bool, nargs='?', const=True, default=False)
 
    parser.add_argument('--load_model', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--just_eval', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_example', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--model_name', default='save/conll_prototype_epoch')
    parser.add_argument('--reinit', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--load_model_name', default='save/conll_naiveft_bert_seq128_epoch')
    parser.add_argument('--load_checkpoint', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--load_dataset', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_dataset', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--train_dataset_file', default=None)
    parser.add_argument('--test_dataset_file', default=None)
    parser.add_argument('--label2ids', default=None)
    parser.add_argument('--id2labels', default=None)
    parser.add_argument('--train_label_sentence_dict',default=None)
    parser.add_argument('--test_label_sentence_dict',default=None)



    args = parser.parse_args()
    print(args)
    print("let's use ", torch.cuda.device_count(), "GPUs!")


    total_f1_scores = []
    train_text_file = args.train_text
    train_ner_file = args.train_ner
    test_example_file = args.test_example
    test_example_ner_file = args.test_example_ner
    for round in range(args.few_shot_sets):
        if '.' not in train_text_file:
            args.train_text = train_text_file + f'_{round}.words'
            args.train_ner = train_ner_file + f'_{round}.ner'
            args.test_example = test_example_file + f'_{round}.words'
            args.test_example_ner = test_example_ner_file + f'_{round}.ner'
        print(f"train text is {args.train_text}")

    
        datasets = args.dataset.split('_')
        print(datasets)
        train_texts = [os.path.join(args.datapath, dataset, args.train_text) for dataset in datasets]
        train_ners = [os.path.join(args.datapath, dataset, args.train_ner) for dataset in datasets]
        test_texts = [os.path.join(args.datapath, dataset, args.test_text) for dataset in datasets]
        test_ners = [os.path.join(args.datapath, dataset, args.test_ner) for dataset in datasets]
        test_examples = [os.path.join(args.datapath, dataset, args.test_example) for dataset in datasets]
        test_example_ners = [os.path.join(args.datapath, dataset, args.test_example_ner) for dataset in datasets]

        SUP_CLS_NUM = args.train_sup_cls_num
        TEST_SUP_CLS_NUM = args.test_sup_cls_num
        SUP_PER_CLS = args.sup_per_cls
        QUR_PER_CLS = args.qur_per_cls
        max_seq_len = args.max_seq_len
        o_sent_ratio = args.o_sent_ratio
        LOAD_MODEL = args.load_model
        JUST_EVAL = args.just_eval
        USE_EXAMPLE = args.use_example
        model_name = args.model_name
        base_model = args.base_model
        use_multi_prototype = args.use_multi_prototype
        # print(f"o_sent_ratio: {o_sent_ratio}")

        if base_model == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        elif base_model == 'bert':
            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        if not args.load_dataset:
            label2ids, id2labels = [], []
            processed_training_set, train_label_sentence_dicts, processed_test_set, test_label_sentence_dicts = [], [], [], []
            for train_text, train_ner, test_text, test_ner in zip(train_texts, train_ners, test_texts, test_ners):
                with open(train_ner) as fner, open(train_text) as f:
                    train_ner_tags, train_words = fner.readlines(), f.readlines() 
                with open(test_ner) as fner, open(test_text) as f:
                    test_ner_tags, test_words = fner.readlines(), f.readlines()      
                label2id, id2label = get_label_dict([train_ner_tags, test_ner_tags])
                train_ner_tags, train_words, train_label_sentence_dict = process_data(train_ner_tags, train_words, tokenizer, label2id, max_seq_len,base_model=base_model)
                test_ner_tags, test_words, test_label_sentence_dict = process_data(test_ner_tags, test_words, tokenizer, label2id, max_seq_len,base_model=base_model)

                sub_train_ = [[train_words[i], train_ner_tags[i]] for i in range(len(train_ner_tags))]
                sub_valid_ = [[test_words[i], test_ner_tags[i]] for i in range(len(test_ner_tags))]

                label2ids.append(label2id)
                id2labels.append(id2label)

                train_label_sentence_dicts.append(train_label_sentence_dict)
                test_label_sentence_dicts.append(test_label_sentence_dict)

                print([(x,len(train_label_sentence_dict[x])) for x in train_label_sentence_dict])
                processed_training_set.append(sub_train_) 
                processed_test_set.append(sub_valid_) 
            dataset_label_nums = [len(x) for x in label2ids]
            print(f"dataset label nums: {dataset_label_nums}")
            train_num_data_point = sum([len(sub_train_) for sub_train_ in processed_training_set])
            print(f"number of all training data points: {train_num_data_point}")
            test_num_data_point = sum([len(sub_train_) for sub_train_ in processed_test_set])
            print(f"number of all testing data points: {test_num_data_point}")
            if args.save_dataset:
                np.save('roberta_processed_training_set',processed_training_set)
                np.save('roberta_processed_test_set',processed_test_set)
                np.save('train_label_sentence_dict',train_label_sentence_dicts)
                np.save('test_label_sentence_dict',test_label_sentence_dicts)
                np.save('label2ids',label2ids)
                np.save('id2labels',id2labels)
                print("finish saving datasets!")

        if args.load_dataset:
            start_time = time.time()
            print("start loading training dataset!")
            processed_training_set = np.load(args.train_dataset_file,allow_pickle=True)
            print("start loading test dataset!")
            processed_test_set = np.load(args.test_dataset_file,allow_pickle=True)
            print("start loading label ids!")
            label2ids = np.load(args.label2ids,allow_pickle=True)
            id2labels = np.load(args.id2labels,allow_pickle=True)
            print("start loading label sentence dicts!")
            train_label_sentence_dicts = np.load(args.train_label_sentence_dict,allow_pickle=True)
            test_label_sentence_dicts = np.load(args.test_label_sentence_dict,allow_pickle=True)
            dataset_label_nums = [len(x) for x in label2ids]
            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60
            print(f"finish loading datasets! | time in {mins} minutes, {secs} seconds")
            train_num_data_point = sum([len(sub_train_) for sub_train_ in processed_training_set])
            print(f"number of all training data points: {train_num_data_point}")
            test_num_data_point = sum([len(sub_train_) for sub_train_ in processed_test_set])
            print(f"number of all testing data points: {test_num_data_point}")

        if args.use_example == True:
            TEST_SUP_CLS_NUM = int(len(label2ids[0]) / 2)
        if args.dataset == 'conll2003' or args.dataset == 'wikigold':
            SUP_CLS_NUM = 4
            TEST_SUP_CLS_NUM = 4
        print(f"sup_cls_num {SUP_CLS_NUM} test_sup_cls_num {TEST_SUP_CLS_NUM}")
            

        # label2ids, id2labels = [], []
        # train_words_list, train_ner_tags_list= [], []
        # processed_training_set, train_label_sentence_dicts = [], []
        # if not JUST_EVAL: 
        #     for train_text, train_ner in zip(train_texts, train_ners):
        #         with open(train_ner) as fner, open(train_text) as f:
        #             train_ner_tags, train_words = fner.readlines(), f.readlines()       
        #         label2id, id2label = get_label_dict([train_ner_tags])
        #         train_ner_tags, train_words, train_label_sentence_dict = process_data(train_ner_tags, train_words, tokenizer, label2id, max_seq_len)
        #         sub_train_ = [[train_words[i], train_ner_tags[i]] for i in range(len(train_ner_tags))]
        #         train_ner_tags_list.append(train_ner_tags)
        #         train_words_list.append(train_words)
        #         label2ids.append(label2id)
        #         id2labels.append(id2label)
        #         train_label_sentence_dicts.append(train_label_sentence_dict)
        #         print([(x,len(train_label_sentence_dict[x])) for x in train_label_sentence_dict])
        #         processed_training_set.append(sub_train_) 
        #     print(f"dataset label nums: {[len(x) for x in label2ids]}")

        # with open(test_ner) as fner, open(test_text) as f:
        #     test_ner_tags, test_words = fner.readlines(), f.readlines()
        # test_label2id, test_id2label = get_label_dict([test_ner_tags])
        # test_ner_tags, test_words, test_label_sentence_dict = process_data(test_ner_tags, test_words, tokenizer, test_label2id, max_seq_len)
        # sub_valid_ = [[test_words[i], test_ner_tags[i]] for i in range(len(test_ner_tags))]
        
        if USE_EXAMPLE:
            with open(os.path.join(args.datapath, args.dataset, args.test_example_ner)) as fner, open(os.path.join(args.datapath, args.dataset, args.test_example)) as f:
                test_example_ner_tags, test_example_words = fner.readlines(), f.readlines()
            # test_label2id, test_id2label = get_label_dict([test_ner_tags])
            test_example_ner_tags, test_example_words, _ = process_data(test_example_ner_tags, test_example_words, tokenizer, label2ids[0], max_seq_len,base_model=base_model)
            sub_example_ = [[test_example_words[i], test_example_ner_tags[i]] for i in range(len(test_example_ner_tags))]
            print(f"sub example num: {len(sub_example_)}")

        if LOAD_MODEL:
            if 'checkpoint' not in args.load_model_name:
                state = torch.load(args.load_model_name)
                if base_model == 'roberta':
                    model = RobertaNER.from_pretrained('roberta-base', output_attentions=False, output_hidden_states=False, \
                    use_global = False, dataset_label_nums=dataset_label_nums, support_per_class = SUP_PER_CLS, cuda_device = 0, use_bias = False)
                elif base_model == 'bert':
                    model = BertNER.from_pretrained('bert-base-cased', output_attentions=False, output_hidden_states=False, \
                    use_global = False, dataset_label_nums=dataset_label_nums, support_per_class = SUP_PER_CLS, cuda_device = 0, use_bias = False)
                pretrained_dict = state.state_dict()
                model_dict = model.state_dict()
                pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict and 'class_metric' not in k}
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)
                model.dataset_label_nums = dataset_label_nums
                print(model.config.hidden_size)
                # print(model.roberta.config)
                # model.roberta.save_pretrained('save/pretrained_models/prototype/')
                # exit(1)
            else:
                state = torch.load(args.load_model_name)
                if base_model == 'roberta':
                    model = RobertaNER.from_pretrained('roberta-base', output_attentions=False, output_hidden_states=False, \
                    use_global = False, dataset_label_nums=dataset_label_nums, support_per_class = SUP_PER_CLS, cuda_device = 0, use_bias = False)
                elif base_model == 'bert':
                    model = BertNER.from_pretrained('bert-base-cased', output_attentions=False, output_hidden_states=False, \
                    use_global = False, dataset_label_nums=dataset_label_nums, support_per_class = SUP_PER_CLS, cuda_device = 0, use_bias = False)
                # pretrained_dict = state['state_dict']
                # model_dict = model.state_dict()
                # pretrained_dict = {k.split('module.')[1]:v for k,v in pretrained_dict.items() if k.split('module.')[1] in model_dict}
                # model_dict.update(pretrained_dict)
                # model.load_state_dict(model_dict)
                model.dataset_label_nums = dataset_label_nums
                model.load_state_dict(state['state_dict'])
                print(model.config.hidden_size)
        else:
            if base_model == 'roberta':
                model = RobertaNER.from_pretrained('roberta-base', output_attentions=False, output_hidden_states=False, \
                use_global = False, dataset_label_nums=dataset_label_nums, support_per_class = SUP_PER_CLS, cuda_device = 0, use_bias = False)
            elif base_model == 'bert':
                model = BertNER.from_pretrained('bert-base-cased', output_attentions=False, output_hidden_states=False, \
                use_global = False, dataset_label_nums=dataset_label_nums, support_per_class = SUP_PER_CLS, cuda_device = 0, use_bias = False)
        if args.reinit:
            model = weights_init_custom(model)
        model.to(device)
        if torch.cuda.device_count() > 1:
            use_multi_gpus = True
            print("let's use ", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
            # model = torch.nn.parallel.DistributedDataParallel(model,find_unused_parameters=True)
        else:
            use_multi_gpus = False 
        gpu_num = torch.cuda.device_count()
        

        
        N_EPOCHS = args.epoch
        EPISODE_NUM = args.episode_num
        if args.use_example == True:
            if '5shot' in args.data_size:
                EPISODE_NUM = int (train_num_data_point / (SUP_CLS_NUM * (SUP_PER_CLS + QUR_PER_CLS) ) * 2  )
            else:
                EPISODE_NUM = int (train_num_data_point / (SUP_CLS_NUM * (SUP_PER_CLS + QUR_PER_CLS) )) + 1
        print(f"episode num: {EPISODE_NUM}")
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        # num_training_steps = N_EPOCHS * int(len(sub_train_) / BATCH_SIZE)
        if 'shotxxx' in args.train_text:
            num_training_steps = N_EPOCHS * int(EPISODE_NUM / gpu_num) * SUP_CLS_NUM
        else:
            num_training_steps = N_EPOCHS * int(EPISODE_NUM / gpu_num)
        num_warmup_steps = int(args.warmup_proportion * num_training_steps)
        print(f"num training steps: {num_training_steps}")
        print(f"num warmup steps: {num_warmup_steps}")
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=True)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        SOFT_KMEANS = args.soft_kmeans
        writer = SummaryWriter(args.tensorboard_path)

        if not USE_EXAMPLE:
            sub_example_ = None
        
        if JUST_EVAL:
            #evaluate
            if USE_EXAMPLE:
                # for eval_model_epoch in range(10):
                print(f"epoch: {eval_model_epoch+1}")
                valid_loss, microp, micror, microf1, microp_per_type, micror_per_type, microf1_per_type = test(sub_valid_, eval_model_epoch, test_label_sentence_dict, soft_kmeans = SOFT_KMEANS, test_all_data = True, test_example = sub_example_)
                print(f'\tLoss: {valid_loss:.4f}(val)\t|\tPrec: {microp * 100:.1f}%(val)\t|\tRecall: {micror * 100:.1f}%(val)\t|\tF1: {microf1 * 100:.1f}%(val)')
                print(f'precision per type: {microp_per_type}')
                print(f'recall per type: {micror_per_type}')
                print(f'f1-score per type: {microf1_per_type}')
            else:
                valid_loss, microp, micror, microf1, microp_per_type, micror_per_type, microf1_per_type = test(sub_valid_, eval_model_epoch, test_label_sentence_dict, soft_kmeans = SOFT_KMEANS, test_all_data = True)
                print(f'\tLoss: {valid_loss:.4f}(val)\t|\tPrec: {microp * 100:.1f}%(val)\t|\tRecall: {micror * 100:.1f}%(val)\t|\tF1: {microf1 * 100:.1f}%(val)')
                print(f'precision per type: {microp_per_type}')
                print(f'recall per type: {micror_per_type}')
                print(f'f1-score per type: {microf1_per_type}')

        else:
            with open(f'../results/prototype_{args.dataset}_{args.data_size}.txt','a') as fout:
                fout.write('\n')
                fout.write(str(args))
                fout.write('\n')
            
            start_time = time.time()
            if USE_EXAMPLE:
                valid_loss, microp, micror, microf1, microp_per_type, micror_per_type, microf1_per_type = test(processed_test_set, 0, id2labels, test_label_sentence_dicts, soft_kmeans = SOFT_KMEANS, test_all_data = True, test_example = sub_example_)
            else:
                valid_loss, microp, micror, microf1, microp_per_type, micror_per_type, microf1_per_type = test(processed_test_set, 0, id2labels, test_label_sentence_dicts, soft_kmeans = SOFT_KMEANS, test_all_data = False, test_example = sub_example_)
            print(f'\tLoss: {valid_loss:.4f}(val)\t|\tPrec: {microp * 100:.1f}%(val)\t|\tRecall: {micror * 100:.1f}%(val)\t|\tF1: {microf1 * 100:.1f}%(val)')
            secs = int(time.time() - start_time)
            mins = secs / 60
            secs = secs % 60

            print("Zero-shot result | time in %d minutes, %d seconds" %(mins, secs))
            with open(f'../results/prototype_{args.dataset}_{args.data_size}.txt','a') as fout:
                fout.write(f'{microf1} ')

            for epoch in range(N_EPOCHS):

                start_time = time.time()
                train_loss, microp, micror, microf1 = train_func(processed_training_set, epoch, tokenizer, id2labels, train_label_sentence_dicts, soft_kmeans = SOFT_KMEANS)

                
                print(f'\tLoss: {train_loss:.4f}(train)\t|\tPrec: {microp * 100:.1f}%(train)\t|\tRecall: {micror * 100:.1f}%(train)\t|\tF1: {microf1 * 100:.1f}%(train)')
                # if use_multi_gpus:
                #     torch.save(model.module, model_name+str(epoch + 1)+'.pt')
                # else:
                #     torch.save(model, model_name+str(epoch + 1)+'.pt')
                # writer.add_scalars( 'prototype_'+args.dataset+'/'+model_name.split('/')[1]+' train', {'F1-score': microf1, 'Precision': microp, 'Recall': micror}, epoch+1)
                # writer.add_scalar('prototype_'+args.dataset+'/'+model_name.split('/')[1]+' Loss train',train_loss,epoch+1)

                # valid_loss, microp, micror, microf1, microp_per_type, micror_per_type, microf1_per_type = test(sub_valid_, epoch, test_label_sentence_dict, soft_kmeans = SOFT_KMEANS)
                if USE_EXAMPLE:
                    valid_loss, microp, micror, microf1, microp_per_type, micror_per_type, microf1_per_type = test(processed_test_set, epoch, id2labels, test_label_sentence_dicts, soft_kmeans = SOFT_KMEANS, test_all_data = True, test_example = sub_example_)
                else:
                    valid_loss, microp, micror, microf1, microp_per_type, micror_per_type, microf1_per_type = test(processed_test_set, epoch, id2labels, test_label_sentence_dicts, soft_kmeans = SOFT_KMEANS, test_all_data = False, test_example = sub_example_)
                print(f'\tLoss: {valid_loss:.4f}(val)\t|\tPrec: {microp * 100:.1f}%(val)\t|\tRecall: {micror * 100:.1f}%(val)\t|\tF1: {microf1 * 100:.1f}%(val)')
                print(f'microp per type: {microp_per_type} \nmicror_per_type: {micror_per_type} \nmicrof1_per_type: {microf1_per_type}')
                secs = int(time.time() - start_time)
                mins = secs / 60
                secs = secs % 60

                print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
                with open(f'../results/prototype_{args.dataset}_{args.data_size}.txt','a') as fout:
                    fout.write(f'{microf1} ')
                print(f'precision per type: {microp_per_type}')
                print(f'recall per type: {micror_per_type}')
                print(f'f1-score per type: {microf1_per_type}')
                print(args.model_name)

        total_f1_scores.append(microf1)
        # with open(f'../results/prototype_{args.dataset}_{args.data_size}.txt','a') as fout:
        #     fout.write(f'\n')

    print(f"f1 scores: {total_f1_scores} \n average f1 scores: {sum(total_f1_scores)/len(total_f1_scores)}")




        