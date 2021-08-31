"""
this file modified from the word_language_model example
"""
import os
import random
import torch
import json
import truecase
import re
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
from collections import Counter, defaultdict

from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import normalize
from sample_few_shot import get_label_dict

# moved to train_xxx.py
# def generate_batch(batch):
#     text = [F.pad(torch.tensor(x[0]), (0,max_seq_len-len(x[0])), "constant", 1) for x in batch] # batch_size * max_seq_len 
#     text = pad_sequence(text, batch_first = True)
#     attention_mask = [torch.cat((torch.ones_like(torch.tensor(x[0])), torch.zeros(max_seq_len-len(x[0]), dtype=torch.int64)), dim=0)
#         if len(x[0]) < max_seq_len else torch.ones_like(torch.tensor(x[0]))[:max_seq_len] for x in batch]
#     attention_mask = pad_sequence(attention_mask, batch_first = True)
#     # print(batch[0][1])
#     # print(len(batch[0][1]))
#     label = [F.pad(torch.tensor(x[1]), (0,max_seq_len-len(x[1])), "constant", -100) for x in batch]
#     label = pad_sequence(label, batch_first = True)
#     orig_len = [len(x[0]) for x in batch]

#     return text, attention_mask, label, orig_len

def generate_episode(data_list, epoch, EPISODE_NUM, max_seq_len, sup_cls_num, SUP_PER_CLS, QUR_PER_CLS, id2labels, label_sentence_dicts, 
    gpu_num = 1, use_multipledata = True, train_other_queries = False, train_other_labels = True, uniform = True, all_o_sent = 0, unsup_data_ = None, unsup_query = 0):
    if not use_multipledata:
        data_list = [data_list]
        id2labels = [id2labels]
        label_sentence_dicts = [label_sentence_dicts]

    processed_data = []
    episode_num = EPISODE_NUM
    logging.info(f"Generating {episode_num} episodes, starting from {epoch*episode_num}!")
    data_len_list = np.array([len(x) for x in data_list])
    data_chosen_list = []
    for epi in range(int(episode_num/gpu_num)):
        # if epi % 5 == 0:
        #     print(epi)
        start_epi_time = time.time()
        random.seed(epi + epoch*int(episode_num/gpu_num))
        np.random.seed(epi + epoch*int(episode_num/gpu_num))
        # print(data_len_list)
        
        if not uniform:
            dataset_chosen = np.random.multinomial(1, data_len_list/sum(data_len_list), size=1)[0]
        else:
            dataset_chosen = np.random.multinomial(1, np.ones(len(data_len_list))/len(data_len_list), size=1)[0]
        
        dataset_chosen = int(np.where(dataset_chosen == 1)[0])
        data_chosen_list.append(dataset_chosen)
        data, label_sentence_dict, id2label = data_list[dataset_chosen], label_sentence_dicts[dataset_chosen], id2labels[dataset_chosen]
        # print(dataset_chosen)
        # print(len(label_sentence_dict))
        # print(len(id2label))
        label_in_dict = set([int((x+1)/2) for x in label_sentence_dict if len(label_sentence_dict[x]) > 0])
        # print(len(label_in_dict))
        label_in_dict = label_in_dict & set(range(int((len(id2label)+1) / 2))[1:])
        # print(len(label_in_dict))
        support_class_root = np.random.choice(list(label_in_dict), size = sup_cls_num, replace=False).tolist() 
        # print(f"random seed: {epi + epoch*int(episode_num/gpu_num)} support class root: {support_class_root}")     
        
        support_class = [0]
        for cls in support_class_root:
            support_class.append(2 * cls - 1)
            support_class.append(2 * cls)
        if train_other_labels:
            inv_support_classes = defaultdict(int) # size: SUP_CLS_NUM * 2
        else:
            inv_support_classes = defaultdict(lambda:-100)
        for i,cls in enumerate(support_class):
            inv_support_classes[cls] = i
        inv_support_classes[-100]=-100
        idx = 0
        support_class = {idx:0}  # size: SUP_CLS_NUM * 2
        support_class[-100] = -100
        for cls in support_class_root:
            idx += 1
            support_class[idx] = 2 * cls - 1
            idx += 1
            support_class[idx] = 2 * cls
        if epi == 0:
            logging.info(f"{support_class}")
            logging.info(f"{inv_support_classes}")
        text = []
        attention_mask = []
        label = []
        orig_len = []
        epi_iter_num = []
        sent_idx = {}
        for epi_iter in range(gpu_num):
            # print("sampling sentences")
            # start_time = time.time()
            for cls in range(int((len(id2label)+1)/2))[1:]:
                if train_other_queries == False and cls in support_class_root:
                    # print(cls)
                    while True:            
                        if all_o_sent > 0:
                            if len(label_sentence_dict[cls*2-1]) < SUP_PER_CLS + QUR_PER_CLS - all_o_sent:
                                # sent_idx[cls] = np.random.choice(label_sentence_dict[cls * 2 - 1], size = SUP_PER_CLS + QUR_PER_CLS - all_o_sent, replace = True)
                                sent_idx[cls] = random.choices(label_sentence_dict[cls * 2 - 1], k = SUP_PER_CLS + QUR_PER_CLS - all_o_sent)
                                sent_idx[cls] = np.append(sent_idx[cls], np.array(random.sample(label_sentence_dict[0], k=all_o_sent))  )
                            else:
                                # sent_idx[cls] = np.random.choice(label_sentence_dict[cls * 2 - 1], size = SUP_PER_CLS + QUR_PER_CLS - all_o_sent, replace = False)
                                sent_idx[cls] = random.sample(label_sentence_dict[cls * 2 - 1], k = SUP_PER_CLS + QUR_PER_CLS - all_o_sent)
                                sent_idx[cls] = np.append(sent_idx[cls], np.array(random.sample(label_sentence_dict[0], k=all_o_sent))   )
                            # if epi == 0:
                            #     print(id2label[cls*2-1])
                            #     print(sent_idx[cls])
                        else:
                            if len(label_sentence_dict[cls*2-1]) < SUP_PER_CLS + QUR_PER_CLS:
                                sent_idx[cls] = np.array(random.choices(label_sentence_dict[cls * 2 - 1], k = SUP_PER_CLS + QUR_PER_CLS))
                            else:
                                sent_idx[cls] = np.array(random.sample(label_sentence_dict[cls * 2 - 1], k = SUP_PER_CLS + QUR_PER_CLS))
                            # if epi == 0:
                            #     print(id2label[cls*2-1])
                            #     print(sent_idx[cls])
                        # if query has I-cls but support does not, we resample the sentences.
                        # print(f"all_o_sent {all_o_sent} sent_idx_len {len(sent_idx[cls])}")
                        sup_idx = sent_idx[cls][:SUP_PER_CLS]
                        qur_idx = sent_idx[cls][SUP_PER_CLS:]
                        if np.any([x in label_sentence_dict[cls * 2] for x in sup_idx]) == False and np.any([x in label_sentence_dict[cls * 2] for x in qur_idx]) == True:
                            continue
                        else:
                            break
                if train_other_queries:
                    sent_idx[cls] = np.random.choice(label_sentence_dict[cls * 2 - 1], size = QUR_PER_CLS, replace = False)
            # print(sent_idx)
            # secs = int(time.time() - start_time)
            # mins = secs / 60
            # print(f"finish sampling one episode! | time in {mins} minutes, {secs} seconds")
            # print("preparing one episode!")
            # start_time = time.time()
            for cls in support_class_root:
                # print(cls)
                text.extend([F.pad(torch.tensor(data[x][0]), (0,max_seq_len-len(data[x][0])), "constant", 1) for x in sent_idx[cls][:SUP_PER_CLS]])           
                attention_mask.extend([torch.cat((torch.ones_like(torch.tensor(data[x][0])), torch.zeros(max_seq_len-len(data[x][0]), \
                    dtype=torch.int64)), dim=0) \
                    if len(data[x][0]) < max_seq_len else torch.ones_like(torch.tensor(data[x][0]))[:max_seq_len] for x in sent_idx[cls][:SUP_PER_CLS]])
                label.extend([F.pad(torch.tensor([inv_support_classes[y] for y in data[x][1]]), (0,max_seq_len-len(data[x][1])), "constant", -100) \
                    for x in sent_idx[cls][:SUP_PER_CLS]])
                orig_len.extend([len(data[x][0]) for x in sent_idx[cls][:SUP_PER_CLS]])
   
            for cls in sent_idx:
                text.extend([F.pad(torch.tensor(data[x][0]), (0,max_seq_len-len(data[x][0])), "constant", 1) for x in sent_idx[cls][-QUR_PER_CLS:]])           
                attention_mask.extend([torch.cat((torch.ones_like(torch.tensor(data[x][0])), torch.zeros(max_seq_len-len(data[x][0]), \
                    dtype=torch.int64)), dim=0) \
                    if len(data[x][0]) < max_seq_len else torch.ones_like(torch.tensor(data[x][0]))[:max_seq_len] for x in sent_idx[cls][-QUR_PER_CLS:]])
                label.extend([F.pad(torch.tensor([inv_support_classes[y] for y in data[x][1]]), (0,max_seq_len-len(data[x][1])), "constant", -100) \
                    for x in sent_idx[cls][-QUR_PER_CLS:]])
                orig_len.extend([len(data[x][0]) for x in sent_idx[cls][-QUR_PER_CLS:]])
                    
                
            # epi_iter_num.extend([epi_iter for x in range(sup_cls_num * SUP_PER_CLS + sup_cls_num * QUR_PER_CLS)])
            # secs = int(time.time() - start_time)
            # mins = secs / 60
            # print(f"finish preparing one episode! | time in {mins} minutes, {secs} seconds")    
        text = pad_sequence(text, batch_first = True)
        attention_mask = pad_sequence(attention_mask, batch_first = True)
        label = pad_sequence(label, batch_first = True)
        # print(len(text))
        processed_data.append((text, attention_mask, label, orig_len, support_class, inv_support_classes))
        # secs = int(time.time() - start_epi_time)
        # mins = secs / 60
        # print(f"finish a whole batch! | time in {mins} minutes, {secs} seconds")

    logging.info(f"dataset chosen list: {data_chosen_list}")

    return processed_data, data_chosen_list

def generate_testing_episode(data, episode_num, batch_size, max_seq_len, sup_cls_num, SUP_PER_CLS, id2label, label_sentence_dict, train_other_queries = False, train_other_labels = True):
    processed_data = []
    np.random.seed(episode_num)
    support_class = {}
    inv_support_classes = {}
    for cls in range(len(id2label)):
        support_class[cls]=cls
        inv_support_classes[cls]=cls
    support_class[-100]=-100
    inv_support_classes[-100]=-100
    sent_idx = {}

    #generating support set
    for cls in range(int((len(id2label)+1)/2))[1:]:
        sent_idx[cls] = np.random.choice(label_sentence_dict[cls * 2 - 1], size = SUP_PER_CLS, replace = False)    
    for batch in range((len(data) + batch_size - 1) // batch_size): 
        text = []
        attention_mask = []
        label = []
        orig_len = []   
        if SUP_PER_CLS > 0:      
            for cls in sent_idx:
                # print(cls)
                text.extend([F.pad(torch.tensor(data[x][0]), (0,max_seq_len-len(data[x][0])), "constant", 1) for x in sent_idx[cls][:SUP_PER_CLS]])           
                attention_mask.extend([torch.cat((torch.ones_like(torch.tensor(data[x][0])), torch.zeros(max_seq_len-len(data[x][0]), \
                    dtype=torch.int64)), dim=0) \
                    if len(data[x][0]) < max_seq_len else torch.ones_like(torch.tensor(data[x][0]))[:max_seq_len] for x in sent_idx[cls][:SUP_PER_CLS]])
                label.extend([F.pad(torch.tensor([inv_support_classes[y] for y in data[x][1]]), (0,max_seq_len-len(data[x][1])), "constant", -100) \
                    for x in sent_idx[cls][:SUP_PER_CLS]])
                orig_len.extend([len(data[x][0]) for x in sent_idx[cls][:SUP_PER_CLS]])
        
        text.extend([F.pad(torch.tensor(data[x][0]), (0,max_seq_len-len(data[x][0])), "constant", 1) for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data))) ])           
        attention_mask.extend([torch.cat((torch.ones_like(torch.tensor(data[x][0])), torch.zeros(max_seq_len-len(data[x][0]), \
            dtype=torch.int64)), dim=0) \
            if len(data[x][0]) < max_seq_len else torch.ones_like(torch.tensor(data[x][0]))[:max_seq_len] for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data)))])
        label.extend([F.pad(torch.tensor([inv_support_classes[y] for y in data[x][1]]), (0,max_seq_len-len(data[x][1])), "constant", -100) \
            for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data)))])
        orig_len.extend([len(data[x][0]) for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data)))])
        
        text = pad_sequence(text, batch_first = True)
        attention_mask = pad_sequence(attention_mask, batch_first = True)
        label = pad_sequence(label, batch_first = True)
        # print(len(text))
        processed_data.append((text, attention_mask, label, orig_len, support_class, inv_support_classes))

    return processed_data

def generate_unsup_episode(data, episode_num, batch_size, max_seq_len, sup_cls_num, SUP_PER_CLS, id2label):
    processed_data = []
    np.random.seed(episode_num)
    support_class = {}
    inv_support_classes = {}
    for cls in range(len(id2label)):
        support_class[cls]=cls
        inv_support_classes[cls]=cls
    support_class[-100]=-100
    inv_support_classes[-100]=-100
   
    for batch in range((len(data) + batch_size - 1) // batch_size): 
        text = []
        attention_mask = []
        label = []
        orig_len = []   
        
        text.extend([F.pad(torch.tensor(data[x][0]), (0,max_seq_len-len(data[x][0])), "constant", 1) for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data))) ])           
        attention_mask.extend([torch.cat((torch.ones_like(torch.tensor(data[x][0])), torch.zeros(max_seq_len-len(data[x][0]), \
            dtype=torch.int64)), dim=0) \
            if len(data[x][0]) < max_seq_len else torch.ones_like(torch.tensor(data[x][0]))[:max_seq_len] for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data)))])
        label.extend([data[x][1] for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data)))])
        orig_len.extend([len(data[x][0]) for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data)))])
        
        text = pad_sequence(text, batch_first = True)
        attention_mask = pad_sequence(attention_mask, batch_first = True)
        label = pad_sequence(label, batch_first = True)
        # print(f"shape of text {text.shape}")
        # print(f"shape of prob {label.shape}")
        # print(len(text))
        processed_data.append((text, attention_mask, label, orig_len, support_class, inv_support_classes))

    return processed_data


def generate_testing_example(data, batch_size, max_seq_len, sup_cls_num, id2label, train_other_queries = False, train_other_labels = True):
    processed_data = []
    support_class = {}
    inv_support_classes = {}
    for cls in range(len(id2label)):
        support_class[cls]=cls
        inv_support_classes[cls]=cls
    support_class[-100]=-100
    inv_support_classes[-100]=-100
    sent_idx = {}
  
    for batch in range((len(data) + batch_size - 1) // batch_size): 
        text = []
        attention_mask = []
        label = []
        orig_len = []         
        
        text.extend([F.pad(torch.tensor(data[x][0]), (0,max_seq_len-len(data[x][0])), "constant", 1) for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data))) ])           
        attention_mask.extend([torch.cat((torch.ones_like(torch.tensor(data[x][0])), torch.zeros(max_seq_len-len(data[x][0]), \
            dtype=torch.int64)), dim=0) \
            if len(data[x][0]) < max_seq_len else torch.ones_like(torch.tensor(data[x][0]))[:max_seq_len] for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data)))])
        label.extend([F.pad(torch.tensor([inv_support_classes[y] for y in data[x][1]]), (0,max_seq_len-len(data[x][1])), "constant", -100) \
            for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data)))])
        orig_len.extend([len(data[x][0]) for x in range(batch * batch_size, min((batch + 1) * batch_size, len(data)))])
        
        text = pad_sequence(text, batch_first = True)
        attention_mask = pad_sequence(attention_mask, batch_first = True)
        label = pad_sequence(label, batch_first = True)
        # print(len(text))
        processed_data.append((text, attention_mask, label, orig_len, support_class, inv_support_classes))

    return processed_data

def truecase_sentence(tokens):
    word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]
    # print(word_lst)
    # print(lst)

    if len(lst) and len(lst) == len(word_lst):
        parts = truecase.get_true_case(' '.join(lst)).split()

        # the trucaser have its own tokenization ...
        # skip if the number of word dosen't match
        if len(parts) != len(word_lst): return tokens

        for (w, idx), nw in zip(word_lst, parts):
            tokens[idx] = nw
    return tokens
    


def process_data(ner_tags, sents, tokenizer, label2id, max_sequence_len, base_model='roberta', lower=False, use_truecase=False):
    logging.info("Start processing raw data!")
    sent_words, sent_wpcs, sent_tags = [], [], []
    max_seq_len = 0
    label_sentence_dict = defaultdict(list)
    idx = 0
    exceed_num = 0
    good_sent = 0
    bad_sent = 0
    for k,line in enumerate(sents):
        sent = line.strip()
        # N.B. don't need to lower, since tokenizer does it automatically
        # if lower:
        #     sent = sent.lower()
        words = sent.split(' ')
        tags = ner_tags[k].strip().split()
        # tags = [x if len(x.split('.')) == 1 else x.split('-')[0]+'-'+x.split('.')[-1] for x in tags]
        # for t in tags:
        #     if t not in label2id:
        #         print(ner_tags[k])
        #         print(t)
        if len(tags) != len(words):
            print(tags)
            print(words)
        assert len(tags) == len(words)
        if use_truecase:
            last_words = [x for x in words]
            words = truecase_sentence(words)
            if last_words != words:
                print(last_words)
                print(words)
            sent = ' '.join(words)
        if base_model == 'bert':
            wpcs = ["[CLS]"]
            wpcs.extend(tokenizer.tokenize(sent))
            wpcs.append("[SEP]")
        elif base_model == 'roberta':
            wpcs = ['<s>']
            wpcs.extend(tokenizer.tokenize(sent))
            wpcs.append('</s>')
        if len(wpcs) > max_sequence_len:
            exceed_num += 1
        if len(wpcs) > max_seq_len:
            max_seq_len = len(wpcs)
        try:
            # print(words)
            # print(wpcs)
            aligns = align_wpcs(words, wpcs, base_model = base_model, lower=lower) 
        except IndexError:
            print("index error")
            print(words)
            print(wpcs)
            bad_sent += 1
            continue
        except AssertionError:
            print("ignoring one")
            print("index error")
            print(words)
            print(wpcs)
            bad_sent += 1
            continue
        sent_wpcs.append(tokenizer.convert_tokens_to_ids(wpcs))
        # sent_words.append(aligns)
        # [(1, 2), (2, 4), (4, 5)]
        sent_tag = [-100]
        for i in range(len(aligns)):
            sent_tag.extend([label2id[tags[i]] if j == 0 else -100 for j in range(aligns[i][1]-aligns[i][0])])
            # if tags[i] == 'O':
            #     sent_tag.extend([label2id[tags[i]] for j in range(aligns[i][1]-aligns[i][0])])
            # else:
            #     sent_tag.extend([label2id[tags[i]] if j == 0 else label2id['I-'+tags[i].split('-')[1]] for j in range(aligns[i][1]-aligns[i][0])])
        sent_tag.append(-100)
        assert len(sent_tag) == len(wpcs)
        sent_tags.append(sent_tag)
        good_sent += 1

        # add dictionary: tag to sentence index
        tags_set = set(tags)
        all_O = True
        for t in tags_set:
            if label2id[t] != 0:
                all_O = False
                # if k == 0:
                #     print(t)
                label_sentence_dict[label2id[t]].append(idx)
        if all_O:
            label_sentence_dict[0].append(idx)
        # if k == 0:
        #     print(label_sentence_dict)
        idx += 1
    # out_file = open('label_sentence_dict','w')
    # json.dump(label_sentence_dict, out_file)

    logging.info(f"max seq length: {max_seq_len}")
    logging.info(f"truncate ratio: {exceed_num/len(sents)}")
    logging.info(f"good sent: {good_sent} bad sent: {bad_sent}")
    return sent_tags, sent_wpcs, label_sentence_dict

def align_wpcs(words, wpcs, base_model = 'roberta', lower=False):
    """
    maps each word idx to start and end idx w/ in wpcs.
    assumes wpcs is padded on either end with CLS and SEP
    """
    if base_model == 'roberta':
        align = []
        curr_start, curr_wrd = 1, 0 # start at 1, b/c of CLS
        buf = []
        for i in range(1, len(wpcs)-1): # ignore [SEP] final token
            # strpd = wpcs[i][2:] if wpcs[i].startswith("##") else wpcs[i]
            # strpd = wpcs[i][1:] if wpcs[i].startswith("Ġ") else wpcs[i]
            # buf.append(strpd)
            # fwrd = ''.join(buf)
            # wrd = words[curr_wrd].lower() if lower else words[curr_wrd]
            # if fwrd == wrd or fwrd == "[UNK]":
            if wpcs[i].startswith("Ġ") or wpcs[i] == "<unk>":
                align.append((curr_start, i))
                curr_start = i
        align.append((curr_start, len(wpcs)-1))
    elif base_model == 'bert':
        align = []
        curr_start, curr_wrd = 1, 0 # start at 1, b/c of CLS
        buf = []
        for i in range(1, len(wpcs)-1): # ignore [SEP] final token
            strpd = wpcs[i][2:] if wpcs[i].startswith("##") else wpcs[i]
            buf.append(strpd)
            #buf.append(wpcs[i].lstrip('##'))
            fwrd = ''.join(buf)
            wrd = words[curr_wrd].lower() if lower else words[curr_wrd]
            if fwrd == wrd or fwrd == "[UNK]":
                align.append((curr_start, i+1))
                curr_start = i+1
                curr_wrd += 1
                buf = []
        
    if len(align) != len(words):
        pass
        # print(words)
        # print(wpcs)
        # print(align)
    assert len(align) == len(words)
    return align
# words = ['Xinhua', 'News', 'Agency', ',', 'Beijing', ',', 'September', '1st']
# wpcs = ['<s>', 'X', 'in', 'hua', 'ĠNews', 'ĠAgency', 'Ġ,', 'ĠBeijing', 'Ġ,', 'ĠSeptember', 'Ġ1', 'st', '</s>']
# ['Book', 'the', 'Labworth', 'Café', 'in', 'Ethiopia', 'for', 'six', 'people.']
# ['<s>', 'Book', 'Ġthe', 'ĠLab', 'worth', 'ĠCafÃ©', 'Ġin', 'ĠEthiopia', 'Ġfor', 'Ġsix', 'Ġpeople', '.', '</s>']