import torch
import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel, RobertaForTokenClassification, RobertaConfig
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertConfig
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class RobertaNER(RobertaForTokenClassification):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, support_per_class, cuda_device, use_bias = True, use_global = False, dataset_label_nums = None):
        super().__init__(config)
        self.support_per_class = support_per_class
        self.cuda_device = cuda_device
        self.roberta = RobertaModel(config)
        self.use_global = use_global
        # self.tokenizer = tokenizer
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.use_bias = use_bias
        # self.background = torch.nn.Parameter(torch.zeros(1)- 4., requires_grad=True)
        self.background = torch.nn.Parameter(torch.zeros(1) + 70., requires_grad=True)
        self.class_metric = torch.nn.Parameter(torch.ones(dataset_label_nums)+0.0, requires_grad=True)
        self.classifiers = []
        if self.use_global:
            for i in range(len(dataset_label_nums)):
                self.classifiers.append(torch.nn.Linear(config.hidden_size, dataset_label_nums[i]))
        self.layer1 = torch.nn.Sequential(torch.nn.Linear(config.hidden_size, 128),torch.nn.ReLU())
        self.layer2 = torch.nn.Linear(128, 1)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        self.alpha = torch.nn.Parameter(torch.Tensor(1))
        torch.nn.init.constant_(self.alpha, 0)
        self.beta = torch.nn.Parameter(torch.Tensor(1))
        torch.nn.init.constant_(self.beta, 0)

        self.init_weights()

    def compute_prototypes(self, input_ids, 
        support_class_num,
        orig_prototypes = None,
        orig_embed_class_len = None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        support_sets = outputs[0]
        batch_size,max_len,feat_dim = support_sets.shape

        if self.use_bias:
            embeds_per_class = [[] for _ in range(support_class_num * 2)]
            embed_class_len = [0 for _ in range(support_class_num * 2)]
        else:
            embeds_per_class = [[] for _ in range(support_class_num * 2 + 1)]
            embed_class_len = [0 for _ in range(support_class_num * 2 + 1)]
        labels_numpy = labels.data#.cpu().numpy()

        for i_sen, sentence in enumerate(support_sets):
            for i_word, word in enumerate(sentence):
                if attention_mask[i_sen, i_word] == 1:
                    tag = labels_numpy[i_sen][i_word]
                    if tag > 0 and self.use_bias:
                        embeds_per_class[tag - 1].append(word)
                    if tag >= 0 and self.use_bias == False:
                        embeds_per_class[tag].append(word)
                    
        # Here we compute embeddings
        prototypes = [torch.zeros_like(embeds_per_class[0][0]) for _ in range(len(embeds_per_class))]

        for i in range(len(embeds_per_class)):
            if orig_embed_class_len is not None:
                embed_class_len[i] = len(embeds_per_class[i]) + orig_embed_class_len[i]
            else:
                embed_class_len[i] = len(embeds_per_class[i])
            if orig_prototypes is not None and embed_class_len[i] > 0:
                prototypes[i] += orig_prototypes[i] * orig_embed_class_len[i] / embed_class_len[i]
            for embed in embeds_per_class[i]:
                prototypes[i] += embed / embed_class_len[i]
        
        prototypes = torch.cat([x.unsqueeze(0) for x in prototypes])#.cuda(self.cuda_device)

        return prototypes, embed_class_len

    def instance_scale(self, input): # prototype: class * 768, query: batch * seq_len * 768
        sigma = self.layer1(input)
        sigma = self.layer2(sigma)
        sigma = torch.sigmoid(sigma)
        sigma = torch.exp(self.alpha) * sigma + torch.exp(self.beta)
        return sigma

    def direct_forward(self, input_ids, 
        support_class_num,
        prototypes,
        use_global = False,
        dataset_chosen = 0,
        global_class_map = None,
        embed_class_len = None,
        soft_kmeans=False,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_logits=False,
        metric='dp',
        norm=False,
        class_metric=False,
        instance_metric=False):      

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        query_sets = outputs[0]
        batch_size,max_len,feat_dim = query_sets.shape
        # for k in range(len(prototypes)):
        #     print(f"{k} prototype {torch.sum(torch.pow(prototypes[k], 2))}")
        

        # support_sets = sequence_output[:self.support_class_num * self.support_per_class]
        # support_labels = labels[:self.support_class_num * self.support_per_class]
        # query_sets = sequence_output[self.support_class_num * self.support_per_class:]
        # query_labels = labels[self.support_class_num * self.support_per_class:]

        if use_global:
            sequence_output = self.dropout(query_sets)
            logits = self.classifiers[dataset_chosen](sequence_output)
            global_logits = torch.index_select(logits,2,global_class_map)
        
        if norm:
            query_sets = F.normalize(query_sets, dim=2)
            prototypes = F.normalize(prototypes, dim=1)
        if class_metric:
            prototypes = torch.transpose(prototypes.T * self.class_metric, 0, 1)
        elif instance_metric:
            prototypes = prototypes/self.instance_scale(prototypes)
            query_sets = query_sets/self.instance_scale(query_sets)
        if metric == 'dp':
            logits = torch.matmul(query_sets, prototypes.T)
        else:
            if self.use_bias:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
            else:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
            # del query_sets
            logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
            del query_sets_copy
        
        if self.use_bias:
            logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)
        # for k in range(len(logits[0][0])):
        #     print(f"{k} dot product {logits[0][0][k]}")

        if not soft_kmeans:

            outputs = torch.argmax(logits, dim=2)

            # sequence_output = self.dropout(sequence_output)
            # logits = self.classifier(sequence_output)
            # outputs = torch.argmax(logits, dim=2)

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                # l1 = logits.view(-1, support_class_num * 2 + 1)
                # l2 = labels.view(-1)
                # for k in range(10):
                #     print(f"label: {l2[k]} prob:\n{F.softmax(l1[k])}")
                if use_global:
                    loss += loss_fct(global_logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                if output_logits:
                    return loss, outputs, logits
                else:
                    return loss, outputs
            else:
                return outputs  
        else:
            #update prototypes
            logits = logits.view(-1, support_class_num * 2 + 1)
            # print('')
            # print('logits[0]')
            # print(logits[0])
            logits = F.softmax(logits, dim=1)
            query_sets = query_sets.view(-1, feat_dim)
            query_scores = torch.matmul(logits.T, query_sets)
            # print(logits[0])
            sum_per_cls = torch.sum(logits, dim=0)
            # print('sum')
            # print(sum_per_cls)
            # print(embed_class_len)
            prototypes1 = torch.zeros_like(prototypes)

            for cls in range(len(embed_class_len)):
                if embed_class_len[cls] == 0:
                    prototypes1[cls] = prototypes[cls]
                else:
                    prototypes1[cls] = (0.1*query_scores[cls] + embed_class_len[cls] * prototypes[cls]) / (0.1*sum_per_cls[cls] +  embed_class_len[cls])
            
            # print('distance')
            # print(torch.sum(torch.pow(prototypes1 - prototypes, 2), dim=1))
            # print(torch.matmul(prototypes, prototypes1.T))
            # print(torch.matmul(prototypes, prototypes.T))

            prototypes = prototypes1

            del query_scores
            del sum_per_cls

            query_sets = query_sets.view(-1, max_len, feat_dim)

            if self.use_bias:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
            else:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
            del query_sets
            logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
            del query_sets_copy

            if self.use_bias:
                logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)

            # print('logits[0]')
            # print(logits[0][0])
            
            outputs = torch.argmax(logits, dim=2)

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                return loss, outputs
            else:
                return outputs

    def forward_multi_prototype(self, input_ids, 
        support_class_num,
        prototype_groups,
        use_global = False,
        dataset_chosen = 0,
        global_class_map = None,
        embed_class_len = None,
        soft_kmeans=False,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_logits=False,
        metric='dp',
        norm=False,
        class_metric=False,
        instance_metric=False):      

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        query_sets = outputs[0]
        batch_size,max_len,feat_dim = query_sets.shape
        # for k in range(len(prototypes)):
        #     print(f"{k} prototype {torch.sum(torch.pow(prototypes[k], 2))}")
        

        # support_sets = sequence_output[:self.support_class_num * self.support_per_class]
        # support_labels = labels[:self.support_class_num * self.support_per_class]
        # query_sets = sequence_output[self.support_class_num * self.support_per_class:]
        # query_labels = labels[self.support_class_num * self.support_per_class:]

        if use_global:
            sequence_output = self.dropout(query_sets)
            logits = self.classifiers[dataset_chosen](sequence_output)
            global_logits = torch.index_select(logits,2,global_class_map)
        
        for i, prototypes in enumerate(prototype_groups):
            if norm:
                query_sets = F.normalize(query_sets, dim=2)
                prototypes = F.normalize(prototypes, dim=1)
            if class_metric:
                prototypes = torch.transpose(prototypes.T * self.class_metric, 0, 1)
            elif instance_metric:
                prototypes = prototypes/self.instance_scale(prototypes)
                query_sets = query_sets/self.instance_scale(query_sets)
            if metric == 'dp':
                logits = torch.matmul(query_sets, prototypes.T)
            else:
                if self.use_bias:
                    query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
                else:
                    query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
                # del query_sets
                logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
                del query_sets_copy
            
            if self.use_bias:
                logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)
            # for k in range(len(logits[0][0])):
            #     print(f"{k} dot product {logits[0][0][k]}")

            prob = F.softmax(logits, dim=2)
            if i == 0:
                new_prob = prob
            else:
                new_prob += prob
            
        prob = new_prob / len(prototype_groups)

        outputs = torch.argmax(prob, dim=2)

        # sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)
        # outputs = torch.argmax(logits, dim=2)

        if labels is not None:
            nll_loss = torch.nn.NLLLoss()
            loss = nll_loss(torch.log(prob).view(-1, support_class_num * 2 + 1), labels.view(-1))
            if use_global:
                loss += loss_fct(global_logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
            if output_logits:
                return loss, outputs, logits
            else:
                return loss, outputs
        else:
            return outputs  
        

    def direct_forward_unsup(self, input_ids, 
        support_class_num,
        prototypes,
        use_global = False,
        dataset_chosen = 0,
        global_class_map = None,
        embed_class_len = None,
        soft_kmeans=False,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        t_prob=None,
        metric='euc',
        norm=False,
        class_metric=False,
        instance_metric=False):      

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        query_sets = outputs[0]
        batch_size,max_len,feat_dim = query_sets.shape
        # for k in range(len(prototypes)):
        #     print(f"{k} prototype {torch.sum(torch.pow(prototypes[k], 2))}")

        if use_global:
            sequence_output = self.dropout(query_sets)
            logits = self.classifiers[dataset_chosen](sequence_output)
            global_logits = torch.index_select(logits,2,global_class_map)
        
        if norm:
            query_sets = F.normalize(query_sets, dim=2)
            prototypes = F.normalize(prototypes, dim=1)
        if class_metric:
            prototypes = torch.transpose(prototypes.T * self.class_metric, 0, 1)
        elif instance_metric:
            prototypes = prototypes/self.instance_scale(prototypes)
            query_sets = query_sets/self.instance_scale(query_sets)
        if metric == 'dp':
            logits = torch.matmul(query_sets, prototypes.T)
        else:
            if self.use_bias:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
            else:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
            # del query_sets
            logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
            del query_sets_copy

        outputs = torch.argmax(logits, dim=2)
        

        # sequence_output = self.dropout(sequence_output)
        # logits = self.classifier(sequence_output)
        # outputs = torch.argmax(logits, dim=2)

        if t_prob is not None:
            kl_criterion = torch.nn.KLDivLoss()
            loss = kl_criterion(torch.log(F.softmax(logits.view(-1, support_class_num * 2 + 1), dim=-1)), F.softmax(t_prob.view(-1, support_class_num * 2 + 1), dim=-1))
            # print(outputs[0][:10])
            # print(F.softmax(logits.view(-1, support_class_num * 2 + 1), dim=-1))
            # print(F.softmax(t_prob.view(-1, support_class_num * 2 + 1), dim=-1))
            # print(f"loss {loss}")
            return loss, outputs
        else:
            return outputs  
        

    def forward(self, sup_input_ids, 
        input_ids,
        sup_labels,
        labels,
        sup_attention_mask,
        attention_mask,
        support_class_num,
        use_global = False,
        dataset_chosen = 0,
        global_class_map = None,
        soft_kmeans=False,   
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        epi_iter_num1=None,
        epi_iter_num2=None,
        metric='euc',
        norm=False,
        class_metric=False,
        instance_metric=False
        ):  

        # print(f"input shape in model: {sup_input_ids.shape} {input_ids.shape}")    
        # print(f"length of epi_iter_num inside model: {len(epi_iter_num1)}")
        # print(epi_iter_num1)
        # print(epi_iter_num2)

        # compute prototypes
        # self.dropout = torch.nn.Dropout(0.1)
        outputs = self.roberta(
            sup_input_ids,
            attention_mask=sup_attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        support_sets = outputs[0]
        batch_size,max_len,feat_dim = support_sets.shape
        # support_sets = self.dropout(support_sets)

        if self.use_bias:
            embeds_per_class = [[] for _ in range(support_class_num * 2)]
            embed_class_len = [0 for _ in range(support_class_num * 2)]
        else:
            embeds_per_class = [[] for _ in range(support_class_num * 2 + 1)]
            embed_class_len = [0 for _ in range(support_class_num * 2 + 1)]
        labels_numpy = sup_labels#.data.cpu().numpy()

        for i_sen, sentence in enumerate(support_sets):
            for i_word, word in enumerate(sentence):
                if sup_attention_mask[i_sen, i_word] == 1:
                    tag = labels_numpy[i_sen][i_word]
                    if tag > 0 and self.use_bias:
                        embeds_per_class[tag - 1].append(word)
                    if tag >= 0 and self.use_bias == False:
                        embeds_per_class[tag].append(word)
                    
        # Here we compute embeddings
        prototypes = [torch.zeros_like(embeds_per_class[0][0]) for _ in range(len(embeds_per_class))]

        for i in range(len(embeds_per_class)):
            embed_class_len[i] = len(embeds_per_class[i])
            for embed in embeds_per_class[i]:
                prototypes[i] += embed / embed_class_len[i]
        
        prototypes = torch.cat([x.unsqueeze(0) for x in prototypes])#.cuda(self.cuda_device)

        del outputs
        del embeds_per_class
        del embed_class_len

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        query_sets = outputs[0]
        batch_size,max_len,feat_dim = query_sets.shape
        # query_sets = self.dropout(query_sets)

        # support_sets = sequence_output[:self.support_class_num * self.support_per_class]
        # support_labels = labels[:self.support_class_num * self.support_per_class]
        # query_sets = sequence_output[self.support_class_num * self.support_per_class:]
        # query_labels = labels[self.support_class_num * self.support_per_class:]

        if use_global:
            sequence_output = self.dropout(query_sets)
            logits = self.classifiers[dataset_chosen](sequence_output)
            global_logits = torch.index_select(logits,2,global_class_map)
        # print(f"{norm} {metric}")
        if norm:
            query_sets = F.normalize(query_sets, dim=2)
            prototypes = F.normalize(prototypes, dim=1)
        if class_metric:
            prototypes = torch.transpose(prototypes.T * self.class_metric, 0, 1)
            print(f"class metric {self.class_metric}")
        elif instance_metric:
            prototypes = prototypes/self.instance_scale(prototypes)
            query_sets = query_sets/self.instance_scale(query_sets)
        if metric == 'dp':
            logits = torch.matmul(query_sets, prototypes.T)
        else:
            if self.use_bias:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
            else:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
            # del query_sets
            logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
            del query_sets_copy
        
        
        
        
        
        if self.use_bias:
            logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)


        if not soft_kmeans:

            outputs = torch.argmax(logits, dim=2)
            # print(f"output shape in model: {outputs.shape}")

            # sequence_output = self.dropout(sequence_output)
            # logits = self.classifier(sequence_output)
            # outputs = torch.argmax(logits, dim=2)

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                if use_global:
                    loss += loss_fct(global_logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                return loss, outputs
            else:
                return outputs  
        else:
            #update prototypes
            logits = logits.view(-1, support_class_num * 2 + 1)
            # print('')
            # print('logits[0]')
            # print(logits[0])
            logits = F.softmax(logits, dim=1)
            query_sets = query_sets.view(-1, feat_dim)
            query_scores = torch.matmul(logits.T, query_sets)
            # print(logits[0])
            sum_per_cls = torch.sum(logits, dim=0)
            # print('sum')
            # print(sum_per_cls)
            # print(embed_class_len)
            prototypes1 = torch.zeros_like(prototypes)

            for cls in range(len(embed_class_len)):
                if embed_class_len[cls] == 0:
                    prototypes1[cls] = prototypes[cls]
                else:
                    prototypes1[cls] = (0.1*query_scores[cls] + embed_class_len[cls] * prototypes[cls]) / (0.1*sum_per_cls[cls] +  embed_class_len[cls])
            
            # print('distance')
            # print(torch.sum(torch.pow(prototypes1 - prototypes, 2), dim=1))
            # print(torch.matmul(prototypes, prototypes1.T))
            # print(torch.matmul(prototypes, prototypes.T))

            prototypes = prototypes1

            del query_scores
            del sum_per_cls

            query_sets = query_sets.view(-1, max_len, feat_dim)

            if self.use_bias:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
            else:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
            del query_sets
            logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
            del query_sets_copy
            if self.use_bias:
                logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)

            # print('logits[0]')
            # print(logits[0][0])
            
            outputs = torch.argmax(logits, dim=2)

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                return loss, outputs
            else:
                return outputs 

    # def output_logits(self, input_ids, 
    #     support_class_num,
    #     prototypes,
    #     embed_class_len = None,
    #     soft_kmeans=False,
    #     attention_mask=None,
    #     position_ids=None,
    #     head_mask=None,
    #     inputs_embeds=None,
    #     labels=None):      

    #     outputs = self.roberta(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #     )

    #     query_sets = outputs[0]
    #     batch_size,max_len,feat_dim = query_sets.shape

    #     # support_sets = sequence_output[:self.support_class_num * self.support_per_class]
    #     # support_labels = labels[:self.support_class_num * self.support_per_class]
    #     # query_sets = sequence_output[self.support_class_num * self.support_per_class:]
    #     # query_labels = labels[self.support_class_num * self.support_per_class:]

    #     if self.use_bias:
    #         query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
    #     else:
    #         query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
    #     # del query_sets
    #     logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
    #     del query_sets_copy
    #     if self.use_bias:
    #         logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)

    #     if not soft_kmeans:

    #         outputs = torch.argmax(logits, dim=2)

    #         # sequence_output = self.dropout(sequence_output)
    #         # logits = self.classifier(sequence_output)
    #         # outputs = torch.argmax(logits, dim=2)

    #         return prototypes, query_sets

    #     else:
    #         #update prototypes
    #         logits = logits.view(-1, support_class_num * 2 + 1)
    #         # print('')
    #         # print('logits[0]')
    #         # print(logits[0])
    #         logits = F.softmax(logits, dim=1)
    #         query_sets = query_sets.view(-1, feat_dim)
    #         query_scores = torch.matmul(logits.T, query_sets)
    #         # print(logits[0])
    #         sum_per_cls = torch.sum(logits, dim=0)
    #         # print('sum')
    #         # print(sum_per_cls)
    #         # print(embed_class_len)
    #         prototypes1 = torch.zeros_like(prototypes)

    #         for cls in range(len(embed_class_len)):
    #             if embed_class_len[cls] == 0:
    #                 prototypes1[cls] = prototypes[cls]
    #             else:
    #                 prototypes1[cls] = (0.1*query_scores[cls] + embed_class_len[cls] * prototypes[cls]) / (0.1*sum_per_cls[cls] +  embed_class_len[cls])
            
    #         # print('distance')
    #         # print(torch.sum(torch.pow(prototypes1 - prototypes, 2), dim=1))
    #         # print(torch.matmul(prototypes, prototypes1.T))
    #         # print(torch.matmul(prototypes, prototypes.T))

    #         prototypes = prototypes1


    #         return prototypes, query_sets
                
class BertNER(BertForTokenClassification):
    config_class = BertConfig

    def __init__(self, config, support_per_class, cuda_device, use_bias = True, use_global = False, dataset_label_nums = None):
        super().__init__(config)
        self.support_per_class = support_per_class
        self.cuda_device = cuda_device
        self.bert = BertModel(config)
        self.use_global = use_global
        # self.tokenizer = tokenizer
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.use_bias = use_bias
        self.background = torch.nn.Parameter(torch.zeros(1)- 4., requires_grad=True)
        self.classifiers = []
        if self.use_global:
            for i in range(len(dataset_label_nums)):
                self.classifiers.append(torch.nn.Linear(config.hidden_size, dataset_label_nums[i]))

        self.init_weights()

    def compute_prototypes(self, input_ids, 
        support_class_num,
        orig_prototypes = None,
        orig_embed_class_len = None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        support_sets = outputs[0]
        batch_size,max_len,feat_dim = support_sets.shape

        if self.use_bias:
            embeds_per_class = [[] for _ in range(support_class_num * 2)]
            embed_class_len = [0 for _ in range(support_class_num * 2)]
        else:
            embeds_per_class = [[] for _ in range(support_class_num * 2 + 1)]
            embed_class_len = [0 for _ in range(support_class_num * 2 + 1)]
        labels_numpy = labels.data.cpu().numpy()

        for i_sen, sentence in enumerate(support_sets):
            for i_word, word in enumerate(sentence):
                if attention_mask[i_sen, i_word] == 1:
                    tag = labels_numpy[i_sen][i_word]
                    if tag > 0 and self.use_bias:
                        embeds_per_class[tag - 1].append(word)
                    if tag >= 0 and self.use_bias == False:
                        embeds_per_class[tag].append(word)
                    
        # Here we compute embeddings
        prototypes = [torch.zeros_like(embeds_per_class[0][0]) for _ in range(len(embeds_per_class))]

        for i in range(len(embeds_per_class)):
            if orig_embed_class_len is not None:
                embed_class_len[i] = len(embeds_per_class[i]) + orig_embed_class_len[i]
            else:
                embed_class_len[i] = len(embeds_per_class[i])
            if orig_prototypes is not None and embed_class_len[i] > 0:
                prototypes[i] += orig_prototypes[i] * orig_embed_class_len[i] / embed_class_len[i]
            for embed in embeds_per_class[i]:
                prototypes[i] += embed / embed_class_len[i]
        
        prototypes = torch.cat([x.unsqueeze(0) for x in prototypes]).cuda(self.cuda_device)

        return prototypes, embed_class_len

    def direct_forward(self, input_ids, 
        support_class_num,
        prototypes,
        use_global = False,
        dataset_chosen = 0,
        global_class_map = None,
        embed_class_len = None,
        soft_kmeans=False,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None):      

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        query_sets = outputs[0]
        batch_size,max_len,feat_dim = query_sets.shape

        # support_sets = sequence_output[:self.support_class_num * self.support_per_class]
        # support_labels = labels[:self.support_class_num * self.support_per_class]
        # query_sets = sequence_output[self.support_class_num * self.support_per_class:]
        # query_labels = labels[self.support_class_num * self.support_per_class:]

        if use_global:
            sequence_output = self.dropout(query_sets)
            logits = self.classifiers[dataset_chosen](sequence_output)
            global_logits = torch.index_select(logits,2,global_class_map)

        if self.use_bias:
            query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
        else:
            query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
        # del query_sets
        logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
        del query_sets_copy
        if self.use_bias:
            logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)


        if not soft_kmeans:

            outputs = torch.argmax(logits, dim=2)

            # sequence_output = self.dropout(sequence_output)
            # logits = self.classifier(sequence_output)
            # outputs = torch.argmax(logits, dim=2)

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                if use_global:
                    loss += loss_fct(global_logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                return loss, outputs
            else:
                return outputs  
        else:
            #update prototypes
            logits = logits.view(-1, support_class_num * 2 + 1)
            # print('')
            # print('logits[0]')
            # print(logits[0])
            logits = F.softmax(logits, dim=1)
            query_sets = query_sets.view(-1, feat_dim)
            query_scores = torch.matmul(logits.T, query_sets)
            # print(logits[0])
            sum_per_cls = torch.sum(logits, dim=0)
            # print('sum')
            # print(sum_per_cls)
            # print(embed_class_len)
            prototypes1 = torch.zeros_like(prototypes)

            for cls in range(len(embed_class_len)):
                if embed_class_len[cls] == 0:
                    prototypes1[cls] = prototypes[cls]
                else:
                    prototypes1[cls] = (0.1*query_scores[cls] + embed_class_len[cls] * prototypes[cls]) / (0.1*sum_per_cls[cls] +  embed_class_len[cls])
            
            # print('distance')
            # print(torch.sum(torch.pow(prototypes1 - prototypes, 2), dim=1))
            # print(torch.matmul(prototypes, prototypes1.T))
            # print(torch.matmul(prototypes, prototypes.T))

            prototypes = prototypes1

            del query_scores
            del sum_per_cls

            query_sets = query_sets.view(-1, max_len, feat_dim)

            if self.use_bias:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
            else:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
            del query_sets
            logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
            del query_sets_copy
            if self.use_bias:
                logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)

            # print('logits[0]')
            # print(logits[0][0])
            
            outputs = torch.argmax(logits, dim=2)

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                return loss, outputs
            else:
                return outputs


    def forward(self, sup_input_ids, 
        input_ids,
        sup_labels,
        labels,
        sup_attention_mask,
        attention_mask,
        support_class_num,
        use_global = False,
        dataset_chosen = 0,
        global_class_map = None,
        soft_kmeans=False,   
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        epi_iter_num1=None,
        epi_iter_num2=None,
        ):  

        # print(f"input shape in model: {sup_input_ids.shape} {input_ids.shape}")    
        # print(f"length of epi_iter_num inside model: {len(epi_iter_num1)}")
        # print(epi_iter_num1)
        # print(epi_iter_num2)

        # compute prototypes
        outputs = self.bert(
            sup_input_ids,
            attention_mask=sup_attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        support_sets = outputs[0]
        batch_size,max_len,feat_dim = support_sets.shape

        if self.use_bias:
            embeds_per_class = [[] for _ in range(support_class_num * 2)]
            embed_class_len = [0 for _ in range(support_class_num * 2)]
        else:
            embeds_per_class = [[] for _ in range(support_class_num * 2 + 1)]
            embed_class_len = [0 for _ in range(support_class_num * 2 + 1)]
        labels_numpy = sup_labels#.data.cpu().numpy()

        for i_sen, sentence in enumerate(support_sets):
            for i_word, word in enumerate(sentence):
                if sup_attention_mask[i_sen, i_word] == 1:
                    tag = labels_numpy[i_sen][i_word]
                    if tag > 0 and self.use_bias:
                        embeds_per_class[tag - 1].append(word)
                    if tag >= 0 and self.use_bias == False:
                        embeds_per_class[tag].append(word)
                    
        # Here we compute embeddings
        prototypes = [torch.zeros_like(embeds_per_class[0][0]) for _ in range(len(embeds_per_class))]

        for i in range(len(embeds_per_class)):
            embed_class_len[i] = len(embeds_per_class[i])
            for embed in embeds_per_class[i]:
                prototypes[i] += embed / embed_class_len[i]
        
        prototypes = torch.cat([x.unsqueeze(0) for x in prototypes])#.cuda(self.cuda_device)

        del outputs
        del embeds_per_class
        del embed_class_len

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        query_sets = outputs[0]
        batch_size,max_len,feat_dim = query_sets.shape

        # support_sets = sequence_output[:self.support_class_num * self.support_per_class]
        # support_labels = labels[:self.support_class_num * self.support_per_class]
        # query_sets = sequence_output[self.support_class_num * self.support_per_class:]
        # query_labels = labels[self.support_class_num * self.support_per_class:]

        if use_global:
            sequence_output = self.dropout(query_sets)
            logits = self.classifiers[dataset_chosen](sequence_output)
            global_logits = torch.index_select(logits,2,global_class_map)

        if self.use_bias:
            query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
        else:
            query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
        # del query_sets
        logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
        del query_sets_copy
        if self.use_bias:
            logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)


        if not soft_kmeans:

            outputs = torch.argmax(logits, dim=2)
            # print(f"output shape in model: {outputs.shape}")

            # sequence_output = self.dropout(sequence_output)
            # logits = self.classifier(sequence_output)
            # outputs = torch.argmax(logits, dim=2)

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                if use_global:
                    loss += loss_fct(global_logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                return loss, outputs
            else:
                return outputs  
        else:
            #update prototypes
            logits = logits.view(-1, support_class_num * 2 + 1)
            # print('')
            # print('logits[0]')
            # print(logits[0])
            logits = F.softmax(logits, dim=1)
            query_sets = query_sets.view(-1, feat_dim)
            query_scores = torch.matmul(logits.T, query_sets)
            # print(logits[0])
            sum_per_cls = torch.sum(logits, dim=0)
            # print('sum')
            # print(sum_per_cls)
            # print(embed_class_len)
            prototypes1 = torch.zeros_like(prototypes)

            for cls in range(len(embed_class_len)):
                if embed_class_len[cls] == 0:
                    prototypes1[cls] = prototypes[cls]
                else:
                    prototypes1[cls] = (0.1*query_scores[cls] + embed_class_len[cls] * prototypes[cls]) / (0.1*sum_per_cls[cls] +  embed_class_len[cls])
            
            # print('distance')
            # print(torch.sum(torch.pow(prototypes1 - prototypes, 2), dim=1))
            # print(torch.matmul(prototypes, prototypes1.T))
            # print(torch.matmul(prototypes, prototypes.T))

            prototypes = prototypes1

            del query_scores
            del sum_per_cls

            query_sets = query_sets.view(-1, max_len, feat_dim)

            if self.use_bias:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
            else:
                query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
            del query_sets
            logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
            del query_sets_copy
            if self.use_bias:
                logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)

            # print('logits[0]')
            # print(logits[0][0])
            
            outputs = torch.argmax(logits, dim=2)

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, support_class_num * 2 + 1), labels.view(-1))
                return loss, outputs
            else:
                return outputs 

    def output_logits(self, input_ids, 
        support_class_num,
        prototypes,
        embed_class_len = None,
        soft_kmeans=False,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None):      

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        query_sets = outputs[0]
        batch_size,max_len,feat_dim = query_sets.shape

        # support_sets = sequence_output[:self.support_class_num * self.support_per_class]
        # support_labels = labels[:self.support_class_num * self.support_per_class]
        # query_sets = sequence_output[self.support_class_num * self.support_per_class:]
        # query_labels = labels[self.support_class_num * self.support_per_class:]

        if self.use_bias:
            query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2,1)#.view(-1,self.support_class_num * 2,feat_dim)
        else:
            query_sets_copy = query_sets.unsqueeze(-2).repeat(1,1,support_class_num * 2 + 1,1)
        # del query_sets
        logits = - torch.sum(torch.pow(query_sets_copy - prototypes, 2), dim=3)
        del query_sets_copy
        if self.use_bias:
            logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(len(labels), max_len, 1), logits), dim=2)

        if not soft_kmeans:

            outputs = torch.argmax(logits, dim=2)

            # sequence_output = self.dropout(sequence_output)
            # logits = self.classifier(sequence_output)
            # outputs = torch.argmax(logits, dim=2)

            return prototypes, query_sets

        else:
            #update prototypes
            logits = logits.view(-1, support_class_num * 2 + 1)
            # print('')
            # print('logits[0]')
            # print(logits[0])
            logits = F.softmax(logits, dim=1)
            query_sets = query_sets.view(-1, feat_dim)
            query_scores = torch.matmul(logits.T, query_sets)
            # print(logits[0])
            sum_per_cls = torch.sum(logits, dim=0)
            # print('sum')
            # print(sum_per_cls)
            # print(embed_class_len)
            prototypes1 = torch.zeros_like(prototypes)

            for cls in range(len(embed_class_len)):
                if embed_class_len[cls] == 0:
                    prototypes1[cls] = prototypes[cls]
                else:
                    prototypes1[cls] = (0.1*query_scores[cls] + embed_class_len[cls] * prototypes[cls]) / (0.1*sum_per_cls[cls] +  embed_class_len[cls])
            
            # print('distance')
            # print(torch.sum(torch.pow(prototypes1 - prototypes, 2), dim=1))
            # print(torch.matmul(prototypes, prototypes1.T))
            # print(torch.matmul(prototypes, prototypes.T))

            prototypes = prototypes1


            return prototypes, query_sets
