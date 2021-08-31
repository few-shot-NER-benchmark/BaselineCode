# naive token-level classification model (including bert-based and roberta-based)

import torch
import transformers
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel, RobertaForTokenClassification, RobertaConfig
from transformers import BertTokenizer, BertModel, BertForTokenClassification, BertConfig
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class RobertaNER(RobertaForTokenClassification):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, dataset_label_nums, multi_gpus=False):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dataset_label_nums = dataset_label_nums
        self.multi_gpus = multi_gpus
        # self.tokenizer = tokenizer
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels-1)
        # used in training dataset
        # self.classifier1 = torch.nn.Linear(config.hidden_size, dataset_label_nums[0])
        # used in testing dataset (if from different domains)
        # self.classifier2 = torch.nn.Linear(config.hidden_size, dataset_label_nums[1])
        self.classifiers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, x) for x in dataset_label_nums])
        self.background = torch.nn.Parameter(torch.zeros(1)- 2., requires_grad=True)

        self.init_weights()

    def forward(self, input_ids, 
        attention_mask=None,
        dataset = 0,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_logits=False):
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        logits = self.classifiers[dataset](sequence_output)
        # logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1), logits), dim=2)
        outputs = torch.argmax(logits, dim=2)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.dataset_label_nums[dataset]), labels.view(-1))
            if output_logits:
                return loss, outputs, logits
            else:
                return loss, outputs
        else:
            return outputs  

    def forward_unsup(self, input_ids, 
        attention_mask=None,
        dataset = 0,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        t_prob=None,
        output_logits=False):
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        logits = self.classifiers[dataset](sequence_output)
        # logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1), logits), dim=2)
        outputs = torch.argmax(logits, dim=2)
        sel_idx = torch.tensor([j+i*len(x) for i,x in enumerate(attention_mask) for j in range(len(x)) if x[j] == 1]).cuda()
        # print(f"shape: {sel_idx.shape}")
        # print(sel_idx)
        log_pred_prob = torch.log(F.softmax(logits.view(-1, self.dataset_label_nums[dataset]), dim=-1))
        log_pred_prob = torch.index_select(log_pred_prob, 0, sel_idx)
        t_prob = F.softmax(t_prob.view(-1, self.dataset_label_nums[dataset]), dim=-1)
        t_prob = torch.index_select(t_prob, 0, sel_idx)

        
        kl_criterion = torch.nn.KLDivLoss()
        loss = kl_criterion(log_pred_prob, t_prob)
        if output_logits:
            return loss, outputs, logits
        else:
            return loss, outputs
        


class BertNER(BertForTokenClassification):
    config_class = BertConfig

    def __init__(self, config, dataset_label_nums,multi_gpus=False):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dataset_label_nums = dataset_label_nums
        self.multi_gpus = multi_gpus
        # self.tokenizer = tokenizer
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels-1)
        # self.classifier1 = torch.nn.Linear(config.hidden_size, dataset_label_nums[0])
        # self.classifier2 = torch.nn.Linear(config.hidden_size, dataset_label_nums[1])
        # self.classifiers = [self.classifier1, self.classifier2]
        self.classifiers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, x) for x in dataset_label_nums])
        self.background = torch.nn.Parameter(torch.zeros(1)- 2., requires_grad=True)

        self.init_weights()

    def forward(self, input_ids, 
        attention_mask=None,
        dataset = 0,
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

        sequence_output = outputs[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)
        logits = self.classifiers[dataset](sequence_output)
        # logits = torch.cat((self.background.unsqueeze(0).unsqueeze(0).repeat(batch_size, max_len, 1), logits), dim=2)
        outputs = torch.argmax(logits, dim=2)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.dataset_label_nums[dataset]), labels.view(-1))
            return loss, outputs
        else:
            return outputs  
