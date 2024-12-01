

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.structure import PointWiseFeedForward
import os
from data.pretrain import Pretrain
#
# torch.cuda.set_device(1)
# current_device = torch.cuda.current_device()

class ColdHotRec_Model(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate, feature, datasetFile):
        super(ColdHotRec_Model, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self.feature = feature
        self.datasetFile = datasetFile

        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_

        if (self.feature == 'text' or self.feature == 'id+text'):
            self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda(0)
            if (len(self.datasetFile.split(",")) == 1):
                if not os.path.exists(self.datasetFile + "whole_tensor.pt"):
                    mask = 0
                    pre = Pretrain(self.data, self.datasetFile, mask)
                tensor = torch.load(self.datasetFile + "whole_tensor.pt")
                tensor = tensor.to(0)
                self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)


                self.mlps = MLPS(self.emb_size)
            elif (len(self.datasetFile.split(",")) > 1):
                self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                for dataset in self.datasetFile.split(","):
                    if not os.path.exists(dataset + "whole_tensor.pt"):
                        mask = 0
                        pre = Pretrain(self.data, dataset, mask)
                    tensor = torch.load(dataset + "whole_tensor.pt")
                    self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)



                # self.bert_tensor = torch.load('./dataset/fuse_tensor'+ filename+'.pt')
                self.mlps = MLPS(self.emb_size)
        
        self.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num + 1, self.emb_size)))
        self.sasmodel1=sasStructure(self.emb_size, self.max_len, self.drop_rate, self.head_num,self.block_num)
        self.sasmodel2 = sasStructure(self.emb_size, self.max_len, self.drop_rate, self.head_num,self.block_num)

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
    def forward(self, aug_seq,seq, pos,masked):
        seq = torch.tensor(seq)
        aug_seq = torch.tensor(aug_seq)
        pos = torch.tensor(pos)

        if (self.feature == 'text'):
            seq_emb = self.mlps(self.bert_tensor[seq.cuda()])
            aug_seq_emb = self.mlps(self.bert_tensor[aug_seq.cuda()])
        elif (self.feature == 'id'):
            seq_emb = self.item_emb[seq]
        elif (self.feature == 'id+text'):
            seq_emb = self.item_emb[seq] + self.mlps(self.bert_tensor[seq.cuda()])


        masked=torch.tensor(masked)
        aug_seq_emb=self.sasmodel1(aug_seq,aug_seq_emb,pos)
        aug_seq_emb[masked == 1] = seq_emb[masked == 1]
        aug_seq_emb=self.sasmodel2(seq,aug_seq_emb,pos)

        return aug_seq_emb



# encoder
class MLPS(nn.Module):
    def __init__(self, H):
        super(MLPS, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of label
        # Instantiate BERT model
        # self.bert = BertModel.from_pretrained('/usr/gao/cwh/bert')
        # self.bert = BertModel.from_pretrained('bert')
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        # self.bert_tensor = bert_tensor
        # self.bert_tensor.requires_grad = True
        # print("self.bert_tensor", self.bert_tensor[0])
        self.H = H
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.H),
            nn.ReLU(),
        )

    def forward(self, bert_tensor):
        # [batchsize,sequenceLen,large_item_embedding]->[batchsize,sequenceLen,small_item_embedding]
        logits = self.classifier(bert_tensor)
        # print("logits", logits.shape)
        # logits=torch.reshape(logits,(batch,m,self.H))
        return logits


class sasStructure(nn.Module):
    def __init__(self,emb_size,max_len,drop_rate,head_num,block_num):
        super(sasStructure, self).__init__()
        initializer = nn.init.xavier_uniform_
        self.emb_size=emb_size
        self.pos_emb = nn.Parameter(initializer(torch.empty(max_len + 1, emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(emb_size, eps=1e-8)


        for n in range(block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(emb_size, eps=1e-8))
            new_attn_layer = torch.nn.MultiheadAttention(emb_size, head_num, drop_rate)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(emb_size, eps=1e-8))

            new_fwd_layer = PointWiseFeedForward(emb_size, drop_rate)
            self.forward_layers.append(new_fwd_layer)

    def forward(self,seq,seq_emb,pos):
        seq_emb = seq_emb * self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        timeline_mask = torch.BoolTensor(seq == 0).cuda()
        # print("timeline_mask",timeline_mask)
        seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        tl = seq_emb.shape[1]

        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).cuda())
        # print("attention_mask",attention_mask)
        for i in range(len(self.attention_layers)):
            seq_emb = torch.transpose(seq_emb, 0, 1)
            # attention_input = seq_emb
            normalized_emb = self.attention_layer_norms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=attention_mask)
            seq_emb = normalized_emb + mha_outputs
            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        seq_emb = self.last_layer_norm(seq_emb)
        return seq_emb