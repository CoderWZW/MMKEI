import torch
import torch.nn as nn
import numpy as np
from base.seq_recommender import SequentialRecommender
from transformers import BertModel,GPT2LMHeadModel
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from util.structure import PointWiseFeedForward
from util.loss_torch import l2_reg_loss
from data import feature
from data import pretrain
from data.pretrain import Pretrain
from data.sequence import Sequence
import os
from model.Module.UnisRec_module import *


# Paper: Self-Attentive Sequential Recommendation


class UnisRec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(UnisRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['UnisRec'])
        datasetFile=self.config['dataset']
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        head_num = int(args['-n_heads'])
        temperature = float(args['-temperature'])
        if os.path.exists('./model/checkpoint/UnisRec.pt'):
            state_dict = torch.load('./model/checkpoint/UnisRec.pt')
            del state_dict["item_emb"]
            self.model = UniSRec(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate,temperature,self.feature,datasetFile)
            self.model.load_state_dict(state_dict, strict=False)
            print("Loaded the model checkpoint!")
        else:
            self.model = UniSRec(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate,temperature,self.feature,datasetFile)


        self.rec_loss = torch.nn.BCEWithLogitsLoss()

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,max_len=self.max_len)):
                seq, pos, y, neg_idx, _ = batch
                seq_emb = model.forward(seq, pos)
                rec_loss = self.calculate_loss(seq_emb, y, neg_idx, pos)
                batch_loss = rec_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', batch_loss.item())
            model.eval()
            self.fast_evaluation(epoch)
        # torch.save(model.state_dict(), './model/checkpoint/SASRec.pt')

    def freeze(self,layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
                
    def calculate_loss(self, seq_emb, y, neg,pos):
        y = torch.tensor(y)
        neg = torch.tensor(neg)
        if (self.feature == 'text'):
            outputs = self.model.mlps(self.model.bert_tensor[y.cuda()])
            y_emb=outputs
            outputs = self.model.mlps(self.model.bert_tensor[neg.cuda()])
            neg_emb=outputs

        elif(self.feature == 'id'):
            y_emb = self.model.item_emb[y]
            neg_emb = self.model.item_emb[neg]
        elif(self.feature=='id+text'):
            y_emb = self.model.item_emb[y]+self.model.mlps(self.model.bert_tensor[y.cuda()])
            neg_emb = self.model.item_emb[neg]+self.model.mlps(self.model.bert_tensor[neg.cuda()])
        pos_logits = (seq_emb * y_emb).sum(dim=-1)
        neg_logits = (seq_emb * neg_emb).sum(dim=-1)
        pos_labels, neg_labels = torch.ones(pos_logits.shape).cuda(), torch.zeros(neg_logits.shape).cuda()
        indices = np.where(pos != 0)
        loss = self.rec_loss(pos_logits[indices], pos_labels[indices])
        loss += self.rec_loss(neg_logits[indices], neg_labels[indices])
        return loss

    def predict(self,seq, pos,seq_len):
        with torch.no_grad():
            seq_emb = self.model.forward(seq,pos)
            last_item_embeddings = [seq_emb[i,last-1,:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]
            item_emb = self.model.item_emb
            if self.feature == 'text':
                item_emb = self.model.mlps(self.model.bert_tensor)
            if self.feature == 'id+text':
                item_emb = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
            score = torch.matmul(torch.cat(last_item_embeddings, 0), item_emb.transpose(0, 1))
        return score.cpu().numpy()


