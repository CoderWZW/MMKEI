import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base.seq_recommender import SequentialRecommender
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from util.loss_torch import l2_reg_loss
from util.structure import PointWiseFeedForward
from math import floor
import random
from data.pretrain import Pretrain
import os
from model.Module.BERT4Rec_module import MLPS
from model.Module.BERT4Rec_module import BERT_Encoder

# Paper: BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer, CIKM'19

class BERT4Rec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(BERT4Rec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['BERT4Rec'])
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        datasetFile = self.config['dataset']
        head_num = int(args['-n_heads'])
        self.aug_rate = float(args['-mask_rate'])
        if os.path.exists('./model/checkpoint/'+self.feature+'/BERT4Rec.pt'):
            state_dict  = torch.load('./model/checkpoint/'+self.feature+'/BERT4Rec.pt')
            del  state_dict["item_emb"]
            self.model = BERT_Encoder(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate,self.feature,datasetFile)
            self.model.load_state_dict(state_dict,strict=False)
            print("Loaded the model checkpoint!")
        else:
            self.model = BERT_Encoder(self.data, self.emb_size, self.max_len, block_num, head_num, drop_rate,
                                      self.feature, datasetFile)
        self.model.freeze(self.model.last_layer_norm)
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            model.train()
            #self.fast_evaluation(epoch)
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,max_len=self.max_len)):
                seq, pos, y, neg_idx, seq_len = batch
                #mask_id=data.item_num+1
                aug_seq, masked, labels = self.item_mask_for_bert(seq, seq_len, self.aug_rate, self.data.item_num+1)
                seq_emb = model.forward(aug_seq, pos)
                # item mask
                rec_loss = self.calculate_loss(seq_emb,masked,labels)
                batch_loss = rec_loss+ l2_reg_loss(self.reg, model.item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item(), 'rec_loss:', rec_loss.item())
            model.eval()
            self.fast_evaluation(epoch)
       

    def item_mask_for_bert(self,seq,seq_len, mask_ratio, mask_idx):
        augmented_seq = seq.copy()
        masked = np.zeros_like(augmented_seq)
        labels = []
        for i, s in enumerate(seq):
            to_be_masked = random.sample(range(seq_len[i]), max(floor(seq_len[i]*mask_ratio),1))
            #batch_size*seqlen
            masked[i, to_be_masked] = 1
            # print("masked",masked)
            labels =labels+ list(augmented_seq[i, to_be_masked])
            augmented_seq[i, to_be_masked] = mask_idx
        return augmented_seq, masked, np.array(labels)

    def calculate_loss(self, seq_emb, masked, labels):
        seq_emb = seq_emb[masked>0].view(-1, self.emb_size)
        if self.feature == 'text':
            emb=self.model.mlps(self.model.bert_tensor)
        elif self.feature == 'id':
            emb= self.model.item_emb
        elif self.feature == 'id+text':
            emb = self.model.item_emb+self.model.mlps(self.model.bert_tensor)
        logits = torch.mm(seq_emb, emb.t())
        loss = F.cross_entropy(logits, torch.tensor(labels).to(torch.int64).cuda())/labels.shape[0]
        return loss

    def predict(self,seq, pos,seq_len):
        with torch.no_grad():
            for i,length in enumerate(seq_len):
                if length == self.max_len:
                    seq[i,:length-1] = seq[i,1:]
                    pos[i,:length-1] = pos[i,1:]
                    pos[i, length-1] = length
                    seq[i, length-1] = self.data.item_num+1
                else:
                    
                    pos[i, length] = length+1
                    seq[i,length] = self.data.item_num+1
            seq_emb = self.model.forward(seq,pos)
            last_item_embeddings = [seq_emb[i,last-1,:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]
         
            item_emb = self.model.item_emb
            if self.feature == 'text':
                item_emb = self.model.mlps(self.model.bert_tensor)
            if self.feature == 'id+text':
                item_emb = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
            score = torch.matmul(torch.cat(last_item_embeddings, 0), item_emb.transpose(0, 1))

        return score.cpu().numpy()
