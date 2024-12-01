#=https://github.com/ChengxiLi5/STRec/blob/main/models.py train
import torch
import torch.nn as nn
import numpy as np
from base.seq_recommender import SequentialRecommender
from transformers import BertModel,GPT2LMHeadModel
from util.conf import OptionConf
from util.sampler import next_batch_sequence
import array
from util.structure import PointWiseFeedForward
from util.loss_torch import l2_reg_loss
import random
from data import feature
import torch.nn.functional as F
from data import pretrain
from datetime import datetime
from data.pretrain import Pretrain
from data.sequence import Sequence
import math
import os
import pandas as pd
from model.Module.STRec_module import MLPS
from model.Module.STRec_module import STRec_Model
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from util.adjust_grad import assign_values_linear,adjust_learning_rate_of_rows
from random import sample
import os
import sys
sys.path.append("/usr/gao/cwh/plot_pca_embedding")

# 导入模块
from plot_pca_embedding import plot_pca_embedding

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Paper: Self-Attentive Sequential Recommendation
torch.cuda.set_device(1)
current_device = torch.cuda.current_device()

class STRec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(STRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['STRec'])
        datasetFile=self.config['dataset']
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        self.cl_rate = float(args['-lambda'])
        self.cl_type=args['-cltype']
        self.cl=float(args['-cl'])
        head_num = int(args['-n_heads'])
        self.uni = float(args['-uni'])
        self.model = STRec_Model(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate,self.feature,datasetFile)
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self.eps = float(args['-eps'])
        self.listcountitem = [0] * (self.data.item_num + 1)

    def train(self):
        model = self.model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), self.lRate)
        model_performance = []
        listUNI = []

        listdistance_ave = []
        listdistance_pop = []
        listHR = []
        listNDCG = []

        # print(sum(listcountitem))
        for epoch in range(self.maxEpoch):
            self.listcountitem = [0] * (self.data.item_num + 1)
            model.train()

            # self.fast_evaluation(epoch)
            for n, batch in enumerate(
                    next_batch_sequence(self.data, self.batch_size, self.labelgap, max_len=self.max_len)):
                seq, seqfull, pos, posfull, y, neg_idx, _,popgap_batch = batch
                # self.listcountitem = np.sum([self.count_tensor_elements(seq, self.data.item_num), self.listcountitem],
                #                             axis=0).tolist()


                seq_emb = model.forward(seq, pos,popgap_batch)

                rec_loss= self.calculate_loss(seq_emb, y, neg_idx, pos)

                optimizer.zero_grad()
                # optimizer1.zero_grad()
                # print(listcountitem)
                # self.model.bert_tensor.retain_grad()
                rec_loss.backward()


                optimizer.step()

                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())


                # 用于检查偏移量
                # if epoch == 0:
                #     torch.save(self.model.item_emb, f'epoch_{epoch}_tensor.pt')

            model.eval()
            model_performance.append(model.state_dict)
            measure, HR, NDCG = self.fast_evaluation(epoch, self.data1)
        plot_pca_embedding(self.model.word_embeddings5.detach().cpu().numpy(), 'llama',
                           self.model.item_emb.detach().cpu().numpy(), 'id',
                           self.model.MoE.word_embeddinglist[0].detach().cpu().numpy(), 'word')

    def calculate_loss(self, seq_emb, y, neg, pos):
        y = torch.tensor(y)
        neg = torch.tensor(neg)

        # if self.feature == 'text':
        #     outputs = self.model.mlps(self.model.bert_tensor[y.cuda()])
        #     y_emb = outputs
        #     outputs = self.model.mlps(self.model.bert_tensor[neg.cuda()])
        #     neg_emb = outputs
        # elif self.feature == 'id':
        #     y_emb = self.model.item_emb[y]
        #     neg_emb = self.model.item_emb[neg]
        # elif self.feature == 'id+text':
        #     y_emb = self.model.item_emb[y] + self.model.mlps(self.model.bert_tensor[y.cuda()])
        #     neg_emb = self.model.item_emb[neg] + self.model.mlps(self.model.bert_tensor[neg.cuda()])
        # 找到每个序列中最后一个 pos 不等于 0 的位置
        # 使用 (pos != 0) 生成布尔掩码，然后通过 max 找到最后一个 True 的位置
        pos = torch.tensor(pos)
        indices = (pos != 0).long()  # 将布尔掩码转换为 long，方便后续处理
        indices = indices.cumsum(dim=1).eq(indices.sum(dim=1, keepdim=True))  # 获取最后一个非零的位置

        y_emb = self.model.item_emb[y]
        y_emb=y_emb[indices]
        neg_emb = self.model.item_emb[neg]
        neg_emb = neg_emb[indices]

        # 计算 logits
        pos_logits = (seq_emb * y_emb).sum(dim=-1)
        neg_logits = (seq_emb * neg_emb).sum(dim=-1)

        # 创建正负标签
        pos_labels, neg_labels = torch.ones(pos_logits.shape).cuda(), torch.zeros(neg_logits.shape).cuda()


        # 索引有效的 logits 进行损失计算
        loss = self.rec_loss(pos_logits, pos_labels)
        loss += self.rec_loss(neg_logits, neg_labels)

        return loss
    def predict(self,seq, pos,seq_len,time_stamps_seq):
        with torch.no_grad():
            seq_emb = self.model.forward(seq,pos,time_stamps_seq)
            # print(seq_emb.shape)
            # last_item_embeddings = [seq_emb[i,last:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]
            # print(last_item_embeddings.shape)
            item_emb=self.model.item_emb
            # if self.feature == 'text':
            #       item_emb=self.model.mlps(self.model.bert_tensor)
            #       # item_emb=self.model.bert_tensor
            # if self.feature=='id+text':
            #       item_emb=self.model.mlps(self.model.bert_tensor)+self.model.item_emb
            score = torch.matmul(seq_emb.squeeze(1) ,  item_emb.transpose(0, 1))

        return score.cpu().numpy()


