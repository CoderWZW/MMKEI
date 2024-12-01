import torch
import torch.nn as nn
import numpy as np
from base.seq_recommender import SequentialRecommender
from transformers import BertModel,GPT2LMHeadModel
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from util.structure import PointWiseFeedForward
from util.loss_torch import l2_reg_loss,InfoNCE,batch_softmax_loss
from data import feature
from data import pretrain
from data.pretrain import Pretrain
from data.sequence import Sequence
from util.adjust_grad import assign_values_linear,adjust_learning_rate_of_rows
import os
from model.Module.UnisRec_module import *
from data.augmentor import SequenceAugmentor
from random import sample
import random
torch.cuda.set_device(0)
current_device = torch.cuda.current_device()

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
        lam = float(args['-lam'])
        self.cl_rate2=float(args['-lambda'])
        self.cl=float(args['-cl'])
        self.uni = float(args['-uni'])
        self.aug_rate = 0.2
        self.cl_rate = 0.1
        self.model = UniSRec(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate,temperature,self.feature, datasetFile)
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        with open("./item-popular/count_movie.txt", 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    data = line[1:-1].split(", ")
                    data = np.asfarray(data, float)
        file.close()
        self.data1=data
    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        listUNI = []
        for epoch in range(self.maxEpoch):
            model.train()
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,self.data1, max_len=self.max_len)):
                seq, seqfull,pos,posfull, y, neg_idx, seq_len = batch

                seq_emb = model.forward(seq, pos)
                y = torch.tensor(y)
                # print(seq_emb.size())
                if (self.feature == 'text'):
                    y_emb = model.mlps(model.bert_tensor[y.cuda()])
                elif(self.feature == 'id'):
                    y_emb = model.item_emb[y]
                elif(self.feature=='id+text'):
                    y_emb = model.item_emb[y]+model.mlps(model.bert_tensor[y.cuda()])
                # print(seq_emb.shape)
                # seq_emb = model.moe_adaptor(seq_emb)
                y_emb = model.moe_adaptor(y_emb)

                cl_emb1 = [seq_emb[i, 0:last , :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
                cl_emb2 = [y_emb[i, 0:last , :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
                cl_loss_item = InfoNCE(torch.cat(cl_emb1, 0), torch.cat(cl_emb2, 0), 1, True)
                
                if self.cl == 1:
                    cl_loss_item2 = self.cl_rate_2 * self.cal_cl_loss2(y_emb,pos)
                else:
                    cl_loss_item2=0
                aug_seq1 = SequenceAugmentor.item_mask(seq, seq_len, self.aug_rate, self.data.item_num+1)
                aug_seq2 = SequenceAugmentor.item_mask(seq, seq_len, self.aug_rate, self.data.item_num+1)
                aug_emb1 = model.forward(aug_seq1, pos)
                aug_emb2 = model.forward(aug_seq2, pos)
                aug_emb1 = model.moe_adaptor(aug_emb1)
                aug_emb2 = model.moe_adaptor(aug_emb2)
                cl_emb1 = [aug_emb1[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
                cl_emb2 = [aug_emb2[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
                cl_loss_sentence = self.cl_rate * InfoNCE(torch.cat(cl_emb1, 0), torch.cat(cl_emb2, 0), 1,True)
                UNI = self.uniformity_loss_index()
                rec_loss = self.calculate_loss(seq_emb, y, neg_idx, pos)
                # 衡量uni-loss
                if self.uni == 1:
                    # Standardized Sampling
                    uni_loss = self.uniformity_loss(y, pos,self.data1)
                elif self.uni == 2:
                    # User Sequence Sampling
                    uni_loss = self.uniformity_loss_designed(y, pos, neg_idx)
                elif self.uni == 3:
                    # Popularity Sampling
                    uni_loss = self.uniformity_loss_popularity(y, pos, neg_idx, self.data1)
                else:
                    uni_loss = 0

                batch_loss = cl_loss_item + cl_loss_sentence+cl_loss_item2+rec_loss+0.01*uni_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    # self.cosSimi()
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(),'uni_loss_item',uni_loss,"UNI",UNI.item())
                if n % 200==0:
                    listUNI.append(UNI.item())

            model.eval()
            measure, HR, NDCG = self.fast_evaluation(epoch, self.data1)
        # torch.save(model.state_dict(), './model/checkpoint/UnisRec.pt')
        # with open("./train_loss_UnisRec_Movielen.txt", 'a') as train_los:
        #     train_los.write(str(listUNI) + '\n')
    def calculate_loss(self, seq_emb, y, neg,pos):

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



    # def calculate_loss(self, seq_emb, y, neg,pos):
    #     y = torch.tensor(y)
    #     neg = torch.tensor(neg)
    #     if (self.feature == 'text'):
    #         outputs = self.model.mlps(self.model.bert_tensor[y.cuda()])
    #         y_emb=outputs
    #         outputs = self.model.mlps(self.model.bert_tensor[neg.cuda()])
    #         neg_emb=outputs
    #
    #     elif(self.feature == 'id'):
    #         y_emb = self.model.item_emb[y]
    #         neg_emb = self.model.item_emb[neg]
    #     elif(self.feature=='id+text'):
    #         y_emb = self.model.item_emb[y]+self.model.mlps(self.model.bert_tensor[y.cuda()])
    #         neg_emb = self.model.item_emb[neg]+self.model.mlps(self.model.bert_tensor[neg.cuda()])
    #     pos_logits = (seq_emb * y_emb).sum(dim=-1)
    #     neg_logits = (seq_emb * neg_emb).sum(dim=-1)
    #     pos_labels, neg_labels = torch.ones(pos_logits.shape).cuda(), torch.zeros(neg_logits.shape).cuda()
    #     indices = np.where(pos != 0)
    #     loss = self.rec_loss(pos_logits[indices], pos_labels[indices])
    #     loss += self.rec_loss(neg_logits[indices], neg_labels[indices])
    #     return loss
    def uniformity_loss(self, label,pos,data,t=2):
        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        if self.feature == 'id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif (self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb


        label = torch.tensor(label)
        label = label[np.where(pos != 0)]
        label = label[np.where(data[label] >= 30)]
        # print(len(label))
        x=item_view[label]
        x=x.reshape([-1,64])
        # realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1 * x.shape[0]))
        # neg_emb = item_view[realneg]
        # neg_emb = neg_emb.reshape([-1, 64])
        #
        # # neg_emb = neg_emb[:int(2.5*x.shape[0])]
        # # x=neg_emb
        # x = torch.cat([x, neg_emb], dim=0)
        x = F.normalize(x, dim=-1)

        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
    def uniformity_loss_index(self, t=2):
        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        if self.feature == 'id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif (self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
        sample= random.sample(list(range(1, self.data.item_num + 1)),2000)
        emb = item_view[sample]
        emb = emb.reshape([-1, 64])
        emb = F.normalize(emb, dim=-1)
        return torch.pdist(emb, p=2).pow(2).mul(-t).exp().mean().log()

    def uniformity_loss_designed(self, label, pos, neg, t=2):

        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        if self.feature == 'id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif (self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
        label = torch.tensor(label)
        labelforcount = label.clone().detach()
        non_zero_counts = np.count_nonzero(labelforcount, axis=1)
        cumulative_counts = np.cumsum(non_zero_counts)
        cumulative_counts.tolist()
        # final_counts = cumulative_counts - 1
        # final_counts.tolist()

        label = label[np.where(pos != 0)]

        x = item_view[label]
        x = x.reshape([-1, 64])

        realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5 * x.shape[0]))
        neg_emb = item_view[realneg]
        neg_emb = neg_emb.reshape([-1, 64])

        # neg_emb = neg_emb[:int(2.5*x.shape[0])]
        x = torch.cat([x, neg_emb], dim=0)
        x = F.normalize(x, dim=-1)
        # list1=self.optimized_find_index_in_final_counts(final_counts,x.shape[0])

        dists = torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
        n = x.size(0)


        i = 0

        for num in range(0, len(cumulative_counts)):
            j = cumulative_counts[num]
            # print("dist",dists)

            dist1 = torch.pdist(x[i:j], p=2).pow(2).mul(-t).exp().mean().log()
            # print(dist1)
            if (torch.isnan(0.9 * dist1 / len(cumulative_counts)) == 0):
                dists = dists - 0.9 * dist1 / len(cumulative_counts)

            i = j
        result = dists
        return result


    def uniformity_loss_popularity(self, label, pos, neg, data, t=2):
        item_view = self.model.item_emb
        if self.feature == 'text':
            item_view = self.model.mlps(self.model.bert_tensor)
        if self.feature == 'id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif (self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
        label = torch.tensor(label)

        data = torch.tensor(data)

        label = label[np.where(pos != 0)]
        label = label[np.where(data[label] < 30)]
        Xdata = data[label].reshape(-1)
        # print("cc",Xdata.shape)
        x = item_view[label]
        x = x.reshape([-1, 64])
        # print(x.shape)
        realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1 * x.shape[0]))
        neg_emb = item_view[realneg]
        neg_emb = neg_emb.reshape([-1, 64])
        Ydata = torch.ones(int(1* x.shape[0]))
        '''
        Ydata = data[realneg]
        Ydata[Ydata == 0] = 1
        '''
        data = torch.cat([Xdata, Ydata], dim=0).cuda()

        # print(data.mean())
        # neg_emb = neg_emb[:int(2.5*x.shape[0])]
        x = torch.cat([x, neg_emb], dim=0)
        x = F.normalize(x, dim=-1)
        # data=data.view(-1, 1)

        # data= multiply_tensor_elements(data)
        distance = torch.triu(torch.ger(data, data), diagonal=1)
        distance = distance[distance != 0]

        # around400
        distance = distance / 5
        # print(distance.mean())
        # return torch.div(torch.pdist(x, p=2).pow(2), distance).mul(-t).exp().mean().log()
        return torch.div(torch.pdist(x, p=2).pow(2).mul(-t).exp(), distance).mean().log()

    def cosSimi(self):
        ItemInd = [i for i in range(1,self.data.item_num+1)]
        ItemInd = random.sample(ItemInd, 100)
        tensors=self.model.item_emb[ItemInd]

        cos_similarities = []
        # print(len(tensors))
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                cos_sim = F.cosine_similarity(tensors[i].unsqueeze(0), tensors[j].unsqueeze(0))
                cos_similarities.append(cos_sim)
        # 计算平均余弦相似度
        average_cos_similarity = torch.mean(torch.stack(cos_similarities))
        # print("Average Cosine Similarity of ID:", average_cos_similarity.item())

        if self.feature == 'id+text'or self.feature == 'text':
            ItemInd = [i for i in range(1,self.data.item_num+1)]
            ItemInd = random.sample(ItemInd, 100)
            tensors = self.model.mlps(self.model.bert_tensor)[ItemInd]
            cos_similarities = []
            # print(len(tensors))
            for i in range(len(tensors)):
                for j in range(i + 1, len(tensors)):
                    cos_sim = F.cosine_similarity(tensors[i].unsqueeze(0), tensors[j].unsqueeze(0))
                    cos_similarities.append(cos_sim)
            average_cos_similarity = torch.mean(torch.stack(cos_similarities))
            print("Average Cosine Similarity of Text:", average_cos_similarity.item())