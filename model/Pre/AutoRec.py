import torch
import torch.nn as nn
import numpy as np
from base.seq_recommender import SequentialRecommender
from transformers import BertModel,GPT2LMHeadModel
from util.conf import OptionConf
from util.sampler import next_batch_sequence
import array
from math import floor
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
from model.Module.AutoMLPBlock_module import MLPS
from model.Module.AutoMLPBlock_module import AuToMLPS,AutoRec_Model,BERT_Encoder
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from util.adjust_grad import assign_values_linear,adjust_learning_rate_of_rows
from random import sample
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Paper: Self-Attentive Sequential Recommendation
torch.cuda.set_device(1)
current_device = torch.cuda.current_device()

class AutoRec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(AutoRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SASRec'])
        datasetFile = self.config['dataset']
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        self.cl_rate = float(args['-lambda'])
        self.cl_type = args['-cltype']
        self.cl = float(args['-cl'])
        head_num = int(args['-n_heads'])

        self.aug_rate = float(args['-mask_rate'])

        self.uni = float(args['-uni'])
        self.model = AutoRec_Model(self.data, self.emb_size, self.max_len, block_num, head_num, drop_rate, self.feature,
        #                           datasetFile)
        # self.model = BERT_Encoder(self.data, self.emb_size, self.max_len, block_num, head_num, drop_rate, self.feature,
                                                              datasetFile)
        self.model2=AuToMLPS(self.emb_size)
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self.eps = float(args['-eps'])
        self.listcountitem = [0] * (self.data.item_num + 1)
        # self.listcountitem = [0] * (self.data.item_num + 2)
    def train(self):

        model = self.model.cuda()
        model2=self.model2.cuda()
        # 关于梯度更新调整的模块实验

        if self.feature == 'id':
            param = self.model.item_emb
        else:
            # param = self.model.bert_tensor  # 例如模型某层的权重
            param =self.model.mlps(self.model.bert_tensor)
        # 使用.register_hook来应用钩子
        # 这里假设你想要为param的第0行和第1行设置不同的学习率

        # rows_to_adjust = [i for i in range(self.data.item_num + 1)][1:]
        #
        #
        # hook = param.register_hook(
        #     lambda grad: adjust_learning_rate_of_rows(grad, rows_to_adjust, 0.005, self.lRate, self.listcountitem))

        optimizer = torch.optim.Adam(model.parameters(), self.lRate)
        model_performance = []
        listUNI = []

        listdistance_ave = []
        listdistance_pop = []
        listHR = []
        listNDCG = []

        # # 重新在util里写一个提取labelgap的
        # labelgap = []
        # with open('./dataset/Amazon-Office/PopGap.txt', 'r') as file:
        #     for line in file:
        #         # 分割每一行的数字，并将它们转换为整数
        #         numbers = list(map(int, line.split()))
        #         if len(numbers) < 49:
        #             numbers.extend([0] * (49 - len(numbers)))
        #             # 如果当前行的长度大于 50，截取前 50 个数字
        #         elif len(numbers) > 49:
        #             numbers = numbers[:49]
        #
        #             # 添加处理后的数据到列表中
        #         labelgap.append(numbers)
        #
        #
        # # 将数据列表转换为 PyTorch 张量
        # labelgap = torch.tensor(labelgap)

        # print(sum(listcountitem))
        for epoch in range(self.maxEpoch):
            self.listcountitem = [0] * (self.data.item_num + 1)
            # self.listcountitem = [0] * (self.data.item_num + 2)
            model.train()

            # self.fast_evaluation(epoch)
            for n, batch in enumerate(
                    next_batch_sequence(self.data, self.batch_size, self.labelgap, max_len=self.max_len)):

                seq, seqfull,pos,posfull, y, neg_idx, _ = batch
              

                # aug_seq, masked, labels = self.item_mask_for_bert(seq, _, self.aug_rate, self.data.item_num + 1)

                self.listcountitem = np.sum([self.count_tensor_elements(seq, self.data.item_num), self.listcountitem],
                                            axis=0)
                seq_emb = model.forward(seq, pos)
                ###
                ###计算group_loss模块
                #看来得前向传播两次，因为这次前向传播seq没把最后一个包进去

                #这次前传不更新参数
                seqfull=model.forward(seqfull,posfull)
                gaptensor = seqfull.narrow(1, 0, seqfull.size(1) - 1) - seqfull.narrow(1, 1, seqfull.size(1) - 1)

                # end_row = min((n + 1) * self.batch_size,   len(labelgap))
                # #还得再变换一次
                # selected_rows = labelgap[n * self.batch_size:  end_row]
                #
                # selected_rows=torch.tensor(selected_rows)
                #
                # max_index =  torch.argmax(selected_rows, dim=1, keepdim=True)
                # group_logit= model2.forward(gaptensor)
                # group_loss=self.calculate_group_loss(max_index, group_logit,selected_rows)
                ##
                ##

                if self.cl == 1:
                    cl_loss = self.cl_rate * self.cal_cl_loss(y, pos)
                else:
                    cl_loss = 0

                if self.uni == 1:
                    # Standardized Sampling
                    uni_loss = self.uniformity_loss(y, pos, self.data1)
                elif self.uni == 2:
                    # User Sequence Sampling
                    uni_loss = self.uniformity_loss_designed(y, pos, neg_idx)
                elif self.uni == 3:
                    # Popularity Sampling
                    uni_loss = self.uniformity_loss_popularity(y, pos, neg_idx, self.data1)
                else:
                    uni_loss = 0
                UNI = self.uniformity_loss_index()
                rec_loss = self.calculate_loss(seq_emb, y, neg_idx, pos)
                # rec_loss = self.calculate_loss(seq_emb, masked, labels)
                # print(10*group_loss)
                #
                # print(rec_loss)
                batch_loss = rec_loss + cl_loss+0.03*uni_loss
                # 可选择加正则化
                # batch_loss = rec_loss+ l2_reg_loss(self.reg, model.item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                # optimizer1.zero_grad()
                # print(listcountitem)
                # self.model.bert_tensor.retain_grad()
                batch_loss.backward()

                # 判断勾子是否调用
                # if param._backward_hooks:
                #     print("Hook(s) registered.")
                # else:
                #     print("No hooks registered.")

                # print(self.model.bert_tensor.grad)
                optimizer.step()
                # optimizer1.zero_grad()

                # print(self.model.bert_tensor[3][0])
                if n % 50 == 0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'uni_loss:', uni_loss,
                          "UNI", UNI.item())
                if n % 200 == 0:
                    listUNI.append(UNI.item())

                # 用于检查偏移量
                # if epoch == 0:
                #     torch.save(self.model.item_emb, f'epoch_{epoch}_tensor.pt')
            model.eval()
            model_performance.append(model.state_dict)
            measure, HR, NDCG = self.fast_evaluation(epoch, self.data1)
            # ave, pop = self.count_distance()
            # listdistance_ave.append(ave.item())
            # listdistance_pop.append(pop.item())
            listHR.append(HR)
            listNDCG.append(NDCG)

        # 下面挺重要，建议别删

        # with open("./count_yelp.txt", 'w') as train_los:
        #     train_los.write(str(self.listcountitem))

        # torch.save(model_performance[self.bestPerformance[0]-1], './model/checkpoint/'+self.feature+'/SASRec.pt')

        # self.count_distance()

        # self.drawtsne()

        # 之前存uni值用的
        # with open("./train_loss_SASRec_1.txt", 'a') as train_los:
        #     train_los.write(str(listUNI) + '\n')

        # 存一下偏移量
        # with open("./train_HR_and_distance_SASRec_beauty_grad_adjust.txt", 'a') as train_HR_and_distance:
        #     train_HR_and_distance.write(str(listdistance_ave) + '\n')
        #     train_HR_and_distance.write(str(listdistance_pop) + '\n')
        #     train_HR_and_distance.write(str(listHR) + '\n')
        #     train_HR_and_distance.write(str(listNDCG) + '\n')
    def calculate_group_loss(self, max_index, group_logit,selected_rows):




        ######该段代码为最大的gap计算出的loss函数
        # # max_index.squeeze(1).size()=batchsize
        # #print(group_logit.size())=【batchsize,49]
        # group_loss = F.cross_entropy(group_logit,  max_index.squeeze(1).cuda().to(0))
        ######


        #####该代码为多gap计算出的loss函数
        k_list = (selected_rows != 0).sum(dim=1).tolist()
        list1_ranks = []
        list2_ranks= []
        weights=[]

        for i in range(selected_rows.size(0)):
            # 提取前 k 个元素
            k = k_list[i]
            list1_topk = selected_rows[i, :k].tolist()
            list2_topk = group_logit[i, :k].tolist()



            # 计算排名
            list1_rank = [sorted(list1_topk,reverse=True).index(x) for x in list1_topk]
            list2_rank = [sorted(list2_topk,reverse=True  ).index(x) for x in list2_topk]
            if not list1_rank:
                continue

            max_rank = max(list1_rank) + 1
            weight = [(max_rank - rank) for rank in list1_rank]

            min_weight = min(weight)
            max_weight = max(weight)
            weight = [(w - min_weight) / (max_weight - min_weight+1) for w in weight]

            list1_ranks.append(list1_rank)
            list2_ranks.append(list2_rank)
            weights.append(weight)
        # 将排名转换为张量并填充为相同长度
        max_len = max(len(rank) for rank in list1_ranks)

        list1_ranks_padded = torch.tensor([rank + [0] * (max_len - len(rank)) for rank in list1_ranks],
                                          dtype=torch.float32)
        list2_ranks_padded = torch.tensor([rank + [0] * (max_len - len(rank)) for rank in list2_ranks],
                                          dtype=torch.float32)
        weights_padded = torch.tensor([weight + [0] * (max_len - len(weight)) for weight in weights],
                                      dtype=torch.float32)
        # 定义损失函数
        loss_fn = nn.MSELoss()
        # 计算损失
        group_loss = loss_fn(list1_ranks_padded, list2_ranks_padded)
        weighted_loss =  group_loss  * weights_padded
        group_loss = weighted_loss.mean()

        #####



        return group_loss

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
        sample = random.sample(list(range(1, self.data.item_num + 1)), 2000)
        emb = item_view[sample]
        emb = emb.reshape([-1, 64])
        emb = F.normalize(emb, dim=-1)
        return torch.pdist(emb, p=2).pow(2).mul(-t).exp().mean().log()
    def calculate_loss(self, seq_emb, y, neg,pos):

        y = torch.tensor(y)
        neg = torch.tensor(neg)
        if (self.feature == 'text'):
            # new_inputs = self.model.train_inputs[y]
            # new_masks = self.model.train_masks[y]

            outputs = self.model.mlps(self.model.bert_tensor[y.cuda()])

            #做文本空间直接偏移时用到的代码
            # outputs =  self.model.bert_tensor[y.cuda()]
            y_emb=outputs

            # new_inputs = self.model.train_inputs[neg]
            # new_masks = self.model.train_masks[neg]


            outputs = self.model.mlps(self.model.bert_tensor[neg.cuda()])
            # 做文本空间直接偏移时用到的代码
            # outputs = self.model.bert_tensor[neg.cuda()]
            neg_emb=outputs

        elif(self.feature == 'id'):
            y_emb = self.model.item_emb[y]
            neg_emb = self.model.item_emb[neg]
        elif(self.feature=='id+text'):
            y_emb = self.model.item_emb[y]+self.model.mlps(self.model.bert_tensor[y.cuda()])
            neg_emb = self.model.item_emb[neg]+self.model.mlps(self.model.bert_tensor[neg.cuda()])
        # print("seq_emb", seq_emb.shape)
        # print("y_emb", y_emb.shape)
        pos_logits = (seq_emb * y_emb).sum(dim=-1)
        neg_logits = (seq_emb * neg_emb).sum(dim=-1)
        pos_labels, neg_labels = torch.ones(pos_logits.shape).cuda(), torch.zeros(neg_logits.shape).cuda()
        indices = np.where(pos != 0)
        loss = self.rec_loss(pos_logits[indices], pos_labels[indices])
        loss += self.rec_loss(neg_logits[indices], neg_labels[indices])
        return loss

    def item_mask_for_bert(self, seq, seq_len, mask_ratio, mask_idx):
        augmented_seq = seq.copy()
        masked = np.zeros_like(augmented_seq)
        labels = []
        for i, s in enumerate(seq):
            to_be_masked = random.sample(range(seq_len[i]), max(floor(seq_len[i] * mask_ratio), 1))
            masked[i, to_be_masked] = 1
            # print("masked",masked)
            labels = labels + list(augmented_seq[i, to_be_masked])
            augmented_seq[i, to_be_masked] = mask_idx
        return augmented_seq, masked, np.array(labels)
    #bert的算损失
    # def calculate_loss(self, seq_emb, masked, labels):
    #
    #     masked = torch.tensor(masked)
    #     seq_emb = seq_emb[masked > 0]
    #     seq_emb = seq_emb.view(-1, self.emb_size)
    #     if self.feature == 'text':
    #         emb = self.model.mlps(self.model.bert_tensor)
    #         # emb = self.model.bert_tensor
    #     elif self.feature == 'id':
    #         emb = self.model.item_emb
    #     elif self.feature == 'id+text':
    #         emb = self.model.item_emb + self.model.mlps(self.model.bert_tensor)
    #     logits = torch.mm(seq_emb, emb.t())
    #     # F.cross_entropy(logits, torch.tensor(labels).to(torch.int64).cuda())/labels.shape[0]
    #     loss = F.cross_entropy(logits, torch.tensor(labels).to(torch.int64).cuda())
    #     return loss
    #bert的predict
    # def predict(self, seq, pos, seq_len):
    #     with torch.no_grad():
    #         for i, length in enumerate(seq_len):
    #             if length == self.max_len:
    #                 seq[i, :length - 1] = seq[i, 1:]
    #                 pos[i, :length - 1] = pos[i, 1:]
    #                 pos[i, length - 1] = length
    #                 seq[i, length - 1] = self.data.item_num + 1
    #             else:
    #                 pos[i, length] = length + 1
    #                 seq[i, length] = self.data.item_num + 1
    #         seq_emb = self.model.forward(seq, pos)
    #         last_item_embeddings = [seq_emb[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
    #
    #         item_emb = self.model.item_emb
    #         if self.feature == 'text':
    #             item_emb = self.model.mlps(self.model.bert_tensor)
    #             # item_emb = self.model.bert_tensor
    #         if self.feature == 'id+text':
    #             item_emb = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
    #         score = torch.matmul(torch.cat(last_item_embeddings, 0), item_emb.transpose(0, 1))
    #
    #     return score.cpu().numpy()
    # def predict(self, aug_seq, seq, pos, seq_len, masked):
    def predict(self,  seq, pos, seq_len):
        with torch.no_grad():
            seq_emb = self.model.forward(seq, pos)
            last_item_embeddings = [seq_emb[i, last - 1, :].view(-1, self.emb_size) for i, last in enumerate(seq_len)]
            item_emb = self.model.item_emb
            if self.feature == 'text':
                item_emb=self.model.mlps(self.model.bert_tensor)
                # item_emb = self.model.bert_tensor
            if self.feature == 'id+text':
                item_emb = self.model.mlps(self.model.bert_tensor) + self.model.item_emb
            score = torch.matmul(torch.cat(last_item_embeddings, 0), item_emb.transpose(0, 1))

        return score.cpu().numpy()

    def count_tensor_elements(self, tensor, max_value):

        count_list = [0] * (max_value + 1)

        for element in tensor.reshape(-1):
            count_list[int(element.item())] += 1
        # print(count_list)
        return count_list
    def find_popular_num(self):
        list_popular=list(self.data1)

        # popularity_updated = [3, 5, 32, 31, 2, 88, 90, 12, 2, 1]


        num_to_select_updated = max(1, int(len(list_popular) * 0.2))

        # 找出流行度最高的元素的索引
        top_indices_updated = sorted(range(len(list_popular)), key=lambda i: list_popular[i], reverse=True)[
                              :num_to_select_updated]

        # 输出索引和相应的流行度
        top_popularity_updated = [list_popular[index] for index in top_indices_updated]
        print(top_indices_updated[0])
        del top_indices_updated[0]
        boundary_popularity = min(top_popularity_updated)
        print(boundary_popularity)
        return boundary_popularity,top_indices_updated


    def  count_distance(self):
        if self.feature == 'text':
            epoch0 =  torch.load("./dataset/Amazon-Office/whole_tensor.pt",map_location='cuda:0')
            # epochfinal=self.model.mlps(self.model.bert_tensor)
            epochfinal = self.model.bert_tensor[1:]
        if self.feature == 'id':
            #其实这个是id在epoch1的时候，应该把
            epoch0 = torch.load("./epoch_0_tensor.pt", map_location='cuda:1')
            epoch0 = epoch0[1:]
            epochfinal = self.model.item_emb[1:]
        distances=torch.sqrt(torch.sum((epoch0 - epochfinal) ** 2, dim=1))
        average_distance = torch.mean(distances)
        pupularnum, top_index = self.find_popular_num()
        top_index = [*map(lambda x: x - 1, top_index)]
        # print(top_index[:10])
        pop_distance=torch.mean(distances[top_index])

        print("average_distance",average_distance)
        print("popular_distance",pop_distance)
        return average_distance,pop_distance

    def uniformity_loss(self, label, pos, data, t=2):

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
        # label = label[np.where(data[label] >= 10)]

        x = item_view[label]

        x = x.reshape([-1, 64])
        # realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5 * x.shape[0]))
        # neg_emb = item_view[realneg]
        # neg_emb = neg_emb.reshape([-1, 64])

        # neg_emb = neg_emb[:int(2.5*x.shape[0])]
        # x=neg_emb
        # x = torch.cat([x, neg_emb], dim=0)

        x = F.normalize(x, dim=-1)

        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

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

        label = label[np.where(pos != 0)]
        x = item_view[label]
        x = x.reshape([-1, 64])

        realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5 * x.shape[0]))
        neg_emb = item_view[realneg]
        neg_emb = neg_emb.reshape([-1, 64])

        # neg_emb = neg_emb[:int(2.5*x.shape[0])]
        x = torch.cat([x, neg_emb], dim=0)
        # list1=self.optimized_find_index_in_final_counts(final_counts,x.shape[0])
        x = F.normalize(x, dim=-1)
        dists = torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
        n = x.size(0)

        i = 0
        for num in range(0, len(cumulative_counts)):
            j = cumulative_counts[num]
            # print("dist",dists)

            dist1 = torch.pdist(x[i:j], p=2).pow(2).mul(-t).exp().mean().log()
            # print(dist1)
            if (torch.isnan(0.8 * dist1 / len(cumulative_counts)) == 0):
                dists = dists - 0.8 * dist1 / len(cumulative_counts)

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
        Xdata = data[label].reshape(-1)

        x = item_view[label]
        x = x.reshape([-1, 64])
        # print(x.shape)
        realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5 * x.shape[0]))
        neg_emb = item_view[realneg]
        neg_emb = neg_emb.reshape([-1, 64])
        Ydata = torch.ones(int(1.5 * x.shape[0]))
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

        distance = distance / 10
        # distance=torch.tensor(distance)

        # return torch.div(torch.pdist(x, p=2).pow(2),distance).mul(-t).exp().mean().log()

        return torch.div(torch.pdist(x, p=2).pow(2).mul(-t).exp(), distance).mean().log()







    def multiply_tensor_elements(tensor):
        """
        Multiply each element of the tensor with each subsequent element of the same tensor.
        """
        result = []
        for i in range(len(tensor)):
            for j in range(i + 1, len(tensor)):
                result.append(tensor[i] * tensor[j])

        return torch.tensor(result)
