import torch
import torch.nn as nn
import numpy as np
from base.seq_recommender import SequentialRecommender
from transformers import BertModel,GPT2LMHeadModel
from util.conf import OptionConf
from util.sampler import next_batch_sequence
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
from model.Module.FMLPRec_module import MLPS
from util.adjust_grad import assign_values_linear,adjust_learning_rate_of_rows
from model.Module.FMLPRec_module import FMLPRecModel
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from random import sample
#
torch.cuda.set_device(1)
current_device = torch.cuda.current_device()

class FMLPRec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(FMLPRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['FMLPRec'])
        datasetFile=self.config['dataset']
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        self.cl_rate = float(args['-lambda'])
        self.cl_type=args['-cltype']
        self.cl=float(args['-cl'])
        head_num = int(args['-n_heads'])
        self.uni = float(args['-uni'])
        self.model = FMLPRecModel(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate,self.feature,datasetFile)
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self.eps = float(args['-eps'])

    def train(self):
        model = self.model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        model_performance=[]
        listUNI = []

        listdistance_ave = []
        listdistance_pop = []
        listHR = []
        
        listcountitem = [0] * (self.data.item_num + 1)
        # print(sum(listcountitem))
        for epoch in range(self.maxEpoch):
            model.train()

            #self.fast_evaluation(epoch)
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,self.labelgap,max_len=self.max_len,)):
                seq, seqfull, pos, posfull, y, neg_idx, _,gap = batch
                # seq, pos, y, neg_idx, _ = batch
                listcountitem=np.sum([self.count_tensor_elements(y,self.data.item_num), listcountitem], axis=0).tolist()
                seq_emb = model.forward(seq, pos)

                if self.cl == 1:
                    cl_loss = self.cl_rate * self.cal_cl_loss(y,pos)
                else:
                    cl_loss=0

                if self.uni == 1:
                    # Standardized Sampling
                    uni_loss = self.uniformity_loss(y, pos)
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
                # batch_loss = rec_loss+cl_loss+0.03*uni_loss
                batch_loss = rec_loss + cl_loss+0.01*uni_loss
                #
                #batch_loss = rec_loss+ l2_reg_loss(self.reg, model.item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                # model.bert_tensor.retain_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(),'uni_loss:',uni_loss,"UNI",UNI.item())
                if n % 200==0:
                    listUNI.append(UNI.item())
            model.eval()
            model_performance.append(model.state_dict)
            # self.fast_evaluation(epoch)
            measure,HR,NDCG = self.fast_evaluation(epoch, self.data1)
            # ave, pop = self.count_distance()
            # listdistance_ave.append(ave.item())
            # listdistance_pop.append(pop.item())
            listHR.append(HR)
        # with open("./count_pantry.txt", 'w') as train_los:
        #     train_los.write(str(listcountitem))

        # torch.save(model_performance[self.bestPerformance[0]-1], './model/checkpoint/'+self.feature+'/FMLPRec.pt')

        # self.draw()
        # with open("./train_loss_SASRec_1.txt", 'a') as train_los:
        #     train_los.write(str(listUNI) + '\n')
        #
        # with open("./train_HR_and_distance_FMLPRec_1_office.txt", 'a') as train_HR_and_distance:
        #     train_HR_and_distance.write(str(listdistance_ave) + '\n')
        #     train_HR_and_distance.write(str(listdistance_pop) + '\n')
        #     train_HR_and_distance.write(str(listHR) + '\n')
    def calculate_loss(self, seq_emb, y, neg,pos):

        y = torch.tensor(y)
        neg = torch.tensor(neg)
        if (self.feature == 'text'):

            # y_emb = self.model.mlps(self.model.bert_tensor[y])
            outputs = self.model.bert_tensor[y.cuda()]
            y_emb=outputs

            # new_inputs = self.model.train_inputs[neg]
            # new_masks = self.model.train_masks[neg]

            # neg_emb = self.model.mlps(self.model.bert_tensor[neg])
            outputs = self.model.bert_tensor[neg.cuda()]
            neg_emb = outputs
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

    def predict(self,seq, pos,seq_len,gap):
        with torch.no_grad():
            seq_emb = self.model.forward(seq,pos)
            last_item_embeddings = [seq_emb[i,last-1,:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]
            item_emb=self.model.item_emb
            if self.feature == 'text':
                  # item_emb=self.model.mlps(self.model.bert_tensor)
                  item_emb = self.model.bert_tensor
            if self.feature=='id+text':
                  item_emb=self.model.mlps(self.model.bert_tensor)+self.model.item_emb
            score = torch.matmul(torch.cat(last_item_embeddings, 0),  item_emb.transpose(0, 1))

        return score.cpu().numpy()

    def cal_cl_loss(self,y,pos):
        y=torch.tensor(y)
        label=y[np.where(pos!=0)]
        # label = torch.unique(label)
        item_view=self.model.item_emb
        if self.feature == 'text':
            item_view= self.model.mlps(self.model.bert_tensor)
        if self.feature=='id+text':
            if (self.cl_type == 'id'):
                item_view = self.model.item_emb
            elif(self.cl_type == 'text'):
                item_view = self.model.mlps(self.model.bert_tensor)
            else:
                item_view = self.model.mlps(self.model.bert_tensor)+self.model.item_emb

        random_noise1 = torch.rand_like(item_view).cuda()
        random_noise2 = torch.rand_like(item_view).cuda()
        item_view_1 =item_view+ torch.sign(item_view) * F.normalize(random_noise1, dim=-1) * self.eps

        item_view_2 = item_view + torch.sign(item_view) * F.normalize(random_noise2, dim=-1) * self.eps
        item_cl_loss = InfoNCE(item_view_1[label] , item_view_2[label] , 0.2)
        return  item_cl_loss

    def draw(self):
        ItemInd = [i for i in range(self.data.item_num)]
        ItemInd = random.sample(ItemInd, self.data.item_num)
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
        item_view = item_view[1:]
        Pi=item_view.cpu().detach().numpy()
        import seaborn as sns
        sns.set_theme(style="white")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 20), dpi=100)
        plt.rc('font', weight='bold')
        from sklearn.manifold import TSNE
        Pi=TSNE(n_components=2, perplexity=100, learning_rate=200).fit_transform(Pi)
        colors=[]
        for i in range(0,len(Pi)):
            k=math.sqrt(Pi[i][0]*Pi[i][0]+Pi[i][1]*Pi[i][1])
            Pi[i][0]=(Pi[i][0]/k)
            Pi[i][1] = (Pi[i][1]/k)
            # colors.append(Pi[i][0]+Pi[i][1])
        # print(tsne.view())
        x1 = np.array(Pi[ItemInd, 0])
        y1 = np.array(Pi[ItemInd, 1])
        # s1 = plt.scatter(x1, y1, c='lightsteelblue',alpha=0.01,s=170)
        # plt.xticks(fontsize=18, weight='normal')
        columns = [' ', '  ']
        Pi = pd.DataFrame(Pi, columns = columns)

        sns.jointplot(x=' ',y='  ', data=Pi,kind="kde",cmap="Blues", shade=True, shade_lowest=True)
        # plt.yticks(fontsize=18, weight='normal')
        plt.title("SASRec+ID",y=-0.17,fontsize=20,weight='bold')
        plt.show()
        now = datetime.now()
        plt.savefig('./picture/fig'+str(datetime.now())+'.svg', dpi=300, bbox_inches='tight',format="svg")
        plt.close()
    pass
    def uniformity_loss_designed(self, label,pos,neg,t=2):

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
        labelforcount=label.clone().detach()
        non_zero_counts = np.count_nonzero(labelforcount, axis=1)
        cumulative_counts = np.cumsum(non_zero_counts)
        cumulative_counts.tolist()


        label = label[np.where(pos != 0)]
        x = item_view[label]
        x = x.reshape([-1, 64])

        realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5* x.shape[0]))
        neg_emb = item_view[realneg]
        neg_emb = neg_emb.reshape([-1, 64])

        # neg_emb = neg_emb[:int(2.5*x.shape[0])]
        x = torch.cat([x, neg_emb], dim=0)
        x = F.normalize(x, dim=-1)
        # list1=self.optimized_find_index_in_final_counts(final_counts,x.shape[0])

        dists = torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
        n = x.size(0)

        i = 0

        for num in range(0,len(cumulative_counts)):
             j = cumulative_counts[num]
             # print("dist",dists)

             dist1=torch.pdist(x[i:j], p=2).pow(2).mul(-t).exp().mean().log()
             # print(dist1)
             if(torch.isnan(0.8*dist1/len(cumulative_counts))==0  ):
                dists = dists-0.8*dist1/len(cumulative_counts)

             i=j
        result = dists
        return result
    def uniformity_loss(self, label,pos,t=2):
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
        # print(len(label))
        x=item_view[label]
        x=x.reshape([-1,64])
        realneg=np.random.choice(list(range(1, self.data.item_num+1)), int(2.5* x.shape[0]))
        # realneg = random.sample(list(range(1, self.data.item_num+1)), int(2.5* x.shape[0]))
        neg_emb = item_view[realneg]
        neg_emb = neg_emb.reshape([-1, 64])

        #neg_emb = neg_emb[:int(2.5*x.shape[0])]
        # x=neg_emb
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

    def count_tensor_elements(self, tensor, max_value):

        count_list = [0] * (max_value + 1)

        for element in tensor.reshape(-1):
            count_list[int(element.item())] += 1
        # print(count_list)
        return count_list

    def uniformity_loss_popularity(self, label, pos, neg, data,t=2):
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
        data=torch.tensor(data)


        label = label[np.where(pos != 0)]
        Xdata = data[label].reshape(-1)
        # print("cc",Xdata.shape)
        x = item_view[label]
        x = x.reshape([-1, 64])
        # print(x.shape)
        realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5 * x.shape[0]))
        neg_emb = item_view[realneg]
        neg_emb = neg_emb.reshape([-1, 64])
        Ydata=torch.ones(int(1.5 * x.shape[0]))
        '''
        Ydata = data[realneg]
        Ydata[Ydata == 0] = 1
        '''
        data=torch.cat([Xdata, Ydata], dim=0).cuda()


        x = torch.cat([x, neg_emb], dim=0)
        x = F.normalize(x, dim=-1)
        # data=data.view(-1, 1)

        # data= multiply_tensor_elements(data)
        distance= torch.triu(torch.ger(data,data),  diagonal=1)
        distance=distance[distance != 0]

        #around 400
        distance=distance/10
        # print(distance.mean())
        # return torch.div(torch.pdist(x, p=2).pow(2),distance).mul(-t).exp().mean().log()
        return torch.div(torch.pdist(x, p=2).pow(2).mul(-t).exp(), distance).mean().log()

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
            epoch0 = torch.load("./epoch_0_tensor.pt", map_location='cuda:0')
            epoch0 = epoch0[1:]
            epochfinal = self.model.item_emb[1:]
        distances=torch.sqrt(torch.sum((epoch0 - epochfinal) ** 2, dim=1))
        average_distance = torch.mean(distances)
        pupularnum, top_index = self.find_popular_num()
        top_index = [*map(lambda x: x - 1, top_index)]
        print(top_index[:10])
        pop_distance=torch.mean(distances[top_index])

        print("average_distance",average_distance)
        print("popular_distance",pop_distance)
        return average_distance,pop_distance
