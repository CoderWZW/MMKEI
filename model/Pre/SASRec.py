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
from model.Module.SASRec_module import MLPS
from model.Module.SASRec_module import SASRec_Model
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from util.adjust_grad import assign_values_linear,adjust_learning_rate_of_rows
from random import sample
import os
import sys
sys.path.append("/usr/gao/cwh/plot_pca_embedding")

# 导入模块
from plot_pca_embedding import plot_pca_embedding,plot_tsne_embedding
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Paper: Self-Attentive Sequential Recommendation
torch.cuda.set_device(0)
current_device = torch.cuda.current_device()





class SASRec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(SASRec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['SASRec'])
        datasetFile=self.config['dataset']
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        self.cl_rate = float(args['-lambda'])
        self.cl_type=args['-cltype']
        self.cl=float(args['-cl'])
        head_num = int(args['-n_heads'])
        self.uni = float(args['-uni'])
        self.model = SASRec_Model(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate,self.feature,datasetFile)
        self.rec_loss = torch.nn.BCEWithLogitsLoss()
        self.eps = float(args['-eps'])

        self.listcountitem = [0] * (self.data.item_num + 1)
    def train(self):

        model = self.model.cuda()

        #关于梯度更新调整的模块实验
        # param = self.model.bert_tensor # 例如模型某层的权重
        # param = self.model.mlps(self.model.bert_tensor)
        # 使用.register_hook来应用钩子
        # 这里假设你想要为param的第0行和第1行设置不同的学习率
     
        # rows_to_adjust=[i for i in range(self.data.item_num + 1)][1:]
        # hook=param.register_hook(lambda grad: adjust_learning_rate_of_rows(grad, rows_to_adjust, 0.005,self.lRate,self.listcountitem))



        optimizer = torch.optim.Adam(model.parameters(), self.lRate)

        # special_parameters =self.model.item_emb[1:][self.find_popular_num()[1]]
        # base_parameters = [p for p in model.parameters() if p not in special_parameters]
        # optimizer = torch.optim.Adam([{'params':special_parameters, 'lr': 0.005},{'params': base_parameters, 'lr': self.lRate}])

        # optimizer1 = torch.optim.Adam([self.model.bert_tensor], lr=self.lRate)
        model_performance=[]
        listUNI = []

        listdistance_ave=[]
        listdistance_pop=[]
        listHR=[]
        listNDCG=[]

        # print(sum(listcountitem))
        for epoch in range(self.maxEpoch):
            self.listcountitem = [0] * (self.data.item_num + 1)
            model.train()

            #self.fast_evaluation(epoch)
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,self.labelgap,max_len=self.max_len)):

                seq, seqfull, pos,posfull, y, neg_idx, _,gap = batch
                self.listcountitem=np.sum([self.count_tensor_elements(seq,self.data.item_num), self.listcountitem], axis=0).tolist()
                seq_emb = model.forward(seq, pos)

                if self.cl == 1:
                    cl_loss = self.cl_rate * self.cal_cl_loss(y,pos)
                else:
                    cl_loss=0

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
                # batch_loss = rec_loss+cl_loss+0.03*uni_loss
                batch_loss = rec_loss + cl_loss+0.005*uni_loss
                #可选择加正则化
                #batch_loss = rec_loss+ l2_reg_loss(self.reg, model.item_emb)
                # Backward and optimize
                optimizer.zero_grad()
                # optimizer1.zero_grad()
                # print(listcountitem)
                # self.model.bert_tensor.retain_grad()
                batch_loss.backward()

                #判断勾子是否调用
                # if param._backward_hooks:
                #     print("Hook(s) registered.")
                # else:
                #     print("No hooks registered.")

                # print(self.model.bert_tensor.grad)
                optimizer.step()

                if n % 50==0:
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(),'uni_loss:',uni_loss ,"UNI",UNI.item())
                if n % 200==0:
                    listUNI.append(UNI.item())
            
            #用于检查偏移量
            # if epoch == 0:
            #     torch.save(self.model.item_emb, f'epoch_{epoch}_tensor.pt')
            model.eval()
            model_performance.append(model.state_dict)
            measure,HR,NDCG=self.fast_evaluation(epoch,self.data1)
            #self.count_distance_2()
            # listdistance_ave.append(ave.item())
            # listdistance_pop.append(pop.item())
            listHR.append(HR)
            listNDCG .append(NDCG)
        # target_embedding0 = self.model.MoE.experts[0].saved_target_embedding.detach().cpu().numpy()
        # source_embedding0 = self.model.MoE.experts[0].saved_source_embedding.detach().cpu().numpy()

        # # 保存为 .npy 文件
        # np.save('target_embedding0_e.npy', target_embedding0)
        # np.save('source_embedding0_e.npy', source_embedding0)

        # # 调用 plot_tsne_embedding 函数
        # plot_tsne_embedding(target_embedding0, 'Item', source_embedding0, 'Word')


        #
        # with open("./count_cell.txt", 'w') as train_los:
        #     train_los.write(str(self.listcountitem))

        # torch.save(model_performance[self.bestPerformance[0]-1], './model/checkpoint/'+self.feature+'/SASRec.pt')

        # self.count_distance()

        # self.drawtsne()

        #之前存uni值用的
        # with open("./train_loss_LinRec_movie.txt", 'a') as train_los:
        #     train_los.write(str(listUNI) + '\n')

        #存偏移量
        # with open("./train_HR_and_distance_SASRec_beauty_grad_adjust.txt", 'a') as train_HR_and_distance:
        #     train_HR_and_distance.write(str(listdistance_ave) + '\n')
        #     train_HR_and_distance.write(str(listdistance_pop) + '\n')
        #     train_HR_and_distance.write(str(listHR) + '\n')
        #     train_HR_and_distance.write(str(listNDCG) + '\n')
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

    def predict(self,seq, pos,seq_len,gap):
        with torch.no_grad():
            seq_emb = self.model.forward(seq,pos)
            last_item_embeddings = [seq_emb[i,last-1,:].view(-1,self.emb_size) for i,last in enumerate(seq_len)]
            item_emb=self.model.item_emb
            if self.feature == 'text':
                  item_emb=self.model.mlps(self.model.bert_tensor)
                  # item_emb=self.model.bert_tensor
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
            # item_view= self.model.mlps(self.model.bert_tensor)
            item_view = self.model.bert_tensor
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

    def drawtsne(self):
        ItemInd = [i for i in range(1,int(self.data.item_num))]



        import seaborn as sns
        sns.set_theme(style="white")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 20), dpi=100)
        plt.rc('font', weight='bold')
        from sklearn.manifold import TSNE


        item_view2 = self.model.mlps(self.model.bert_tensor)
        item_view2 = item_view2[list(np.where(self.data1 >= 8)[0])]

        ItemInd2 = random.sample([i for i in range(0,len(item_view2))],int(len(item_view2)/10))

        Pi2 = item_view2.cpu().detach().numpy()
        Pi2 = TSNE(n_components=2, perplexity=100, learning_rate=200).fit_transform(Pi2)
        x2 = np.array(Pi2[ItemInd2,0])

        y2 = np.array(Pi2[ItemInd2,1])
        s2 = plt.scatter(x2, y2, c='blue', alpha=0.5, s=170)

        item_view1 = self.model.mlps(self.model.bert_tensor)
        item_view1 = item_view1[1:]
        Pi1 = item_view1.cpu().detach().numpy()
        ItemInd1 = random.sample(ItemInd, int((self.data.item_num) / 10))
        Pi1 = TSNE(n_components=2, perplexity=100, learning_rate=200).fit_transform(Pi1)

        x1 = np.array(Pi1[ItemInd1,0])
        y1 = np.array(Pi1[ItemInd1,1])
        s1 = plt.scatter(x1, y1, c='red', alpha=0.5, s=170)
        # plt.xticks(fontsize=18, weight='normal')
        columns = [' ', '  ']

        # Pi2 = pd.DataFrame(Pi2, columns=columns)
        # sns.jointplot(x=' ', y='  ', data=Pi2, kind="kde", cmap="Reds", shade=True, shade_lowest=True)
        #
        # Pi1 = pd.DataFrame(Pi1, columns=columns)
        #
        # sns.jointplot(x=' ', y='  ', data=Pi1, kind="kde", cmap="Blues", shade=True, shade_lowest=True)
        plt.title("SASRec", y=-0.17, fontsize=20, weight='bold')
        plt.show()

        plt.savefig('./picture/fig' + str(datetime.now()) + '.svg', dpi=300, bbox_inches='tight', format="svg")
        plt.close()

    def draw(self):
        ItemInd = [i for i in range(1,self.data.item_num+1)]
        # ItemInd = random.sample(ItemInd, self.data.item_num)
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
        item_view1 = item_view[1:]


        Pi=item_view1.cpu().detach().numpy()

        import seaborn as sns
        sns.set_theme(style="white")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 20), dpi=100)
        plt.rc('font', weight='bold')
        from sklearn.manifold import TSNE
        Pi=TSNE(n_components=2, perplexity=100, learning_rate=200).fit_transform(Pi)
        Picold = TSNE(n_components=2, perplexity=100, learning_rate=200).fit_transform(Picold)
        colors=[]
        for i in range(0,len(Pi)):
            k=math.sqrt(Pi[i][0]*Pi[i][0]+Pi[i][1]*Pi[i][1])
            Pi[i][0]=(Pi[i][0]/k)
            Pi[i][1] = (Pi[i][1]/k)
            # colors.append(Pi[i][0]+Pi[i][1])
        # print(tsne.view())
        x1 = np.array(Pi[:0])
        y1 = np.array(Pi[:1])


        columns = [' ', '  ']
        Pi = pd.DataFrame(Pi, columns = columns)



        sns.jointplot(x=' ',y='  ', data=Pi,kind="kde",cmap="Blues", shade=True, shade_lowest=True)

        # plt.yticks(fontsize=18, weight='normal')
        plt.title("SASRec",y=-0.17,fontsize=20,weight='bold')
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

        # realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5* x.shape[0]))
        # neg_emb = item_view[realneg]
        # neg_emb = neg_emb.reshape([-1, 64])
        #
        # # neg_emb = neg_emb[:int(2.5*x.shape[0])]
        # x = torch.cat([x, neg_emb], dim=0)
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



        x=item_view[label]
        x=x.reshape([-1,64])
        # realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5 * x.shape[0]))
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

        # print(len(data))
        label = label[np.where(pos != 0)]
        Xdata = data[label].reshape(-1)
        # print("cc",Xdata.shape)
        x = item_view[label]
        x = x.reshape([-1, 64])
        # print(x.shape)
        #realneg = random.sample(list(range(1, self.data.item_num + 1)), int(1.5 * x.shape[0]))
        #neg_emb = item_view[realneg]
        #neg_emb = neg_emb.reshape([-1, 64])
        #Ydata=torch.ones(int(1.5 * x.shape[0]))
        '''
        Ydata = data[realneg]
        Ydata[Ydata == 0] = 1
        '''
        Xdata[Xdata==0]=1
        data=Xdata.cuda()
        #data=torch.cat([Xdata, Ydata], dim=0).cuda()


        #x = torch.cat([x, neg_emb], dim=0)
        x = F.normalize(x, dim=-1)
        # data=data.view(-1, 1)

        # data= multiply_tensor_elements(data)
        distance= torch.triu(torch.ger(data,data),  diagonal=1)
        distance=distance[distance != 0]

        #around 400
        distance=distance/15
        # print(distance.mean())
        # return torch.div(torch.pdist(x, p=2).pow(2),distance).mul(-t).exp().mean().log()
        return torch.div(torch.pdist(x, p=2).pow(2).mul(-t).exp(), distance).mean().log()

    def find_popular_num(self):
        list_popular=list(self.data1)

        # popularity_updated = [3, 5, 32, 31, 2, 88, 90, 12, 2, 1]


        num_to_select_updated = max(1, int(len(list_popular) * 0.4))

        # 找出流行度最高的元素的索引
        top_indices_updated = sorted(range(len(list_popular)), key=lambda i: list_popular[i], reverse=True)[
                              :num_to_select_updated]

        # 输出索引和相应的流行度
        top_popularity_updated = [list_popular[index] for index in top_indices_updated]


        del top_indices_updated[0]
        print(min(top_indices_updated))
        boundary_popularity = min(top_popularity_updated)
        print(boundary_popularity)
        return boundary_popularity,top_indices_updated


    def  count_distance(self):
        if self.feature == 'text':
            epoch0 =  torch.load("./dataset/Amazon-Beauty/whole_tensor.pt",map_location='cuda:1')
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

        # print("average_distance",average_distance)
        # print("popular_distance",pop_distance)
    def  count_distance_2(self):
        if self.feature == 'text':
            epoch_total=self.model.mlps(self.model.bert_tensor)[1:]
        if self.feature=='id':
            epoch_total=self.model.item_emb[1:]
        pupularnum, top_index = self.find_popular_num()
        top_index = [*map(lambda x: x - 1, top_index)]
        epoch_pop=epoch_total[top_index]

        mask = torch.ones(epoch_total.size(0), dtype=torch.bool)  
        mask[top_index] = False 
        epoch_other = epoch_total[mask]
        

        sample_pop= random.sample(list(range(0, len(epoch_pop))),len(epoch_pop)//2)
        # print(len(epoch_pop))
        emb_selectpop = epoch_pop[sample_pop]
        emb_selectpop = emb_selectpop.reshape([-1, 64])
        emb_selectpop = F.normalize(emb_selectpop, dim=-1)
        # dist_pop=torch.pdist(emb_selectpop, p=2).pow(2).mul(-2).exp().mean().log()
        dist_pop=torch.pdist(emb_selectpop, p=2).mean()
        print('dist_pop',dist_pop)
       
        sample_other= random.sample(list(range(0, len(epoch_other))),len(epoch_other)//10)
        # print(len(epoch_other))
        emb_selectother = epoch_other[sample_other]
        emb_selectother = emb_selectother.reshape([-1, 64])
        emb_selectother = F.normalize(emb_selectother, dim=-1)
        # dist_other=torch.pdist(emb_selectother, p=2).pow(2).mul(-2).exp().mean().log()
        dist_other = torch.pdist(emb_selectother, p=2).mean()
        print('dist_other',dist_other)

   
