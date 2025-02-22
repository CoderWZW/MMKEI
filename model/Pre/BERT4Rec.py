import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base.seq_recommender import SequentialRecommender
from util.conf import OptionConf
from util.sampler import next_batch_sequence
from util.loss_torch import l2_reg_loss
from util.structure import PointWiseFeedForward
from random import sample
from math import floor
import random
import math
import pandas as pd
from datetime import datetime
from data.pretrain import Pretrain
import os
from util.loss_torch import bpr_loss, l2_reg_loss, InfoNCE
from model.Module.BERT4Rec_module import MLPS
from model.Module.BERT4Rec_module import BERT_Encoder
from util.adjust_grad import assign_values_linear,adjust_learning_rate_of_rows
import sys
sys.path.append("/usr/gao/cwh/plot_pca_embedding")
#
# from plot_pca_embedding import plot_pca_embedding
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.set_device(0)
current_device = torch.cuda.current_device()

# Paper: BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer, CIKM'19
class BERT4Rec(SequentialRecommender):
    def __init__(self, conf, training_set, test_set):
        super(BERT4Rec, self).__init__(conf, training_set, test_set)
        args = OptionConf(self.config['BERT4Rec'])
        block_num = int(args['-n_blocks'])
        drop_rate = float(args['-drop_rate'])
        datasetFile = self.config['dataset']
        self.eps = float(args['-eps'])
        head_num = int(args['-n_heads'])
        self.cl_type = args['-cltype']
        self.cl_rate = float(args['-lambda'])
        self.cl = float(args['-cl'])
        self.strategy=float(args['-strategy'])
        self.aug_rate = float(args['-mask_rate'])
        self.model = BERT_Encoder(self.data, self.emb_size, self.max_len, block_num,head_num,drop_rate,self.feature,datasetFile,self.strategy)
        # print(self.data1)
        self.listcountitem=[0] * (self.data.item_num + 2)
    def train(self):

        model = self.model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        listdistance_ave = []
        listdistance_pop = []
        listHR = []
        listNDCG=[]

        for epoch in range(self.maxEpoch):
            self.listcountitem = [0] * (self.data.item_num + 2)
            model.train()

            #self.fast_evaluation(epoch)
            for n, batch in enumerate(next_batch_sequence(self.data, self.batch_size,self.labelgap,max_len=self.max_len)):
                seq, seqfull, pos, posfull, y, neg_idx, seq_len ,labelgap= batch

                aug_seq, masked, labels = self.item_mask_for_bert(seq, seq_len, self.aug_rate, self.data.item_num+1)
                self.listcountitem = np.sum([self.count_tensor_elements(seq, self.data.item_num), self.listcountitem],
                                       axis=0)
                seq_emb = model.forward(aug_seq, pos)
                if self.cl == 1:
                    cl_loss = self.cl_rate * self.cal_cl_loss(labels,seq_emb,masked)
                else:
                    cl_loss = 0


                rec_loss = self.calculate_loss(seq_emb,masked,labels)

                # batch_loss = cl_loss+rec_loss
                batch_loss = cl_loss+rec_loss
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 50==0:
                    # print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item(), 'uni_item_loss:',uni_item_loss.item())
                    print('training:', epoch + 1, 'batch', n, 'rec_loss:', rec_loss.item())

               
            # if epoch == 0:
            #     torch.save(self.model.item_emb, f'epoch_{epoch}_bert4rec_tensor.pt')
            model.eval()
            # self.fast_evaluation(epoch,self.data1)
            measure, HR,NDCG = self.fast_evaluation(epoch, self.data1)
            # ave, pop = self.count_distance()
            # listdistance_ave.append(ave.item())
            # listdistance_pop.append(pop.item())
            listHR.append(HR)
            listNDCG.append(NDCG)
        # self.draw()
        # torch.save(model.state_dict(), './model/checkpoint/'+self.feature+'/BERT4Rec.pt')

        # plot_pca_embedding(self.model.word_embeddings5.detach().cpu().numpy(), 'llama',
        #                    self.model.item_emb.detach().cpu().numpy(), 'item',
        #                    self.model.MoE.word_embeddinglist[0].detach().cpu().numpy(), 'word')
        #train_los.close()


    def item_mask_for_bert(self,seq,seq_len, mask_ratio, mask_idx):
        augmented_seq = seq.copy()
        masked = np.zeros_like(augmented_seq)
        labels = []
        for i, s in enumerate(seq):
            to_be_masked = random.sample(range(seq_len[i]), max(floor(seq_len[i]*mask_ratio),1))
            masked[i, to_be_masked] = 1
            # print("masked",masked)
            labels =labels+ list(augmented_seq[i, to_be_masked])
            augmented_seq[i, to_be_masked] = mask_idx
        return augmented_seq, masked, np.array(labels)

    def calculate_loss(self, seq_emb, masked, labels):
        
        masked=torch.tensor(masked)
        seq_emb = seq_emb[masked>0]
        seq_emb=seq_emb.view(-1, self.emb_size)
        if self.feature == 'text':
            emb=self.model.mlps(self.model.bert_tensor)
            # emb = self.model.bert_tensor
        elif self.feature == 'id':
            emb= self.model.item_emb
        elif self.feature == 'id+text':
            emb = self.model.item_emb+self.model.mlps(self.model.bert_tensor)
        logits = torch.mm(seq_emb, emb.t())
        #F.cross_entropy(logits, torch.tensor(labels).to(torch.int64).cuda())/labels.shape[0]
        loss = F.cross_entropy(logits, torch.tensor(labels).to(torch.int64).cuda())
        return loss

    def predict(self,seq, pos,seq_len,gap_batch):
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

            item_emb=self.model.item_emb
            if self.feature == 'text':
                  item_emb=self.model.mlps(self.model.bert_tensor)
                  # item_emb = self.model.bert_tensor
            if self.feature=='id+text':
                  item_emb=self.model.mlps(self.model.bert_tensor)+self.model.item_emb
            score = torch.matmul(torch.cat(last_item_embeddings, 0),  item_emb.transpose(0, 1))

        return score.cpu().numpy()
    def cal_cl_loss(self,label,seq_emb,masked):
        label=torch.tensor(label)
        # label=torch.unique(label)
        user_view = seq_emb[masked > 0]
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

        random_noise1 = torch.rand_like(item_view).cuda()
        item_view_1 =item_view+ torch.sign(item_view)* F.normalize(random_noise1, dim=-1) * self.eps
        random_noise2 = torch.rand_like(item_view).cuda()
        item_view_2 = item_view + torch.sign(item_view) * F.normalize(random_noise2, dim=-1) * self.eps


        random_noise3 = torch.rand_like(user_view).cuda()
        random_noise4 = torch.rand_like(user_view).cuda()
        user_view1=user_view+torch.sign(user_view)* F.normalize(random_noise3, dim=-1) * self.eps
        user_view2 = user_view + torch.sign(user_view) * F.normalize(random_noise4, dim=-1) * self.eps
        item_cl_loss_item = InfoNCE(item_view_1[label], item_view_2[label], 0.2)
        item_cl_loss_user=InfoNCE(user_view1, user_view2, 0.2)
        return  item_cl_loss_item
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
        # plt.figure(figsize=(12, 4))
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

        # plt.subplot(1, 2, 2)
        sns.jointplot(x=' ',y='  ', data=Pi,kind="kde",cmap="Blues", shade=True, shade_lowest=True)
        # plt.yticks(fontsize=18, weight='normal')
        plt.title("BERT4Rec+ID",y=-0.17,fontsize=20,weight='bold')
        plt.show()

        now = datetime.now()
        plt.savefig('./picture/BERT4Rec/fig'+str(datetime.now())+'.svg', dpi=300, bbox_inches='tight',format="svg")
        plt.close()
    pass


    def find_popular_num(self):
        list_popular = list(self.data1)

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
        return boundary_popularity, top_indices_updated

    def count_distance(self):
        if self.feature == 'text':
            epoch0 = torch.load("./dataset/Amazon-Office/whole_tensor.pt", map_location='cuda:0')
            # epochfinal=self.model.mlps(self.model.bert_tensor)
            epochfinal = self.model.bert_tensor[1:-1]
        if self.feature == 'id':
            # 其实这个是id在epoch1的时候，应该把
            epoch0 = torch.load("./epoch_0_tensor.pt", map_location='cuda:0')
            epoch0 = epoch0[1:-1]
            epochfinal = self.model.item_emb[1:-1]
        distances = torch.sqrt(torch.sum((epoch0 - epochfinal) ** 2, dim=1))
        average_distance = torch.mean(distances)
        pupularnum, top_index = self.find_popular_num()
        top_index = [*map(lambda x: x - 1, top_index)]
        # print(top_index[:10])
        pop_distance = torch.mean(distances[top_index])

        print("average_distance", average_distance)
        print("popular_distance", pop_distance)
        return average_distance, pop_distance
    def count_tensor_elements(self, tensor, max_value):

        count_list = [0] * (max_value + 2)

        for element in tensor.reshape(-1):
            count_list[int(element.item())] += 1
        # print(count_list)
        return count_list
def multiply_tensor_elements(tensor):
    """
    Multiply each element of the tensor with each subsequent element of the same tensor.
    """
    result = []
    for i in range(len(tensor)):
        for j in range(i + 1, len(tensor)):
            result.append(tensor[i] * tensor[j])

    return torch.tensor(result)
