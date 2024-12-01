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
import os
from data.sequence import Sequence
import torch.nn.functional as F
from util.conf import OptionConf,ModelConf
from data.loader import FileIO
from transformers import AutoTokenizer, AutoModel
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer

# Paper: Self-Attentive Sequential Recommendation

class Pretrain(object):
    def __init__(self, data,datasetfile,mask):
        # super(pretrain, self).__init__(conf, training_set, test_set)
        self.datasetfile=datasetfile
        self.data=data
        self.bert=Bert().cuda()
        self.llama_config = LlamaConfig.from_pretrained('/usr/gao/cwh/LLaMA')
        self.llama_config.num_hidden_layers = 7
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True

        self.llm_model = LlamaModel.from_pretrained(
        # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
        '/usr/gao/cwh/LLaMA',
        trust_remote_code=True,
        local_files_only=True,
        config=self.llama_config,).cuda()

        for param in self.llm_model.parameters():
            param.requires_grad = False
        initializer = nn.init.xavier_uniform_
        #whole_tensor=nn.Parameter(initializer(torch.empty(1,768))).cuda()
        #for dataset in self.datasetfile.split(","):

        self.functionName=(datasetfile.split("/")[2]).split("-")[0]
        self.train_inputs, self.train_masks,emb_id = feature.load_feature(self.data.id2item,self.datasetfile)

        # whole_list = []
        # i = 0
        # while len(self.train_inputs) > ((i+1) * 100):
        #     outputs = self.bert(self.train_inputs[i*100:(i+1)*100].cuda(), self.train_masks[i*100:(i+1)*100].cuda())[0][:, 0, :]
        #     whole_list.append(outputs)
        #     i = i + 1
        # #经过bert后取[0][:, 0, :]的意思是用bert中一个单词代表整个句子
        # outputs = self.bert(self.train_inputs[i*100:len(self.train_inputs)].cuda(), self.train_masks[i*100:len(self.train_inputs)].cuda())[0][:, 0, :]
        # #tensor_size = [outputs.shape[0], outputs.shape[1]]
        # whole_list.append(outputs)
        # #把所有编码过的item拼接成一个大tensor
        # whole_tensor = whole_list[0]
        # for i in range(1, len(whole_list)):
        #     whole_tensor = torch.cat([whole_tensor, whole_list[i]], 0)
        #
        # torch.save(whole_tensor, self.datasetfile+"whole_tensor.pt")

        whole_tensor_path = self.datasetfile + "whole_tensor.bin"
        with open(whole_tensor_path, 'wb') as f:
            i = 0
            batch_size =3

            while len(self.train_inputs) > (i + 1) * batch_size:
                print(i, 'i')
                # print((self.train_inputs[i * batch_size:(i + 1) * batch_size]).shape )
                outputs = self.llm_model(
                    self.train_inputs[i * batch_size:(i + 1) * batch_size].cuda(),
                    self.train_masks[i * batch_size:(i + 1) * batch_size].cuda()

                ).last_hidden_state
                # print(outputs.size())
                outputs= outputs[:, int(emb_id),:].cpu().numpy()
                # 打印输出的形状


                # 确保输出的维度正确，然后进行索引
                # outputs= outputs[:, 0, :].cpu().numpy()
                # 将 NumPy 数组写入文件
                np.save(f, outputs)

                i += 1

            # 处理剩余的数据
            outputs = self.llm_model(
                self.train_inputs[i * batch_size:len(self.train_inputs)].cuda(),
                self.train_masks[i * batch_size:len(self.train_inputs)].cuda()
            ).last_hidden_state[:, emb_id, :].cpu().numpy()

            # 将 NumPy 数组写入文件
            np.save(f, outputs)



            # 加载并合并所有保存的批次文件
            whole_tensor = self.load_and_concat_tensors(whole_tensor_path)
            print(whole_tensor.shape)
            # 保存合并后的张量
            torch.save(whole_tensor, self.datasetfile + "whole_tensor.pt")
    def execute(self):
        pass

    def load_and_concat_tensors(self,file_path):
        tensors = []
        with open(file_path, 'rb') as f:
            while True:
                try:
                    tensor = np.load(f)
                    tensors.append(torch.tensor(tensor))
                except ValueError:  # Reached the end of the file
                    break
        return torch.cat(tensors, dim=0)
def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    # print(model_output.shape)

    print(token_embeddings.shape)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        #试试看其他模型
        # self.bert = BertModel.from_pretrained('bert')
        self.bert = AutoModel.from_pretrained('all-mpnet-base-v2')
        for param in self.bert.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        #这两行是有all-mpnet-base-v2时的编码操作
        ####
        sentence_embeddings = mean_pooling(outputs, attention_mask)
        outputs = F.normalize(sentence_embeddings, p=2, dim=1)
        ####

        return outputs