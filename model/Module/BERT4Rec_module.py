import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from util.structure import PointWiseFeedForward
import os

from math import sqrt
from data.pretrain import Pretrain
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer,XLMRobertaTokenizer, XLMRobertaModel,T5Tokenizer, T5Model,T5TokenizerFast
from torch.nn.functional import normalize
torch.cuda.set_device(0)
current_device = torch.cuda.current_device()
class BERT_Encoder(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate, feature, datasetFile):
        super(BERT_Encoder, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self.datasetFile = datasetFile
        self.feature = feature
        self._init_model()



        ##
        # self.bert_config = BertConfig.from_pretrained('/usr/gao/cwh/BGE')
        self.bert_config2 = BertConfig.from_pretrained('bert')
        #
        # self.bert_config.output_attentions = True
        # self.bert_config.output_hidden_states = True
        # self.llm_model =  XLMRobertaModel.from_pretrained(
        #     '/usr/gao/cwh/BGE',
        #
        #     local_files_only=True,
        #     config=self.bert_config,
        # )
        # self.tokenizer = XLMRobertaTokenizer.from_pretrained(
        #     '/usr/gao/cwh/BGE')
        self.llm_model2 =BertModel.from_pretrained(
            'bert',

            local_files_only=True,
            config=self.bert_config2,
        ).cuda()
        self.tokenizer2=BertTokenizer.from_pretrained(
            'bert')
        self.tokenizer3 = T5TokenizerFast.from_pretrained(
            '/usr/gao/cwh/P5-sportbase',legacy=False)
        self.llm_model3 =  T5Model.from_pretrained(
            '/usr/gao/cwh/P5-sportbase',

            local_files_only=True,
            # config=self.bert_config3,
        ).cuda()
        self.tokenizer4 = T5TokenizerFast.from_pretrained(
            '/usr/gao/cwh/P5-beautybase',legacy=False)
        self.llm_model4 = T5Model.from_pretrained(
            '/usr/gao/cwh/P5-beautybase',

            local_files_only=True,
            # config=self.bert_config3,
        ).cuda()
        self.llama_config = LlamaConfig.from_pretrained('/usr/gao/cwh/LLaMA')
        self.llama_config.num_hidden_layers =7
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.tokenizer5 =LlamaTokenizer.from_pretrained(
                    '/usr/gao/cwh/LLaMA')
        self.llm_model5 = LlamaModel.from_pretrained(
        '/usr/gao/cwh/LLaMA',
            local_files_only=True,
            config=self.llama_config,
        ).cuda()
        self.llm_modelgpt = GPT2Model.from_pretrained('/usr/gao/cwh/gpt2').cuda()
        self.tokenizer_gpt2=GPT2Tokenizer.from_pretrained('/usr/gao/cwh/gpt2')
        # self.word_embeddings1 = self.llm_model.get_input_embeddings().weight
        self.word_embeddings2 = self.llm_model2.get_input_embeddings().weight
        self.word_embeddings3 = self.llm_model3.get_input_embeddings().weight
        self.word_embeddings4 = self.llm_model4.get_input_embeddings().weight
        self.word_embeddings5 = self.llm_model5.get_input_embeddings().weight
        are_equal = torch.equal(self.word_embeddings3, self.word_embeddings4)
        self.word_embeddings6=self.word_embeddings2.clone()
        self.word_embeddings7 = self.word_embeddings2.clone()
        self.word_embeddings8=self.word_embeddings2.clone()
        self.word_embeddings9 = self.word_embeddings2.clone()
        self.word_embeddings10 = self.word_embeddings2.clone()
        self.word_embeddings3_1=self.word_embeddings3.clone()
        self.word_embeddings4_1 = self.word_embeddings4.clone()
        self.word_embedding_gpt2=self.llm_modelgpt.get_input_embeddings().weight
        self.word_embedding_gpt2_1 = self.word_embedding_gpt2.clone()
        self.word_embedding_gpt2_2 = self.word_embedding_gpt2.clone()
        # if not are_equal:
        #     # 计算差异
        #     difference = torch.abs(self.word_embeddings3 - self.word_embeddings4)
        #     max_difference = torch.max(difference)
        #     mean_difference = torch.mean(difference)
        #     #确实是不一样的
        #     print(f"两个张量不同，最大差异为: {max_difference.item()}, 平均差异为: {mean_difference.item()}")
        # else:
        #     print("两个张量完全相同")

        self.random_tensor1 = torch.rand_like(self.word_embeddings5 ).cuda()
        self.random_tensor2 = torch.rand_like(self.word_embeddings2 ).cuda()
        self.random_tensor3 =torch.rand_like(self.word_embeddings3 ).cuda()
        self.random_tensor4 = torch.rand_like(self.word_embeddings4 ).cuda()

        self.num_tokens = 1000
        self.mapping = MLPS_for_reprogram(len(self.word_embeddings2))
        self.reprogramming_layer = ReprogrammingLayer(64, 8, d_llm=768, d_keys=32)

        # tokenizer = [self.tokenizer5, self.tokenizer2, self.tokenizer3, self.tokenizer4]
        # llm = [self.llm_model5, self.llm_model2, self.llm_model3, self.llm_model4]
        tokenizer = [self.tokenizer5,self.tokenizer2,self.tokenizer3]
        llm = [self.llm_model5,self.llm_model2,self.llm_model3]
        self.MoE = MoE(64, 4, llm,
                       tokenizer)

        self.linear_layer = nn.Linear(30522, self.num_tokens)
        ##
    def _init_model(self):

        initializer = nn.init.xavier_uniform_

        if (self.feature == 'text' or self.feature == 'id+text'):

            self.bert_tensor = nn.Parameter(initializer(torch.empty(1,768))).cuda()

            if (len(self.datasetFile.split(",")) == 1):
                if not os.path.exists(self.datasetFile + "whole_tensor.pt"):
                    mask = 1
                    pre = Pretrain(self.data, self.datasetFile, mask)
                tensor = torch.load(self.datasetFile + "whole_tensor.pt")
                tensor = tensor.to(0)
                print(tensor)
                torch_mask = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)
                self.bert_tensor = torch.cat([self.bert_tensor, torch_mask], 0)
                
                self.bert_tensor = torch.nn.Parameter(self.bert_tensor.detach())
                self.mlps = MLPS(768)
                # self.bert_tensor.requires_grad_(True)
            elif (len(self.datasetFile.split(",")) > 1):
                torch_mask = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                for dataset in self.datasetFile.split(","):
                    if not os.path.exists(dataset + "whole_tensor.pt"):
                        mask = 1
                        pre = Pretrain(self.data, dataset, mask)
                    tensor = torch.load(dataset + "whole_tensor.pt")
                    self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)
                self.bert_tensor = torch.cat([self.bert_tensor, torch_mask], 0)
                
                # self.bert_tensor = torch.load('./dataset/fuse_tensor'+ filename+'.pt')
                self.mlps = MLPS(self.emb_size)
                # self.bert_tensor.requires_grad_(True)


        self.mlps = MLPS(768)


        self.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num + 2, self.emb_size)))
        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len + 2, self.emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()

        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)

        #basically the same with SASRec
        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_attn_layer = torch.nn.MultiheadAttention(self.emb_size, self.head_num, self.drop_rate)


            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate, 'gelu')
            self.forward_layers.append(new_fwd_layer)

    def freeze(self,layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
    def forward(self, seq, pos):
        seq = torch.tensor(seq)
        pos = torch.tensor(pos)
        if (self.feature == 'text'):
            # seq_emb=self.bert_tensor[seq.cuda()]
            seq_emb = self.mlps(self.bert_tensor[seq.cuda()])
        elif (self.feature == 'id'):
            seq_emb = self.item_emb[seq]
        elif (self.feature == 'id+text'):
            seq_emb = self.item_emb[seq] + self.mlps(self.bert_tensor[seq.cuda()])
        # seq_emb = self.item_emb[seq]




        seq_emb = seq_emb * self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        timeline_mask = torch.BoolTensor(seq == 0).cuda()
        seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        # tl = seq_emb.shape[1]
        # attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).cuda())
        for i in range(len(self.attention_layers)):
            #MoE模式1
            if(i==0):

                seq_emb_moe = self.MoE(seq_emb, seq.cuda(),[self.word_embeddings5,self.word_embeddings2,self.word_embeddings3])
            seq_emb = torch.transpose(seq_emb, 0, 1)
            normalized_emb = self.attention_layer_norms[i](seq_emb)
            mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=None)
            seq_emb = normalized_emb + mha_outputs
            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)

            seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
            if (i == 0):
                 seq_emb =  seq_emb+seq_emb_moe

        seq_emb = self.last_layer_norm(seq_emb)


        #MoE模式2
        # seq_emb_input = seq_emb
        # source_embeddings = self.mapping(self.word_embeddings2.permute(1, 0)).permute(1, 0)
        # # seq_emb = self.MoE(seq_emb, self.word_embeddings1,self.word_embeddings2,self.word_embeddings3,self.word_embeddings4)
        # seq_emb = self.reprogramming_layer(seq_emb, source_embeddings, source_embeddings)
        # seq_emb = seq_emb + seq_emb_input
        # seq_emb = self.last_layer_norm(seq_emb)


        return seq_emb


# encoder
class MLPS(nn.Module):
    def __init__(self, H):
        super(MLPS, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        
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
            nn.Linear(self.H, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
        )

    def forward(self, bert_tensor):
        # [batchsize,sequenceLen,large_item_embedding]->[batchsize,sequenceLen,small_item_embedding]
        logits = self.classifier(bert_tensor)

        # print("logits", logits.shape)
        # logits=torch.reshape(logits,(batch,m,self.H))
        return logits


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=1024, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, 64)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, llm,tokenizer,num_experts=3, dropout_rate=0.1, noisy_gating=False,
                 noise_epsilon=1e-2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon

        # 创建专家和对应的dropout层
        self.experts = nn.ModuleList([
            ReprogrammingLayer(input_dim, output_dim, d_llm=4096) if i == 0
            else ReprogrammingLayer(input_dim, output_dim, d_llm=768)
            for i in range(num_experts)
        ])

        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(num_experts)
        ])

        # Gating网络
        self.gating_network = GatingNetwork(input_dim, num_experts)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, num_experts))  # 用于noisy gating

        # Mapping层
        # mapping_sizes = [ 32000,50257, 50257,50257]
        # mapping_sizes = [32000, 30522, 32100,32100]
        mapping_sizes = [32000,30522]
        dimension_mapping_size=[4096]
        self.dimension_mappings= nn.ModuleList([
           nn.Linear(size,64) for size in dimension_mapping_size[:num_experts]
        ])
        self.mappings = nn.ModuleList([
            MLPS_for_reprogram(size) for size in mapping_sizes[:num_experts]
        ])
        txt_file_path = '/usr/gao/cwh/New_MoRec/ModalRec/ModalRec/dataset/Amazon-Music/gpt4o_introfuction'

        # Step 1: Load words from text file
        with open(txt_file_path, 'r') as file:
            word_list = [line.strip() for line in file if line.strip()]
        self.word_embeddinglist = []
        self.word_embeddinglist_copy=[]
        all_words_as_sentence = " ".join(word_list)
        for i in range(3):
            # 对每个单词独立进行 tokenization 和嵌入

            # 一次性进行 tokenization 和嵌入生成
            inputs = tokenizer[i](all_words_as_sentence, return_tensors="pt", padding=False, truncation=True).to('cuda')
            input_ids = inputs["input_ids"].cuda()

            # 获取所有单词的嵌入
            embeddings = llm[i].get_input_embeddings().cuda()(input_ids).detach().clone().requires_grad_(
                True)  # Shape: [1, token_length, embedding_dim]

            # 去掉 batch 维度并返回 embeddings
            embeddings = embeddings.squeeze(0)
            print(embeddings.shape)
            embedding_copy=torch.rand_like(embeddings)
            self.word_embeddinglist.append(embeddings)
            self.word_embeddinglist_copy.append(embedding_copy)



    def forward(self, seq_emb,seq,word_embeddings_list):
        # 获取gating logits
        clean_logits = self.gating_network(seq_emb)

        # 加噪音到gating logits
        if self.noisy_gating:
            raw_noise_stddev = seq_emb @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits).to(seq_emb.device) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        expert_weights = F.softmax(logits, dim=-1).permute(0, 2, 1).unsqueeze(-1)

        expert_outputs = []
        for i in range(self.num_experts):
            #Three strategy
            # source_embeddings = self.mappings[i](word_embeddings_list[i].permute(1, 0)).permute(1, 0)
            # source_embeddings = self.process_and_select(seq_emb,word_embeddings_list[i],self.dimension_mappings[i], seq)
            source_embeddings = self.encode_and_expand(word_embeddings_list[i], i)
            output = self.dropouts[i](self.experts[i](seq_emb, source_embeddings, source_embeddings))
            expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        output = torch.sum(expert_outputs * expert_weights, dim=1)

        return output

        # 使用 torch.topk 获取前 top-k 的专家索引及其权重
        # 使用 torch.topk 获取前 top-k 的专家索引及其权重
        # avg_logits = torch.mean(logits, dim=1)
        # top_k_logits, top_k_indices = torch.topk(avg_logits, 1, dim=-1)
        #
        # top_k_expert_weights = F.softmax(top_k_logits, dim=-1).unsqueeze(-1).unsqueeze(-1)
        #
        # expert_outputs = []
        # for i, idx in enumerate(top_k_indices[0]):  # 遍历 top_k_indices 只激活的专家
        #     # print(i,idx)
        #     # source_embeddings 的处理可以根据实际需要调整
        #     source_embeddings = self.encode_and_expand(word_embeddings_list[i], i)
        #     # source_embeddings = self.mappings[idx.item()](word_embeddings_list[idx.item()].permute(1, 0)).permute(1, 0)
        #     # source_embeddings = self.encode_and_expand(word_embeddings_list[idx.item()], idx.item())
        #     output = self.dropouts[idx.item()](self.experts[idx.item()](seq_emb, source_embeddings, source_embeddings))
        #     expert_outputs.append(output * top_k_expert_weights[:, i, :, :])  # 直接乘以该专家的权重
        #
        # # 将激活的专家输出求和（只计算激活专家的加权输出）
        # output = torch.sum(torch.stack(expert_outputs, dim=1), dim=1)
        # return output

    def process_and_select(self,A: torch.Tensor, B: torch.Tensor,linear_mapping,seq):

      

        batchsize, seq_len, vec_dim = A.shape
        _, dim = B.shape

        # Step 2: Create a mask where seq is not equal to 0, this is for non-padding items
        mask = seq != 0  # Shape: [batchsize, seq_len]

        # Step 3: Masked sum and count for averaging
        # Masked sum: sum over seq_len dimension only where mask is true
        A_sum = (A * mask.unsqueeze(-1)).sum(dim=1)  # Shape: [batchsize, 64]

        # Masked count: count how many items per batch (to compute average)
        item_count = mask.sum(dim=1).clamp(min=1)  # Shape: [batchsize], clamp to avoid division by zero

        # Step 4: Compute average vector for each batch by dividing the sum by count
        A_avg = A_sum / item_count.unsqueeze(-1)  # Shape: [batchsize, 64]

        # Move linear_mapping to the same device as A_avg
        linear_mapping = linear_mapping.to(A_avg.device)

        # Step 5: Apply the linear transformation to B to map it to 64 dimensions
        B_mapped = linear_mapping(B)  # Shape: [num2, 64]
        A=A_avg

        A_normalized = F.normalize(A, p=2, dim=1).to(torch.float16)  # Shape: [num, 64]
        B_normalized = F.normalize(B_mapped, p=2, dim=1).to(torch.float16)

        similarities = torch.matmul(A_normalized, B_normalized.T)  # Shape: [num, num2]
        similarities = similarities.to(torch.float32)

        top_k_indices = torch.topk(similarities, 1, dim=1, largest=True).indices  # Shape: [num, 50]


        nearest_vectors = torch.stack([B[indices] for indices in top_k_indices])  # Shape: [num, 50, 4096]
        # print(nearest_vectors.shape)
        nearest_vectors = nearest_vectors .view(-1,dim)
        return nearest_vectors

    def encode_and_expand(self, word_embedding, i):
        embeddings = self.word_embeddinglist[i]

        # 正则化嵌入
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)  # Shape: [n, embedding_dim]
        word_embedding_normalized = F.normalize(word_embedding, p=2, dim=1)  # Shape: [vocab_size, embedding_dim]

        # Step 4: 在词表嵌入中为每个词语找到最近的邻居
        similarities = torch.matmul(embeddings_normalized, word_embedding_normalized.T)  # Shape: [n, vocab_size]
        nearest_indices = torch.topk(similarities, 1, dim=1, largest=True).indices.squeeze().to(
            word_embedding.device)  # Shape: [n]
        nearest_indices = nearest_indices.reshape(-1)

        expanded_embeddings = torch.cat((embeddings, word_embedding[nearest_indices]),
                                        dim=0)  # Shape: [2n, embedding_dim]
        # expanded_embeddings=embeddings
        return expanded_embeddings

class MLPS_for_reprogram(nn.Module):
    def __init__(self, H):
        super(MLPS_for_reprogram, self).__init__()

        self.H = H
        self.classifier = nn.Sequential(
            nn.Linear(self.H, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1000),
            nn.ReLU(),
        )

    def forward(self, bert_tensor):
        # [batchsize,sequenceLen,large_item_embedding]->[batchsize,sequenceLen,small_item_embedding]
        logits = self.classifier(bert_tensor)

        # print("logits", logits.shape)
        # logits=torch.reshape(logits,(batch,m,self.H))
        return logits
