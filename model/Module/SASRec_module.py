import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.structure import PointWiseFeedForward
import os
import math
from math import sqrt
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer,XLMRobertaTokenizer, XLMRobertaModel,T5Tokenizer, T5Model,T5TokenizerFast,AutoTokenizer,\
    AutoModelForCausalLM
from data.pretrain import Pretrain



# #
torch.cuda.set_device(0)
current_device = torch.cuda.current_device()

class SASRec_Model(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate, feature, datasetFile):
        super(SASRec_Model, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self.feature = feature
        self.datasetFile = datasetFile

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
        # self.llm_gemma=AutoModelForCausalLM.from_pretrained('/home/chenwuhan/hugging-gemma-2b',
        #                                                     device_map="auto",
        #                                                     )
        # self.gemma_tokenizer=AutoTokenizer.from_pretrained('/home/chenwuhan/hugging-gemma-2b')
        # 加载 GloVe 词向量，指定维度（如 100 维）
        # glove = GloVe(name="6B", dim=300)
        #
        # # 获取词嵌入矩阵的二维 tensor
        # num_embeddings = 32100
        #
        # # 随机选择 32100 个词向量
        # random_indices = torch.randperm(glove.vectors.size(0))[:num_embeddings]
        # random_glove_embedding = glove.vectors[random_indices]
        #
        # # 将选取的词向量移至 GPU
        # self.glove_embedding = random_glove_embedding.cuda()
        #
        # self.glove_embedding_2=self.glove_embedding.clone()
        # self.glove_embedding_3 = self.glove_embedding.clone()
        # self.glove_embedding_4 = self.glove_embedding.clone()
        self.llm_model2 = BertModel.from_pretrained(
            'bert',

            local_files_only=True,
            config=self.bert_config2,
        ).cuda()
        self.tokenizer2 = BertTokenizer.from_pretrained(
            'bert')
        self.tokenizer3 = T5TokenizerFast.from_pretrained(
            '/usr/gao/cwh/P5-sportbase', legacy=False)
        self.llm_model3 = T5Model.from_pretrained(
            '/usr/gao/cwh/P5-sportbase',

            local_files_only=True,
            # config=self.bert_config3,
        ).cuda()
        self.tokenizer4 = T5TokenizerFast.from_pretrained(
            '/usr/gao/cwh/P5-beautybase', legacy=False)
        self.llm_model4 = T5Model.from_pretrained(
            '/usr/gao/cwh/P5-beautybase',

            local_files_only=True,
            # config=self.bert_config3,
        ).cuda()
        self.llama_config = LlamaConfig.from_pretrained('/usr/gao/cwh/LLaMA')
        self.llama_config.num_hidden_layers = 7
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.tokenizer5 = LlamaTokenizer.from_pretrained(
            '/usr/gao/cwh/LLaMA')
        self.llm_model5 = LlamaModel.from_pretrained(
            '/usr/gao/cwh/LLaMA',
            local_files_only=True,
            config=self.llama_config,
        ).cuda()
        self.llm_modelgpt = GPT2Model.from_pretrained('/usr/gao/cwh/gpt2').cuda()
        self.tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('/usr/gao/cwh/gpt2')

        self.word_embeddings2 = self.llm_model2.get_input_embeddings().weight
        self.word_embeddings3 = self.llm_model3.get_input_embeddings().weight
        self.word_embeddings3_1= self.word_embeddings3.clone()
        self.word_embeddings4 = self.llm_model4.get_input_embeddings().weight
        self.word_embeddings4_1 = self.word_embeddings4.clone()
        self.word_embeddings5 = self.llm_model5.get_input_embeddings().weight
        are_equal = torch.equal(self.word_embeddings3, self.word_embeddings4)
        self.word_embeddings6 = self.word_embeddings2.clone()
        self.word_embeddings7 = self.word_embeddings2.clone()
        self.word_embeddings8 = self.word_embeddings2.clone()
        self.word_embedding_gpt2 = self.llm_modelgpt.get_input_embeddings().weight
        self.word_embedding_gpt2_1 = self.word_embedding_gpt2.clone()
        self.word_embedding_gpt2_2 = self.word_embedding_gpt2.clone()

     
        self.random_tensor1 = torch.rand_like(self.word_embeddings5).cuda()
        self.random_tensor2 = torch.rand_like(self.word_embeddings2).cuda()
        self.random_tensor3 = torch.rand_like(self.word_embeddings3).cuda()
        self.random_tensor4 = torch.rand_like(self.word_embeddings4).cuda()

        self.random_tensor5 = torch.rand_like(self.word_embeddings2).cuda()
        self.random_tensor6 = torch.rand_like(self.word_embeddings2).cuda()
        self.random_tensor7 = torch.rand_like(self.word_embeddings2).cuda()

        self.num_tokens = 1000
        self.mapping = MLPS_for_reprogram(len(self.word_embeddings2))
        self.reprogramming_layer = ReprogrammingLayer(64, 8, d_llm=768, d_keys=32)

        tokenizer = [self.tokenizer5,self.tokenizer2,self.tokenizer3]
        llm = [self.llm_model5,self.llm_model2,self.llm_model3]
        self.MoE = MoE(64, 4,llm ,
                       tokenizer )

        self.linear_layer = nn.Linear(30522, self.num_tokens)






        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_

        if (self.feature == 'text' or self.feature == 'id+text'):
            self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()

            if (len(self.datasetFile.split(",")) == 1):
                if not os.path.exists(self.datasetFile + "whole_tensor.pt"):
                    mask = 0
                    pre = Pretrain(self.data, self.datasetFile, mask)
                tensor = torch.load(self.datasetFile + "whole_tensor.pt")

                tensor = tensor.to(0)
                tensor= tensor.view(-1, 768)
                # print(tensor.shape)
                # print(self.bert_tensor.shape)
                self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)
                # self.bert_tensor = self.bert_tensor.clone().detach().requires_grad_(True)
                self.bert_tensor=torch.nn.Parameter(self.bert_tensor.detach())

                self.mlps = MLPS(self.emb_size)
            elif (len(self.datasetFile.split(",")) > 1):
                self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()
                for dataset in self.datasetFile.split(","):
                    if not os.path.exists(dataset + "whole_tensor.pt"):
                        mask = 0
                        pre = Pretrain(self.data, dataset, mask)
                    tensor = torch.load(dataset + "whole_tensor.pt")
                    self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)



                # self.bert_tensor = torch.load('./dataset/fuse_tensor'+ filename+'.pt')
                self.mlps = MLPS(self.emb_size)
        
        self.item_emb = nn.Parameter(initializer(torch.empty(self.data.item_num + 1, self.emb_size)))

        self.pos_emb = nn.Parameter(initializer(torch.empty(self.max_len + 1, self.emb_size)))
        self.attention_layer_norms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layer_norms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.emb_dropout = torch.nn.Dropout(self.drop_rate)
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-8)

        for n in range(self.block_num):
            self.attention_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))
            # new_attn_layer = torch.nn.MultiheadAttention(self.emb_size, self.head_num, self.drop_rate)
            new_attn_layer = MultiHeadAttention(self.head_num,self.emb_size,0.2,0.2,1e-12)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))

            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate)
            self.forward_layers.append(new_fwd_layer)

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False
    def forward(self, seq, pos):
        seq = torch.tensor(seq)
        pos = torch.tensor(pos)
        if (self.feature == 'text'):

            seq_emb = self.mlps(self.bert_tensor[seq.cuda()])
            #维度更改
            # seq_emb=self.bert_tensor[seq.cuda()]
        elif (self.feature == 'id'):
            seq_emb = self.item_emb[seq]
        elif (self.feature == 'id+text'):
            seq_emb = self.item_emb[seq] + self.mlps(self.bert_tensor[seq.cuda()])
        seq_emb = seq_emb * self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]

        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        timeline_mask = torch.BoolTensor(seq == 0).cuda()
        # print("timeline_mask",timeline_mask)
        seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        tl = seq_emb.shape[1]

        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).cuda())
        # print("attention_mask",attention_mask)
        for i in range(len(self.attention_layers)):
            if(i==0):
                seq_emb_moe = self.MoE(seq_emb, seq.cuda(),[self.random_tensor1,self.random_tensor2,self.random_tensor3
                                                            ])


            seq_emb = torch.transpose(seq_emb, 0, 1)
            # attention_input = seq_emb
            # normalized_emb = self.attention_layer_norms[i](seq_emb)
            # mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=attention_mask)
            # seq_emb = normalized_emb + mha_outputs
            seq_emb = self.attention_layers[i](seq_emb, attention_mask)
            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
            if (i == 0):
                seq_emb = seq_emb + seq_emb_moe
        seq_emb = self.last_layer_norm(seq_emb)
        return seq_emb


# encoder
class MLPS(nn.Module):
    def __init__(self, H):
        super(MLPS, self).__init__()

        self.H = H
        self.classifier = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.H),
            nn.ReLU(),
        )

    def forward(self, bert_tensor):
        # [batchsize,sequenceLen,large_item_embedding]->[batchsize,sequenceLen,small_item_embedding]
        logits = self.classifier(bert_tensor)
        # print("logits", logits.shape)
        # logits=torch.reshape(logits,(batch,m,self.H))
        return logits

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
        self.saved_target_embedding = None
        self.saved_source_embedding = None
    def forward(self, target_embedding, source_embedding, value_embedding,seq):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        targetembedding = self.query_projection(target_embedding).view(B, L, H, -1)
        self.saved_source_embedding = self.key_projection(source_embedding).view(S, H, -1)

        mask = seq != 0  # 生成一个形状为 (57, 50) 的布尔数组
        self.saved_target_embedding=targetembedding[mask]

        # 执行正常的 forward 操作
        target_embedding =  targetembedding
        source_embedding = self.saved_source_embedding
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
        mapping_sizes = [ 32000, 30522, 32100,32100] # 根据实际需求设置这些尺寸

        # mapping_sizes = [256000, 30522]
        dimension_mapping_size = [4096, 768, 768]

        self.dimension_mappings = nn.ModuleList([
            nn.Linear(size, 64) for size in dimension_mapping_size[:num_experts]
        ])
        self.mappings = nn.ModuleList([
            MLPS_for_reprogram(size) for size in mapping_sizes[:num_experts]
        ])
        txt_file_path = '/usr/gao/cwh/New_MoRec/ModalRec/ModalRec/dataset/Amazon-Pantry/introduction_gpt4o.txt'

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
            embeddings = llm[i].get_input_embeddings().cuda()(input_ids).detach().clone().requires_grad_(True)  # Shape: [1, token_length, embedding_dim]

            # 去掉 batch 维度并返回 embeddings
            embeddings = embeddings.squeeze(0)

            embedding_copy = torch.rand_like(embeddings)
            self.word_embeddinglist.append(embeddings)
            self.word_embeddinglist_copy.append(embedding_copy)


    def forward(self, seq_emb, seq,word_embeddings_list):
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

        # print(expert_weights.shape)



        expert_outputs = []
        for i in range(self.num_experts):
            # 策略2
            source_embeddings = self.process_and_select(seq_emb, word_embeddings_list[i], self.dimension_mappings[i],
                                                             seq)
            #策略1
            # source_embeddings = self.mappings[i](word_embeddings_list[i].permute(1, 0)).permute(1, 0)
            #策略3
            # source_embeddings = self.encode_and_expand(word_embeddings_list[i],i)

            output = self.dropouts[i](self.experts[i](seq_emb, source_embeddings, source_embeddings,seq))
            expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        output = torch.sum(expert_outputs * expert_weights, dim=1)
        #output = torch.sum(expert_outputs * expert_weights.unsqueeze(2), dim=1)

        return output
        # avg_logits = torch.mean(logits, dim=1)
        # top_k_logits, top_k_indices = torch.topk(avg_logits, 2, dim=-1)
        #
        # top_k_expert_weights = F.softmax(top_k_logits, dim=-1).unsqueeze(-1).unsqueeze(-1)
        #
        # expert_outputs = []
        # for i, idx in enumerate(top_k_indices[0]):  # 遍历 top_k_indices 只激活的专家
        #     # print(i,idx)
        #     # source_embeddings 的处理可以根据实际需要调整
        #     source_embeddings = self.mappings[idx.item()](word_embeddings_list[idx.item()].permute(1, 0)).permute(1, 0)
        #     # source_embeddings = self.encode_and_expand(word_embeddings_list[idx.item()], idx.item())
        #     output = self.dropouts[idx.item()](self.experts[idx.item()](seq_emb, source_embeddings, source_embeddings))
        #     expert_outputs.append(output * top_k_expert_weights[:, i, :, :])  # 直接乘以该专家的权重
        #
        # # 将激活的专家输出求和（只计算激活专家的加权输出）
        # output = torch.sum(torch.stack(expert_outputs, dim=1), dim=1)
        # return output
    def process_and_select(self, A: torch.Tensor, B: torch.Tensor, linear_mapping, seq):
        """
        A: Tensor of shape [num, 64]
        B: Tensor of shape [num2, 4096]

        Returns: 50 closest vectors from B in original dimensions [num, 50, 4096]
        """
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
        A = A_avg

        A_normalized = F.normalize(A, p=2, dim=1).to(torch.float16)  # Shape: [num, 64]
        B_normalized = F.normalize(B_mapped, p=2, dim=1).to(torch.float16)

        similarities = torch.matmul(A_normalized, B_normalized.T)  # Shape: [num, num2]
        similarities = similarities.to(torch.float32)

        top_k_indices = torch.topk(similarities,1, dim=1, largest=True).indices  # Shape: [num, 50]

        nearest_vectors = torch.stack([B[indices] for indices in top_k_indices])  # Shape: [num, 50, 4096]
        # print(nearest_vectors.shape)
        nearest_vectors = nearest_vectors.view(-1, dim)
        return nearest_vectors

    def encode_and_expand(self, word_embedding,i):


        embeddings=self.word_embeddinglist[i]
        # 正则化嵌入
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)  # Shape: [n, embedding_dim]
        word_embedding_normalized = F.normalize(word_embedding, p=2, dim=1)  # Shape: [vocab_size, embedding_dim]

        # Step 4: 在词表嵌入中为每个词语找到最近的邻居
        similarities = torch.matmul(embeddings_normalized, word_embedding_normalized.T)  # Shape: [n, vocab_size]
        nearest_indices = torch.topk(similarities, 1, dim=1, largest=True).indices.squeeze().to(word_embedding.device) # Shape: [n]
        nearest_indices=nearest_indices.reshape(-1)
        # 扩展嵌入
        expanded_embeddings = torch.cat((embeddings, word_embedding[nearest_indices]),
                                        dim=0)  # Shape: [2n, embedding_dim]
        # expanded_embeddings = embeddings
        return expanded_embeddings

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
            self,
            n_heads,
            hidden_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)  # row-wise
        self.softmax_col = nn.Softmax(dim=-2)  # column-wise
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.scale = np.sqrt(hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Our Elu Norm Attention
        elu = nn.ELU()
        # relu = nn.ReLU()
        elu_query = elu(query_layer)
        elu_key = elu(key_layer)
        query_norm_inverse = 1 / torch.norm(elu_query, dim=3, p=2)  # (L2 norm)
        key_norm_inverse = 1 / torch.norm(elu_key, dim=2, p=2)
        normalized_query_layer = torch.einsum('mnij,mni->mnij', elu_query, query_norm_inverse)
        normalized_key_layer = torch.einsum('mnij,mnj->mnij', elu_key, key_norm_inverse)
        context_layer = torch.matmul(normalized_query_layer, torch.matmul(normalized_key_layer,
                                                                          value_layer)) / self.sqrt_attention_head_size

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

