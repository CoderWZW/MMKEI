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
import copy  # 添加这一行
# #
torch.cuda.set_device(1)
current_device = torch.cuda.current_device()


class STRec_Model(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate, feature, datasetFile,strategy):
        super(STRec_Model, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.att_drop_rate = drop_rate
        self.max_len = max_len
        self.feature = feature
        self.layer_norm_eps=1e-12
        self.hidden_dropout_prob=0.5
        self.datasetFile = datasetFile
        self.strategy = strategy
        self.bert_config2 = BertConfig.from_pretrained('/home/chenwuhan/MMKEI/bert')
        self.llm_model2 = BertModel.from_pretrained(
            '/home/chenwuhan/MMKEI/bert',
            local_files_only=True,
            config=self.bert_config2,
        ).cuda()
        self.tokenizer2 = BertTokenizer.from_pretrained(
            '/home/chenwuhan/MMKEI/bert')
        self.tokenizer3 = T5TokenizerFast.from_pretrained(
            '/home/chenwuhan/P5-sportbase', legacy=False)
        self.llm_model3 = T5Model.from_pretrained(
            '/home/chenwuhan/P5-sportbase',

            local_files_only=True,
            # config=self.bert_config3,
        ).cuda()
        self.tokenizer4 = T5TokenizerFast.from_pretrained(
            '/home/chenwuhan/P5-beautybase', legacy=False)
        self.llm_model4 = T5Model.from_pretrained(
            '/home/chenwuhan/P5-beautybase',

            local_files_only=True,
            # config=self.bert_config3,
        ).cuda()
        self.tokenizertoy = T5TokenizerFast.from_pretrained(
            '/home/chenwuhan/P5-toybase', legacy=False)
        self.llm_modeltoy = T5Model.from_pretrained(
            '/home/chenwuhan/P5-toybase',
            local_files_only=True,
        ).cuda()
        self.llama_config = LlamaConfig.from_pretrained('/home/chenwuhan/LLaMA')
        self.llama_config.num_hidden_layers = 7
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.tokenizer5 = LlamaTokenizer.from_pretrained(
            '/home/chenwuhan/LLaMA')
        self.llm_model5 = LlamaModel.from_pretrained(
            '/home/chenwuhan/LLaMA',
            local_files_only=True,
            config=self.llama_config,
        ).cuda()
        self.word_embeddingstoy = self.llm_modeltoy.get_input_embeddings().weight
        self.llm_modelgpt = GPT2Model.from_pretrained('/home/chenwuhan/gpt2').cuda()
        self.tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('/home/chenwuhan/gpt2')

        self.word_embeddings2 = self.llm_model2.get_input_embeddings().weight
        self.word_embeddings3 = self.llm_model3.get_input_embeddings().weight
        self.word_embeddings3_1 = self.word_embeddings3.clone()
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


        self.num_tokens = 1000
        self.mapping = MLPS_for_reprogram(len(self.word_embeddings2))
        self.reprogramming_layer = ReprogrammingLayer(64, 8, d_llm=768, d_keys=32)

        tokenizer = [self.tokenizer5, self.tokenizer2, self.tokenizer3, self.tokenizer4]
        llm = [self.llm_model5, self.llm_model2, self.llm_model3, self.llm_model4]
        self.MoE = MoE(64, 4, llm,
                       tokenizer)
        self.linear_layer = nn.Linear(30522, self.num_tokens)
        self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        self.item_emb=nn.Parameter(initializer(torch.empty(self.data.item_num + 1, self.emb_size)))
        self.pos_emb= nn.Parameter(initializer(torch.empty(self.max_len + 1, self.emb_size)))
        self.trm_encoder = TransformerEncoder(
            n_layers=self.block_num,
            n_heads=self.head_num,
            hidden_size=self.emb_size,
            inner_size=512,
            attn_dropout_prob=self.att_drop_rate,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act='gelu',
            layer_norm_eps= self.layer_norm_eps)
        self.LayerNorm = nn.LayerNorm(self.emb_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.ffn = FFN(50, 50, inner_size=64, hidden_dropout_prob=0,
                       layer_norm_eps=None)

    def creat_index_source(self, time_stamps_seq):
        time_stamps_seq = torch.tensor(time_stamps_seq).cuda()
        non_zero_indices = torch.sum(time_stamps_seq != 0, dim=1) - 1
        processed_intervals_batch = time_stamps_seq.clone()
        processed_intervals_batch[torch.arange(time_stamps_seq.size(0)), non_zero_indices] = 0
        cumulative_intervals_batch = torch.cumsum(processed_intervals_batch.flip(dims=[1]), dim=1).flip(dims=[1])
        time_stamps_seq = cumulative_intervals_batch
        # 序列长度
        # l = time_stamps_seq.size(1)

        time_stamps_seq = self.ffn(time_stamps_seq)

        # 返回处理后的时间间隔特征，去掉最后一维
        return time_stamps_seq

    def forward(self, seq,pos,time_stamps_seq):
        seq = torch.tensor(seq)
        pos = torch.tensor(pos)
        b = seq.size(0)
        pos_emb = self.pos_emb[pos]
        seq_emb= self.item_emb[seq]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.LayerNorm(seq_emb)
        seq_emb = self.dropout(seq_emb)
        index_source = self.creat_index_source(time_stamps_seq)
        seq_emb_moe = self.MoE(seq_emb, seq.cuda(),self.strategy,[self.word_embeddings5, self.word_embeddings2, self.word_embeddings3,
                                self.word_embeddings4])
        pool = nn.AvgPool1d(kernel_size=5, stride=5)
        # pool = nn.AvgPool1d(kernel_size=10, stride=10)
        seq_emb_moe = pool(seq_emb_moe.permute(0, 2, 1)).permute(0, 2, 1)
        #seq_emb=seq_emb_moe
        trm_output = self.trm_encoder(seq_emb, pos,seq_emb_moe,index_source, output_all_encoded_layers=False)

        # output = trm_output[:, -1, :].squeeze(1)

        return trm_output


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            n_layers=2,
            n_heads=2,
            hidden_size=64,
            inner_size=256,
            hidden_dropout_prob=0.5,
            attn_dropout_prob=0.5,
            hidden_act='gelu',
            layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.n_layers = n_layers



    def forward(self, hidden_states, pos,seq_moe_emb,index_source, output_all_encoded_layers=False):
        b = hidden_states.size(0)
        l = hidden_states.size(1)
        i = 0
        rand = torch.randn(b, l).to(hidden_states.device)
        self.layer_norm = nn.LayerNorm(l, eps=1e-10, elementwise_affine=False)
        index_source = self.layer_norm(index_source) + rand
        # index_source[:, -1] = 100

        mask = (pos == 0).cuda()  # 找到pos为0的掩码
        index_source = index_source.masked_fill(mask, float('-inf'))  # 将pos为0的地方设为负无穷
        valid_count = (index_source > float('-inf')).sum(dim=1)
        # top_k = valid_count.min().item()

        if l == 50:
            alpha = [10, 10, 5, 5, 2, 2, 2, 2, 1]

        elif l == 8:
            alpha = [8, 8, 3, 3, 3, 3, 3, 3, 1]
        else:
            raise NotImplementedError("Sparsity not defined")

        for layer_idx, layer_module in enumerate(self.layer):
            if alpha[i+1]==alpha[i] and i!=0:
                indexj = None
            elif i!=0:
                indexj = alpha[i+1]
            else:

                _, indexj = index_source.topk(alpha[i + 1], dim=1, largest=True, sorted=True)

            # print(indexj)


            hidden_states = layer_module(hidden_states, indexj)
            if layer_idx == 0:
                hidden_states=hidden_states+seq_moe_emb
                # hidden_states = hidden_states
            i = i + 1
        return hidden_states

class TransformerLayer(nn.Module):
    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadSparseAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, downsample_index2):
        attention_output = self.multi_head_attention(hidden_states, downsample_index2)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output
class MultiHeadSparseAttention(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadSparseAttention, self).__init__()
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

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, X):
        X = X.view(X.shape[0], X.shape[1], self.num_attention_heads, self.attention_head_size)
        X = X.permute(0, 2, 1, 3).contiguous()
        return X

    def gather_indexes(self, output, gather_index):
        if gather_index is None:
            return output
        if isinstance(gather_index, int):
            return output[:, :gather_index]
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, output.size(-1)).long()
        gather_index.requires_grad = False
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor

    def forward(self, input_tensor, downsample_index):
        h = input_tensor.size(-1)
        downsample_tensor = self.gather_indexes(input_tensor, downsample_index)
        mixed_query_layer = self.query(downsample_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 1, 3, 2)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + downsample_tensor)
        return hidden_states
class FeedForward(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "relu": nn.ReLU,
            "gelu": self.gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states



class FFN(nn.Module):

    def __init__(
            self, inputsize=50, outputsize=50, inner_size=64, hidden_dropout_prob=0.5, layer_norm_eps=1e-10
    ):
        super(FFN, self).__init__()
        self.dense_1 = nn.Linear(inputsize, inner_size)

        self.dense_2 = nn.Linear(inner_size, outputsize)
        self.layer_norm_eps = layer_norm_eps
        if layer_norm_eps is not None:
            self.LayerNorm = nn.LayerNorm(outputsize, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_tensor):
        input_tensor = input_tensor.float()
        hidden_states = self.dense_1(input_tensor)
        if input_tensor.size(1) == 50:
            self.intermediate_act_fn = nn.Tanh()
        elif input_tensor.size(1) == 8:
            self.intermediate_act_fn = nn.ReLU()
        else:
            raise NotImplementedError("Intermediate_act_fn not defined")


        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.layer_norm_eps is not None:
            hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class MLPS(nn.Module):
    def __init__(self, H):
        super(MLPS, self).__init__()

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
            nn.Linear(128, 64),
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

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim,  llm,tokenizer,num_experts=4, dropout_rate=0.1, noisy_gating=False,
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
        mapping_sizes = [ 32000, 30522, 32100,32100]  # 根据实际需求设置这些尺寸
        # mapping_sizes = [256000, 30522]
        # dimension_mapping_size = [4096, 768, 768, 768]
        dimension_mapping_size = [4096, 768, 768, 768]
        self.dimension_mappings = nn.ModuleList([
            nn.Linear(size, 64) for size in dimension_mapping_size[:num_experts]
        ])
        self.mappings = nn.ModuleList([
            MLPS_for_reprogram(size) for size in mapping_sizes[:num_experts]
        ])
        txt_file_path = '/home/chenwuhan/MMKEI_update/dataset/Amazon-Pantry/introduction_gpt4o.txt'

        # Step 1: Load words from text file
        with open(txt_file_path, 'r') as file:
            word_list = [line.strip() for line in file if line.strip()]
        self.word_embeddinglist = []
        self.word_embeddinglist_copy = []
        all_words_as_sentence = " ".join(word_list)
        for i in range(4):
            # 对每个单词独立进行 tokenization 和嵌入
            # 一次性进行 tokenization 和嵌入生成
            inputs = tokenizer[i](all_words_as_sentence, return_tensors="pt", padding=False, truncation=True).to('cuda')
            input_ids = inputs["input_ids"].cuda()

            # 获取所有单词的嵌入
            embeddings = llm[i].get_input_embeddings().cuda()(input_ids).detach().clone().requires_grad_(
                True)  # Shape: [1, token_length, embedding_dim]

            # 去掉 batch 维度并返回 embeddings
            embeddings = embeddings.squeeze(0)

            embedding_copy = torch.rand_like(embeddings)
            self.word_embeddinglist.append(embeddings)
            self.word_embeddinglist_copy.append(embedding_copy)
    def forward(self, seq_emb, seq,strategy,word_embeddings_list):
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


            if strategy ==3:
                source_embeddings = self.encode_and_expand(word_embeddings_list[i], i)
            if strategy ==1:
                source_embeddings = self.mappings[i](word_embeddings_list[i].permute(1, 0)).permute(1, 0)
            if strategy ==2:

                source_embeddings = self.process_and_select(seq_emb, word_embeddings_list[i],
                                                            self.dimension_mappings[i],
                                                            seq)
            output = self.dropouts[i](self.experts[i](seq_emb, source_embeddings, source_embeddings))
            expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        output = torch.sum(expert_outputs * expert_weights, dim=1)
        #output = torch.sum(expert_outputs * expert_weights.unsqueeze(2), dim=1)

        return output

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
        # 扩展嵌入
        expanded_embeddings = torch.cat((embeddings, word_embedding[nearest_indices]),
                                        dim=0)  # Shape: [2n, embedding_dim]

        return expanded_embeddings
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

        top_k_indices = torch.topk(similarities,20, dim=1, largest=True).indices  # Shape: [num, 50]

        nearest_vectors = torch.stack([B[indices] for indices in top_k_indices])  # Shape: [num, 50, 4096]
       
        nearest_vectors = nearest_vectors.view(-1, dim)
        return nearest_vectors

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)