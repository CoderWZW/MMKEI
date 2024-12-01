import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.structure import PointWiseFeedForward
import os
from data.pretrain import Pretrain
#
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer,XLMRobertaTokenizer, XLMRobertaModel,T5Tokenizer, T5Model,T5TokenizerFast
from math import sqrt
torch.cuda.set_device(1)
current_device = torch.cuda.current_device()
class FMLPRecModel(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate, feature, datasetFile):
        super(FMLPRecModel, self).__init__()
        # self.args = args
        # self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        # self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self.feature = feature
        self.datasetFile = datasetFile
        self.LayerNorm = LayerNorm(self.emb_size, eps=1e-12)
        self.dropout = nn.Dropout(self.drop_rate)
        self.item_encoder = Encoder()
        initializer = nn.init.xavier_uniform_
        self.bert_config2 = BertConfig.from_pretrained('bert')
        self.llm_model2 = BertModel.from_pretrained(
            'bert',

            local_files_only=True,
            config=self.bert_config2,
        ).cuda()
        self.tokenizer2 = BertTokenizer.from_pretrained(
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
        ).cuda()
        self.llama_config = LlamaConfig.from_pretrained('/usr/gao/cwh/LLaMA')
        self.llama_config.num_hidden_layers =7
        self.llama_config.output_attentions = True
        self.llama_config.output_hidden_states = True
        self.tokenizer5 =LlamaTokenizer.from_pretrained(
                    '/usr/gao/cwh/LLaMA')
        self.llm_model5 = LlamaModel.from_pretrained( '/usr/gao/cwh/LLaMA',
            local_files_only=True,
            config=self.llama_config,
        ).cuda()
        self.tokenizertoy = T5TokenizerFast.from_pretrained(
            '/usr/gao/cwh/P5-toybase', legacy=False)
        self.llm_modeltoy = T5Model.from_pretrained(
            '/usr/gao/cwh/P5-toybase',
            local_files_only=True,
        ).cuda()
        # self.llm_modelgpt = GPT2Model.from_pretrained('/home/chenwuhan/gpt2').cuda()
        # self.tokenizer_gpt2=GPT2Tokenizer.from_pretrained('/home/chenwuhan/gpt2')
        self.word_embeddings2 = self.llm_model2.get_input_embeddings().weight
        self.word_embeddings3 = self.llm_model3.get_input_embeddings().weight
        self.word_embeddings4 = self.llm_model4.get_input_embeddings().weight
        self.word_embeddings5 = self.llm_model5.get_input_embeddings().weight
        self.word_embeddingstoy=self.llm_modeltoy.get_input_embeddings().weight
        self.word_embeddings6=self.word_embeddings2.clone()
        self.word_embeddings7 = self.word_embeddings2.clone()
        self.word_embeddings8=self.word_embeddings2.clone()
        # self.word_embedding_gpt2=self.llm_modelgpt.get_input_embeddings().weight
        # self.word_embedding_gpt2_1 = self.word_embedding_gpt2.clone()
        # self.word_embedding_gpt2_2 = self.word_embedding_gpt2.clone()
    
        self.random_tensor1 = torch.rand_like(self.word_embeddings5 ).cuda()
        self.random_tensor2 = torch.rand_like(self.word_embeddings2 ).cuda()
        self.random_tensor3 =torch.rand_like(self.word_embeddings3 ).cuda()
        self.random_tensor4 = torch.rand_like(self.word_embeddings4 ).cuda()

        self.num_tokens = 1000
        self.mapping = MLPS_for_reprogram(len(self.word_embeddings2))
        #self.reprogramming_layer = ReprogrammingLayer(64, 8, d_llm=768, d_keys=32)
        # self.reprogramming_layer2 = ReprogrammingLayer2(64, 8, d_llm=768, d_keys=32)
        tokenizer = [self.tokenizer2, self.tokenizer3, self.tokenizer4,self.tokenizertoy]
        llm = [ self.llm_model2, self.llm_model3, self.llm_model4,self.llm_modeltoy]
        self.MoE = MoE(64, 4, llm,
                       tokenizer)

        self.linear_layer = nn.Linear(30522, self.num_tokens)

        if (self.feature == 'text' or self.feature == 'id+text'):
            self.bert_tensor = nn.Parameter(initializer(torch.empty(1, 768))).cuda()

            if (len(self.datasetFile.split(",")) == 1):
                if not os.path.exists(self.datasetFile + "whole_tensor.pt"):
                    mask = 0
                    pre = Pretrain(self.data, self.datasetFile, mask)
                tensor = torch.load(self.datasetFile + "whole_tensor.pt")
                tensor = tensor.to(0)
                self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)
                self.bert_tensor = torch.nn.Parameter(self.bert_tensor.detach())

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
        self.last_layer_norm = torch.nn.LayerNorm(self.emb_size, eps=1e-12)
        self.apply(self.init_weights)

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # same as SASRec
    def forward(self, seq, pos):
        # attention_mask = (input_ids > 0).long()
        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        # max_len = attention_mask.size(-1)
        # attn_shape = (1, max_len, max_len)
        # subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        # subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        # subsequent_mask = subsequent_mask.long()
        #
        # # if self.args.cuda_condition:
        # #     subsequent_mask = subsequent_mask.cuda()
        # extended_attention_mask = extended_attention_mask * subsequent_mask
        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        #
        # sequence_emb = self.add_position_embedding(input_ids)

        seq = torch.tensor(seq, device=current_device)
        pos = torch.tensor(pos, device=current_device)
        if (self.feature == 'text'):
            # seq_emb = self.mlps(self.bert_tensor[seq])

            seq_emb = self.bert_tensor[seq]
        elif (self.feature == 'id'):
            seq_emb = self.item_emb[seq]
        elif (self.feature == 'id+text'):
            seq_emb = self.item_emb[seq] + self.mlps(self.bert_tensor[seq])
        seq_emb = seq_emb * self.emb_size ** 0.5
        pos_emb = self.pos_emb[pos]
        seq_emb = seq_emb + pos_emb
        seq_emb = self.emb_dropout(seq_emb)
        timeline_mask = (seq == 0).to(dtype=torch.bool, device=current_device)
        # print("timeline_mask",timeline_mask)
        seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        tl = seq_emb.shape[1]

        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=current_device))

        # print("attention_mask",attention_mask

        seq_emb_moe = self.MoE(seq_emb, seq,[ self.word_embeddings2,self.word_embeddings3,self.word_embeddings4,self.word_embeddingstoy])
        seq_emb= self.item_encoder(seq_emb,attention_mask,seq_emb_moe,timeline_mask,output_all_encoded_layers=False)

        
        seq_emb=seq_emb[-1]       

        return seq_emb



    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        # if args.hidden_size % args.num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = 1
        self.hiddensize=768
        self.attention_head_size = int( self.hiddensize/ self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(self.hiddensize, self.all_head_size)
        self.key = nn.Linear(self.hiddensize, self.all_head_size)
        self.value = nn.Linear(self.hiddensize, self.all_head_size)
        self.attention_probs_dropout_prob=0.2
        self.attn_dropout = nn.Dropout()
        self.hidden_dropout_prob=0.2
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(self.hiddensize,self.hiddensize)
        self.LayerNorm = LayerNorm(self.hiddensize, eps=1e-12)
        self.out_dropout = nn.Dropout(self.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class FilterLayer(nn.Module):
    def __init__(self):
        super(FilterLayer, self).__init__()
        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.hiddensize=64
        self.max_seq_len=50
        self.complex_weight = nn.Parameter(torch.randn(1,  self.max_seq_len//2 + 1, self.hiddensize, 2, dtype=torch.float32) * 0.02)
        self.hidden_dropout_prob=0.2
        self.out_dropout = nn.Dropout(self.hidden_dropout_prob)

        self.LayerNorm = LayerNorm(self.hiddensize, eps=1e-12)


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        #sequence_emb_fft = torch.rfft(input_tensor, 2, onesided=False)  # [:, :, :, 0]
        #sequence_emb_fft = torch.fft(sequence_emb_fft.transpose(1, 2), 2)[:, :, :, 0].transpose(1, 2)
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm='ortho')
        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Intermediate(nn.Module):
    def __init__(self):
        super(Intermediate, self).__init__()
        self.hiddensize=64
        self.hidden_dropout_prob=0.2
        self.dense_1 = nn.Linear(self.hiddensize, self.hiddensize* 4)

        # self.intermediate_act_fn = gelu()
        self.dense_2 = nn.Linear(4 * self.hiddensize, self.hiddensize)
        self.LayerNorm = LayerNorm(self.hiddensize, eps=1e-12)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = gelu(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.attention=SelfAttention()

        self.filterlayer = FilterLayer()
        self.intermediate = Intermediate()

    def forward(self, hidden_states,attention_mask):
        # hidden_state1 = self.attention(hidden_states, attention_mask)
        hidden_state1 = self.filterlayer(hidden_states)
        intermediate_output = self.intermediate(hidden_state1)
        return intermediate_output

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        layer = Layer()
        self.num_blocks=1
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range( self.num_blocks)])

    def forward(self, hidden_states, attention_mask,emb_moe,timeline_mask, output_all_encoded_layers=False):
        all_encoder_layers = []
        for layer_idx,layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
            if layer_idx==0:
               hidden_states+=emb_moe
               hidden_states = hidden_states * ~timeline_mask.unsqueeze(-1)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

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

        out=self.reprogramming(target_embedding, source_embedding, value_embedding) 
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
    def __init__(self, input_dim, output_dim, llm,tokenizer,num_experts=4, dropout_rate=0.1, noisy_gating=False,
                 noise_epsilon=1e-2):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        self.noise_epsilon = noise_epsilon

        # 创建专家和对应的dropout层
        self.experts = nn.ModuleList([
            ReprogrammingLayer(input_dim, output_dim, d_llm=768) if i == 0
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
        # mapping_sizes = [ 32000,50257, 50257,50257]  # 根据实际需求设置这些尺寸
        mapping_sizes = [30522, 32100,32100,32100]
        # mapping_sizes=[30522,30522,30522,30522]
        # dimension_mapping_size=[4096,768,768,768]
        dimension_mapping_size = [4096, 768, 768, 768]
        self.dimension_mappings= nn.ModuleList([
           nn.Linear(size,64) for size in dimension_mapping_size[:num_experts]
        ])
        self.mappings = nn.ModuleList([
            MLPS_for_reprogram(size) for size in mapping_sizes[:num_experts]
        ])
        txt_file_path = '/usr/gao/cwh/New_MoRec/ModalRec/ModalRec/dataset/beer/gpt4o_introcuction.txt'

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

            #
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
            source_embeddings = self.mappings[i](word_embeddings_list[i].permute(1, 0)).permute(1, 0)
            # source_embeddings= self.process_and_select(seq_emb,word_embeddings_list[i],self.dimension_mappings[i], seq)
            # source_embeddings = self.encode_and_expand(word_embeddings_list[i], i)
            output = self.dropouts[i](self.experts[i](seq_emb, source_embeddings, source_embeddings))
            expert_outputs.append(output)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        output = torch.sum(expert_outputs * expert_weights, dim=1)

        return output



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

        top_k_indices = torch.topk(similarities, 10, dim=1, largest=True).indices  # Shape: [num, 50]


        nearest_vectors = torch.stack([B[indices] for indices in top_k_indices])  # Shape: [num, 50, 4096]
        # print(nearest_vectors.shape)
        nearest_vectors = nearest_vectors .view(-1,dim)
        return nearest_vectors
    def encode_and_expand(self, word_embedding,i):


        embeddings=self.word_embeddinglist[i]
        # 正则化嵌入
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)  # Shape: [n, embedding_dim]
        word_embedding_normalized = F.normalize(word_embedding, p=2, dim=1)  # Shape: [vocab_size, embedding_dim]

        # Step 4: 在词表嵌入中为每个词语找到最近的邻居
        similarities = torch.matmul(embeddings_normalized, word_embedding_normalized.T)  # Shape: [n, vocab_size]
        nearest_indices = torch.topk(similarities, 10, dim=1, largest=True).indices.squeeze().to(word_embedding.device) # Shape: [n]
        nearest_indices=nearest_indices.reshape(-1)
        # 扩展嵌入
        expanded_embeddings = torch.cat((embeddings, word_embedding[nearest_indices]),
                                        dim=0)  # Shape: [2n, embedding_dim]

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
            nn.Linear(1024,512),
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
        return logits
