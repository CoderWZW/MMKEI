import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util.structure import PointWiseFeedForward
import os
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, XLMRobertaTokenizer
from data.pretrain import Pretrain
from math import sqrt
import math


# #
torch.cuda.set_device(1)
current_device = torch.cuda.current_device()

class LinRec_Model(nn.Module):
    def __init__(self, data, emb_size, max_len, n_blocks, n_heads, drop_rate, feature, datasetFile):
        super(LinRec_Model, self).__init__()
        self.data = data
        self.emb_size = emb_size
        self.block_num = n_blocks
        self.head_num = n_heads
        self.drop_rate = drop_rate
        self.max_len = max_len
        self.feature = feature
        self.datasetFile = datasetFile
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
                tensor = tensor.to(1)
                self.bert_tensor = torch.cat([self.bert_tensor, tensor], 0)

                self.mlps = MLPS(768)
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
            new_attn_layer = MultiHeadAttention(self.head_num, self.emb_size, 0.2, 0.2, 1e-12)
            self.attention_layers.append(new_attn_layer)
            self.forward_layer_norms.append(torch.nn.LayerNorm(self.emb_size, eps=1e-8))

            new_fwd_layer = PointWiseFeedForward(self.emb_size, self.drop_rate)
            self.forward_layers.append(new_fwd_layer)
        # self.bert_config = BertConfig.from_pretrained('BGE')
        # self.bert_config2 = BertConfig.from_pretrained('bert')
        #
        # self.bert_config.output_attentions = True
        # self.bert_config.output_hidden_states = True
        # self.llm_model = BertModel.from_pretrained(
        #     '/root/autodl-tmp/BGE',
        #
        #     local_files_only=True,
        #     config=self.bert_config,
        # )
        # # print(os.path.exists('/root/autodl-tmp/BGE/tokenizer.json'))  # 应该返回 True
        # # print(os.path.exists('/root/autodl-tmp/BGE/tokenizer_config.json'))  # 应该返回 True
        # self.tokenizer = XLMRobertaTokenizer.from_pretrained(
        #     '/root/autodl-tmp/BGE')
        # self.tokenizer2 = BertTokenizer.from_pretrained(
        #     'bert')
        # self.llm_model2 = BertModel.from_pretrained(
        #     'bert',
        #
        #     local_files_only=True,
        #     config=self.bert_config2,
        # )
        #
        # self.word_embeddings1 = self.llm_model.get_input_embeddings().weight
        # self.word_embeddings2 = self.llm_model2.get_input_embeddings().weight
        # self.random_tensor = torch.randn_like(self.word_embeddings2).cuda()
        # # self.vocab_size = self.word_embeddings.shape[0]
        # self.num_tokens = 1000
        # # self.mapping= MLPS( self.num_tokens)
        # self.mapping= MLPS(30522)
        # self.reprogramming_layer = ReprogrammingLayer(64, 8,d_llm=768)
        # self.MoE = MoE(64, 8)

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self, seq, pos):
        seq = torch.tensor(seq)
        pos = torch.tensor(pos)
        if (self.feature == 'text'):

            # new_inputs = self.train_inputs[seq]
            # [30000, 100] -> [5, 50, 100]
            # new_masks=self.train_masks[seq]
            # listoutputs=[]
            # for i in range(len(new_inputs[0])):
            #     outputs =self.bert(new_inputs[:,i].cuda(),new_masks[:,i].cuda())
            #     listoutputs.append(outputs)
            #     print('i')
            # c=torch.stack(listoutputs,dim=1)
            # print(c.shape)
            seq_emb = self.mlps(self.bert_tensor[seq.cuda()])
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
            seq_emb = torch.transpose(seq_emb, 0, 1)
            # attention_input = seq_emb
            # normalized_emb = self.attention_layer_norms[i](seq_emb)
            # mha_outputs, _ = self.attention_layers[i](normalized_emb, seq_emb, seq_emb, attn_mask=attention_mask)
            seq_emb = self.attention_layers[i](seq_emb, attention_mask)
            # seq_emb = normalized_emb + mha_outputs
            seq_emb = torch.transpose(seq_emb, 0, 1)
            seq_emb = self.forward_layer_norms[i](seq_emb)
            seq_emb = self.forward_layers[i](seq_emb)
            seq_emb = seq_emb * ~timeline_mask.unsqueeze(-1)
        seq_emb = self.last_layer_norm(seq_emb)
        seq_emb_input = seq_emb
        # source_embeddings = self.mapping(self.random_tensor.permute(1, 0)).permute(1, 0)
        # seq_emb = self.MoE(seq_emb, self.word_embeddings1, self.word_embeddings2)
        # # seq_emb = self.reprogramming_layer(seq_emb,   source_embeddings,   source_embeddings, attn_mask=attention_mask)
        # seq_emb = seq_emb+seq_emb_input
        # seq_emb = self.last_layer_norm(seq_emb)
        return seq_emb


# encoder
class MLPS(nn.Module):
    def __init__(self, H):
        super(MLPS, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of label
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


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads=8, d_keys=None, d_llm=1024, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, 64)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding, attn_mask=None):
        B, L, E = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding, S, attn_mask)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding, S, attn_mask=None):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        # 计算注意力得分
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        # 计算注意力分布
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # 加权求和 value_embedding
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts=2):
        super(MoE, self).__init__()
        # self.experts = nn.ModuleList([ReprogrammingLayer(input_dim, output_dim) for _ in range(num_experts)])
        self.expert1 = ReprogrammingLayer(input_dim, output_dim)
        self.expert2 = ReprogrammingLayer(input_dim, output_dim, d_llm=768)
        self.gating_network = GatingNetwork(input_dim, num_experts)
        self.mapping1 = MLPS(250002)
        self.mapping2 = MLPS(30522)

    def forward(self, seq_emb, word_embeddings1, word_embeddings2):
        # 使用 gating network 计算每个专家的权重
        expert_weights = self.gating_network(seq_emb).permute(0, 2, 1).unsqueeze(-1)
        # 对所有专家的输出进行加权求和
        source_embeddings_1 = self.mapping1(word_embeddings1.permute(1, 0)).permute(1, 0)
        source_embeddings_2 = self.mapping2(word_embeddings2.permute(1, 0)).permute(1, 0)
        expert_outputs = torch.stack([self.expert1(seq_emb, source_embeddings_1, source_embeddings_1),
                                      self.expert2(seq_emb, source_embeddings_2, source_embeddings_2)], dim=1)
        # print(f"expert_outputs shape: {expert_outputs.shape}")  # e.g., (batch_size, num_experts, output_dim)
        # print(f"expert_weights shape: {expert_weights.shape}")  # e.g., (batch_size, num_experts)
        output = torch.sum(expert_outputs * expert_weights, dim=1)

        return output


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

