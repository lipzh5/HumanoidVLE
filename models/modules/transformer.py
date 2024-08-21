# -*- coding:utf-8 -*-
# @Author: Peizhen Li
# @Desc: None
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import copy


class AdditiveAttention(nn.Module):

	def __init__(self, inputs_dim, hidden_dim):

		# 与MLP不同的是，对输入和查询向量使用不同的映射
		# 把输入向量和查询向量映射到相同的维度之后，按维度相加，得到相似度向量, 再和value向量计算余弦相似度得到各个帧级别的分数
		# ====> 即score = value*(P*h + Q*d)  其中P、Q:(hidden_dim, inputs_dim) value: (hidden_dim, 1)
		super(AdditiveAttention, self).__init__()
		self.query_vector = nn.Parameter(torch.randn(inputs_dim))  # the same dim with inputs
		self.value = nn.Linear(hidden_dim, 1)

		self.P = nn.Linear(inputs_dim, hidden_dim)
		self.Q = nn.Linear(inputs_dim, hidden_dim)
		self.tanh = nn.Tanh()

	def forward(self, inputs, mask=None):
		'''
		inputs: (batch, seq_len, inputs_dim)
		mask: (batch, seq_len) 有效的为1, 无效为0

		outputs: (batch, inputs_dim), alpha:(batch, seq_len)
		return (batch, inputs_dim)
		'''
		now_batch_size, max_seq_len, _ = inputs.size()
		if max_seq_len == 1:
			return inputs.squeeze(), 1
		outputs = torch.add(self.P(inputs), self.Q(self.query_vector))

		outputs = self.tanh(outputs)
		scores = self.value(outputs).squeeze()
		if mask is not None:
			# mask = get_mask(inputs, seq_len)
			scores = scores.masked_fill(mask == 0., float('-inf'))  
		alpha = F.softmax(scores, dim=-1)  
		alpha = alpha.view(now_batch_size, 1, max_seq_len)  #utterance下各个单词或者各个frame图片的不同权重
		outputs = torch.bmm(alpha, inputs).squeeze(dim=1) #(batch_size, 1, hidden_size)

		return outputs, alpha


class LayerNorm(nn.Module):
	def __init__(self, hidden_size, eps=None):
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
	def __init__(self, hidden_size, num_attn_heads, attn_probs_dropout_prob):
		super(SelfAttention, self).__init__()
		assert hidden_size % num_attn_heads == 0, "The hidden size should be a multiple of num_attn_heads"
		self.num_attn_heads = num_attn_heads
		self.attn_head_size = int(hidden_size / num_attn_heads)
		self.all_head_size = num_attn_heads * self.attn_head_size

		self.query = nn.Linear(hidden_size, self.all_head_size)
		self.key = nn.Linear(hidden_size, self.all_head_size)
		self.value = nn.Linear(hidden_size, self.all_head_size)
		self.dropout = nn.Dropout(attn_probs_dropout_prob)

	def transpose_for_scores(self, x):
		new_x_shape = x.size()[:-1] + (self.num_attn_heads, self.attn_head_size)
		x = x.view(*new_x_shape)
		return x.permute(0, 2, 1, 3)

	def forward(self, hidden_states, attention_mask=None):
		mixed_query_layer = self.query(hidden_states)
		mixed_key_layer = self.key(hidden_states)
		mixed_value_layer = self.value(hidden_states)

		query_layer = self.transpose_for_scores(mixed_query_layer)
		key_layer = self.transpose_for_scores(mixed_key_layer)
		value_layer = self.transpose_for_scores(mixed_value_layer)

		# Take the dot product between "query" and "key" to get the raw attention scores.
		attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
		attention_scores = attention_scores / math.sqrt(self.attn_head_size)  # (batch_size, 12, embed_dim, embed_dim)

		'''
		attention_scores的shape为(batch_size, 12, embed_dim, embed_dim)
		attention_mask的shape为(batch_size, 1, 1, embed_dim) 值为全-1e-9
		'''
		attention_scores_add_mask = attention_scores if attention_mask is None else attention_scores + attention_mask  # (batch_size, 12, embed_dim, embed_dim)

		# Normalize the attention scores to probabilities.
		attention_probs = nn.Softmax(dim=-1)(attention_scores_add_mask)

		# This is actually dropping out entire tokens to attend to, which might
		# seem a bit unusual, but is taken from the original Transformer paper.
		attention_probs = self.dropout(attention_probs)

		context_layer = torch.matmul(attention_probs, value_layer)
		context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
		new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
		context_layer = context_layer.view(*new_context_layer_shape)
		return context_layer


def gelu(x):
	"""Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
	"""
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class TransformerIntermediate(nn.Module):
	def __init__(self, hidden_size, intermediate_size):
		super(TransformerIntermediate, self).__init__()
		self.dense = nn.Linear(hidden_size, intermediate_size)  # 升维度 从hidden_size变为4*hidden_size
		self.intermediate_act_fn = gelu

	def forward(self, hidden_states):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.intermediate_act_fn(hidden_states)
		return hidden_states


class ResidualNorm(nn.Module):
	def __init__(self, hidden_size, layer_norm_eps, hidden_dropout_prob):
		super(ResidualNorm, self).__init__()
		self.dense = nn.Linear(hidden_size, hidden_size)
		self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)
		self.dropout = nn.Dropout(hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 残差连接
		return hidden_states


class OutputResidualNorm(nn.Module):
	def __init__(self, intermediate_size, hidden_size, layer_norm_eps, hidden_dropout_prob):
		super(OutputResidualNorm, self).__init__()
		self.dense = nn.Linear(intermediate_size, hidden_size)  # 从4H再降回到H
		self.LayerNorm = LayerNorm(hidden_size, eps=layer_norm_eps)
		self.dropout = nn.Dropout(hidden_dropout_prob)

	def forward(self, hidden_states, input_tensor):
		hidden_states = self.dense(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)
		return hidden_states


class MultiHeadSelfAttention(nn.Module):
	def __init__(self, hidden_size, num_attn_heads, attn_probs_dp_prob, ln_eps, hidden_dp_prob):  # data.model.transformer
		super(MultiHeadSelfAttention, self).__init__()
		self.selfatt = SelfAttention(hidden_size, num_attn_heads, attn_probs_dp_prob)
		self.dense_norm = ResidualNorm(hidden_size, ln_eps, hidden_dp_prob)

	def forward(self, input_tensor, attention_mask=None):
		self_output = self.selfatt(input_tensor, attention_mask)  # word级别向量送进去进行multi-head attention操作
		attention_output = self.dense_norm(self_output, input_tensor)
		return attention_output


class TransformerEncoderLayer(nn.Module):
	def __init__(self, hidden_size=768, num_attn_heads=12, intermediate_size=3072, attn_probs_dropout_prob=0.1, layer_norm_eps=1e-12, hidden_dropout_prob=0.1):
		super(TransformerEncoderLayer, self).__init__()
		self.transformer_self_attention = MultiHeadSelfAttention(
			hidden_size, num_attn_heads, attn_probs_dropout_prob, layer_norm_eps, hidden_dropout_prob)
		self.intermediate = TransformerIntermediate(hidden_size, intermediate_size)  # 执行线性层(从H变为4H)和gelu操作
		self.output = OutputResidualNorm(intermediate_size, hidden_size, layer_norm_eps, hidden_dropout_prob)  # 线性层(4H再降到H)

	def forward(self, inputs, attention_mask=None):
		'''完成multi-head self-attention操作 + Residual_Norm操作(线性层、layerNorm、Dropout)'''
		attention_output = self.transformer_self_attention(inputs, attention_mask)
		'''线性层+gelu非线性激活'''
		intermediate_output = self.intermediate(attention_output)
		'''Output_Residual_Norm操作(线性层、layerNorm、Dropout)'''
		layer_output = self.output(intermediate_output, attention_output)
		return layer_output


class TransformerEncoder(nn.Module):  # MELD transformer encoder
	def __init__(self, config, num_layers, max_len, hidden_size):
		super(TransformerEncoder, self).__init__()
		self.position_embedding = nn.Embedding(max_len, hidden_size)
		layer = TransformerEncoderLayer(
			hidden_size=hidden_size,
			num_attn_heads=config.num_attn_heads,
			intermediate_size=config.intermediate_size,
			attn_probs_dropout_prob=config.attn_probs_dropout_prob,
			layer_norm_eps=config.layer_norm_eps,
			hidden_dropout_prob=config.hidden_dropout_prob
		)
		self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])  # lay_num: 3
	
	def forward(self, feature_input, attention_mask=None):
		# feature_input: (batch_size, utt_max_lens, self.hidden_size)
		batch_size, seq_length = feature_input.shape[:2]
		position_ids = torch.arange(seq_length, dtype=torch.long, device=feature_input.device)

		position_embeddings = self.position_embedding(position_ids).unsqueeze(0).repeat(batch_size, 1, 1)
		inputs = feature_input + position_embeddings
		for layer_module in self.layers:  # 共3层
			inputs = layer_module(inputs, attention_mask)
		return inputs


if __name__ == "__main__":
	# hidden_dim = 768
	Attn = AdditiveAttention(768, 768)
	inputs = torch.randn((2, 2, 768))
	feature = Attn(inputs)
	print(type(feature), len(feature), type(feature[0]), type(feature[1]))
	print(feature[0].shape, feature[1].shape, feature[1].detach())
