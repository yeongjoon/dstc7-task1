from torch import nn
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss
from torch import optim
import logging
import numpy as np

# def create_emb_layer(weight_matrix, trainable=True):
# 	num_embeddings, embedding_dim = weight_matrix.size()
# 	embedding = nn.Embedding(num_embeddings, embedding_dim)
# 	embedding.weight = nn.Paramter(weight_matrix)
# 	embedding.weight.requires_grad = trainable
# 	return embedding

class InputEncoding(nn.Module):
	"""
	Input Encoding Layer
	논문에서는 DME 방식을 사용하였으나 우선 Glove 하나만 사용
	Feedforward layer는 그대로 사용
	hidden size 300 사용
	"""

	def __init__(self, weight_matrix, vocab_size=50000, output_embedding_size=300,
				 hidden_size=300):
		super(InputEncoding, self).__init__()
		self.vocab_size = vocab_size
		self.output_embedding_size=output_embedding_size
		self.embedding = nn.Embedding(vocab_size, output_embedding_size)
		if weight_matrix is not None:
			self.embedding.weight = nn.Parameter(weight_matrix)
		self.embedding.weight.requires_grad = True	#추후에 변수 넣어서 수정
		self.biLSTM = nn.LSTM(input_size=output_embedding_size, hidden_size=hidden_size, batch_first=True,
							  bidirectional=True)
		self.fc = nn.Linear(hidden_size, output_embedding_size)

	def forward(self, input_sequence, max_length=None):
		embedded = self.embedding(input_sequence)
		embedded = self.fc(embedded)			#Embedding layer 끝
		embedded = F.relu(embedded)

		if max_length is not None:
			embedded = nn.utils.rnn.pack_padded_sequence(embedded, max_length, batch_first=True, )
		output, hidden = self.biLSTM(embedded)
		if max_length is not None:
			output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
		return output

class LocalMatching(nn.Module):
	def __init__(self, weight_matrix, vocab_size, output_embedding_size, hidden_size):
		super(LocalMatching, self).__init__()
		self.alpha_softmax = nn.Softmax(dim=2)
		self.beta_softmax = nn.Softmax(dim=1)
		self.fc1 = nn.Linear(hidden_size*8, hidden_size)
		self.fc2 = nn.Linear(hidden_size*8, hidden_size)

	def forward(self, context_input, response_input):
		cs = context_input
		rs = response_input

		e = torch.bmm(cs, rs.transpose(1, 2))		#e.shape(batch, context_seq, response_seq)

		alpha = self.alpha_softmax(e)
		beta = self.beta_softmax(e)

		cd = torch.bmm(alpha, rs)
		rd = torch.bmm(beta.transpose(1, 2), cs)

		raw_cl = torch.cat((cs,cd,cs-cd,cs*cd), dim=2)
		raw_rl = torch.cat((rs,rd,rs-rd,rs*rd), dim=2)

		cl = F.relu(self.fc1(raw_cl))
		rl = F.relu(self.fc2(raw_rl))

		return cl, rl



class MatchingComposition(nn.Module):
	def __init__(self, weight_matrix, vocab_size, output_embedding_size, hidden_size):
		super(MatchingComposition, self).__init__()
		self.biLSTM1 = nn.LSTM(input_size=output_embedding_size, hidden_size=hidden_size, batch_first=True,
							   bidirectional=True)
		self.biLSTM2 = nn.LSTM(input_size=output_embedding_size, hidden_size=hidden_size, batch_first=True,
							   bidirectional=True)
		self.fc = nn.Linear(hidden_size*8, hidden_size)
		self.final_layer = nn.Linear(hidden_size, 2)


	"""
	local matching vector들을 input으로 받음
	"""
	def forward(self, context_vector, response_vector, context_max_length=None, response_max_length=None):
		#context 부분 bilstm
		if context_max_length is not None:
			context_vector = nn.utils.rnn.pack_padded_sequence(context_vector, context_max_length, batch_first=True, )
		context_output, context_hidden = self.biLSTM1(context_vector)
		if context_max_length is not None:
			context_output, _ = nn.utils.rnn.pad_packed_sequence(context_output, batch_first=True)

		#response 부분 bilstm
		if response_max_length is not None:
			response_vector = nn.utils.rnn.pack_padded_sequence(response_vector, response_max_length, batch_first=True, )
		response_output, response_hidden = self.biLSTM2(response_vector)
		if response_max_length is not None:
			response_output, _ = nn.utils.rnn.pad_packed_sequence(response_output, batch_first=True)

		c_max = torch.max(context_output, dim=1)[0]
		c_mean = torch.mean(context_output, dim=1)
		r_max = torch.max(response_output, dim=1)[0]
		r_mean = torch.mean(response_output, dim=1)

		raw_final = torch.cat((c_max, c_mean, r_max, r_mean), dim=-1)
		raw_final = F.tanh(self.fc(raw_final))
		final = self.final_layer(raw_final)

		return final

class ESIMModel(nn.Module):
	def __init__(self, weight_matrix, vocab_size, output_embedding_size, hidden_size):
		super(ESIMModel, self).__init__()
		self.context_encoding = InputEncoding(weight_matrix, vocab_size, output_embedding_size, hidden_size)
		self.response_encoding = InputEncoding(weight_matrix, vocab_size, output_embedding_size, hidden_size)
		self.local_matching = LocalMatching(weight_matrix, vocab_size, output_embedding_size, hidden_size)
		self.matching_composition = MatchingComposition(weight_matrix, vocab_size, output_embedding_size, hidden_size)

	def forward(self, context, response, context_max_length=None, response_max_length=None):
		cs = self.context_encoding(context, context_max_length)
		rs = self.response_encoding(response, response_max_length)
		context_local, response_local = self.local_matching(cs, rs)
		final = self.matching_composition(context_local, response_local)

		return final


if __name__=='__main__':
	context_input = [[1, 3, 2, 2], [1, 2, 2, 0]]
	response_input = [[[2, 2], [3, 1], [1, 0]], [[3, 1], [2, 2], [1,3]]]
	context_length = [4, 4]
	response_length = [[2, 2, 2], [2, 2, 2]]
	loss_func = CrossEntropyLoss()
	if torch.cuda.is_available():
		loss_func.cuda()
	model = ESIMModel(None, vocab_size=4, output_embedding_size=10, hidden_size=10)
	optimizer = optim.Adam(model.parameters())
	context_input = torch.tensor(context_input)
	response_input = torch.tensor(response_input)
	lengths = [4, 3, 2]
	outputs, _ = model(context_input, response_input)