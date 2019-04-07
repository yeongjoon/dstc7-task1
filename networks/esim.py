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

class ContextResponseEncoding(nn.Module):
	def __init__(self, weight_matrix1, weight_matrix2, weight_matrix3, weight_matrix4, weight_matrix5, vocab_size=50000,
	             output_embedding_size=300,
	             hidden_size=300):
		super(ContextResponseEncoding, self).__init__()
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.output_embedding_size = output_embedding_size
		self.embedding1 = nn.Embedding(vocab_size, output_embedding_size)
		# self.embedding.weight.requires_grad = True
		if weight_matrix1 is not None:
			self.embedding1.load_state_dict({'weight': weight_matrix1})
			self.embedding1.weight.requires_grad = False  # 추후에 변수 넣어서 수정

		self.embedding2 = nn.Embedding(vocab_size, output_embedding_size)
		if weight_matrix2 is not None:
			self.embedding2.load_state_dict({'weight': weight_matrix2})
			self.embedding2.weight.requires_grad = False  # 추후에 변수 넣어서 수정

		self.embedding3 = nn.Embedding(vocab_size, 200)
		if weight_matrix3 is not None:
			self.embedding3.load_state_dict({'weight': weight_matrix3})
			self.embedding3.weight.requires_grad = False  # 추후에 변수 넣어서 수정

		self.embedding4 = nn.Embedding(vocab_size, output_embedding_size)
		if weight_matrix4 is not None:
			self.embedding4.load_state_dict({'weight': weight_matrix4})
			self.embedding4.weight.requires_grad = False  # 추후에 변수 넣어서 수정

		self.embedding5 = nn.Embedding(vocab_size, output_embedding_size)
		if weight_matrix5 is not None:
			self.embedding5.load_state_dict({'weight': weight_matrix5})
			self.embedding5.weight.requires_grad = False  # 추후에 변수 넣어서 수정

		self.biLSTM1 = nn.LSTM(input_size=output_embedding_size, hidden_size=hidden_size, batch_first=True,
		                      bidirectional=True, dropout=0.5)

		self.biLSTM2 = nn.LSTM(input_size=output_embedding_size, hidden_size=hidden_size, batch_first=True,
		                      bidirectional=True, dropout=0.5)

		self.fc = nn.Linear(output_embedding_size * 4 + 200, hidden_size)
		torch.nn.init.xavier_uniform_(self.fc.weight)

		self.dropout = nn.Dropout(0.3)

	def forward(self, context_sequence, response_sequence, max_length=None):
		c_embedded1 = self.embedding1(context_sequence)
		c_embedded2 = self.embedding2(context_sequence)
		c_embedded3 = self.embedding3(context_sequence)
		c_embedded4 = self.embedding4(context_sequence)
		c_embedded5 = self.embedding5(context_sequence)

		context_embedded = F.relu(
			self.dropout(self.fc(torch.cat((c_embedded1, c_embedded2, c_embedded3, c_embedded4, c_embedded5), dim=-1))))

		r_embedded1 = self.embedding1(response_sequence)
		r_embedded2 = self.embedding2(response_sequence)
		r_embedded3 = self.embedding3(response_sequence)
		r_embedded4 = self.embedding4(response_sequence)
		r_embedded5 = self.embedding5(response_sequence)

		response_embedded = F.relu(
			self.dropout(self.fc(torch.cat((r_embedded1, r_embedded2, r_embedded3, r_embedded4, r_embedded5), dim=-1))))

		# context_h0 = torch.zeros(2, context_embedded.size(0), self.hidden_size).cuda()
		# context_c0 = torch.zeros(2, context_embedded.size(0), self.hidden_size).cuda()
		# response_h0 = torch.zeros(2, context_embedded.size(0), self.hidden_size).cuda()
		# response_c0 = torch.zeros(2, context_embedded.size(0), self.hidden_size).cuda()

		# context_output, context_hidden = self.biLSTM1(context_embedded, (context_h0, context_c0))
		# response_output, response_hidden = self.biLSTM2(response_embedded, (response_h0, response_c0))
		context_output, context_hidden = self.biLSTM1(context_embedded)
		response_output, response_hidden = self.biLSTM2(response_embedded)

		return context_output, response_output

class InputEncoding(nn.Module):
	"""
	Input Encoding Layer
	논문에서는 DME 방식을 사용하였으나 우선 Glove 하나만 사용
	Feedforward layer는 그대로 사용
	hidden size 300 사용
	"""

	def __init__(self, weight_matrix1, weight_matrix2, weight_matrix3, weight_matrix4, weight_matrix5, vocab_size=50000, output_embedding_size=300,
				 hidden_size=300):
		super(InputEncoding, self).__init__()
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.output_embedding_size=output_embedding_size
		self.embedding1 = nn.Embedding(vocab_size, output_embedding_size)
		#self.embedding.weight.requires_grad = True
		if weight_matrix1 is not None:
			self.embedding1.load_state_dict({'weight': weight_matrix1})
			self.embedding1.weight.requires_grad = False	#추후에 변수 넣어서 수정

		self.embedding2 = nn.Embedding(vocab_size, output_embedding_size)
		if weight_matrix2 is not None:
			self.embedding2.load_state_dict({'weight': weight_matrix2})
			self.embedding2.weight.requires_grad = False	#추후에 변수 넣어서 수정

		self.embedding3 = nn.Embedding(vocab_size, 200)
		if weight_matrix3 is not None:
			self.embedding3.load_state_dict({'weight': weight_matrix3})
			self.embedding3.weight.requires_grad = False	#추후에 변수 넣어서 수정

		self.embedding4 = nn.Embedding(vocab_size, output_embedding_size)
		if weight_matrix4 is not None:
			self.embedding4.load_state_dict({'weight': weight_matrix4})
			self.embedding4.weight.requires_grad = False	#추후에 변수 넣어서 수정

		self.embedding5 = nn.Embedding(vocab_size, output_embedding_size)
		if weight_matrix5 is not None:
			self.embedding5.load_state_dict({'weight': weight_matrix5})
			self.embedding5.weight.requires_grad = False	#추후에 변수 넣어서 수정


		self.biLSTM = nn.LSTM(input_size=output_embedding_size, hidden_size=hidden_size, batch_first=True,
							  bidirectional=True, dropout=0.5)

		self.fc = nn.Linear(output_embedding_size*4 + 200, hidden_size)
		torch.nn.init.xavier_uniform_(self.fc.weight)

		self.dropout = nn.Dropout(0.2)

	def forward(self, input_sequence, max_length=None):
		embedded1 = self.embedding1(input_sequence)
		#embedded = self.dropout(self.fc(embedded))			#Embedding layer 끝
		#embedded = F.relu(embedded)

		embedded2 = self.embedding2(input_sequence)
		embedded3 = self.embedding3(input_sequence)
		embedded4 = self.embedding4(input_sequence)
		embedded5 = self.embedding5(input_sequence)

		embedded = F.relu(self.dropout(self.fc(torch.cat((embedded1, embedded2, embedded3, embedded4, embedded5), dim=-1))))
		#embedded = F.relu(self.dropout(self.fc(embedded)))

		#h0 = torch.zeros(2, embedded.size(0), self.hidden_size).cuda()
		#c0 = torch.zeros(2, embedded.size(0), self.hidden_size).cuda()

		if max_length is not None:
			pass
			#embedded = nn.utils.rnn.pack_padded_sequence(embedded, max_length, batch_first=True)
		output, hidden = self.biLSTM(embedded)
		if max_length is not None:
			pass
			#output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
		return output

class LocalMatching(nn.Module):
	def __init__(self, weight_matrix, vocab_size, output_embedding_size, hidden_size):
		super(LocalMatching, self).__init__()
		self.alpha_softmax = nn.Softmax(dim=2)
		self.beta_softmax = nn.Softmax(dim=1)
		self.fc1 = nn.Linear(hidden_size*8, hidden_size)
		torch.nn.init.xavier_uniform_(self.fc1.weight)
		self.fc2 = nn.Linear(hidden_size*8, hidden_size)
		torch.nn.init.xavier_uniform_(self.fc2.weight)
		self.dropout1 = nn.Dropout(0.2)
		self.dropout2 = nn.Dropout(0.2)

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

		cl = F.relu(self.dropout1(self.fc1(raw_cl)))
		rl = F.relu(self.dropout2(self.fc2(raw_rl)))
		#cl = F.relu(self.fc1(raw_cl))
		#rl = F.relu(self.fc2(raw_rl))

		return cl, rl



class MatchingComposition(nn.Module):
	def __init__(self, weight_matrix, vocab_size, output_embedding_size, hidden_size):
		super(MatchingComposition, self).__init__()
		self.hidden_size = hidden_size
		self.biLSTM1 = nn.LSTM(input_size=output_embedding_size, hidden_size=hidden_size, batch_first=True,
							   bidirectional=True, dropout=0.5)
		self.biLSTM2 = nn.LSTM(input_size=output_embedding_size, hidden_size=hidden_size, batch_first=True,
							   bidirectional=True, dropout=0.5)
		self.fc = nn.Linear(hidden_size*8, hidden_size)
		torch.nn.init.xavier_uniform_(self.fc.weight)
		self.final_layer = nn.Linear(hidden_size, 2)
		torch.nn.init.xavier_uniform_(self.final_layer.weight)
		self.dropout1 = nn.Dropout(0.2)
		self.dropout2 = nn.Dropout(0.2)


	"""
	local matching vector들을 input으로 받음
	"""
	def forward(self, context_vector, response_vector, context_max_length=None, response_max_length=None):
		#context 부분 bilstm
		#cc0 = torch.zeros(2, context_vector.size(0), self.hidden_size).cuda()
		#ch0 = torch.zeros(2, context_vector.size(0), self.hidden_size).cuda()
		if context_max_length is not None:
			pass
			#context_vector = nn.utils.rnn.pack_padded_sequence(context_vector, context_max_length, batch_first=True, )
		#context_h0 = torch.zeros(2, context_vector.size(0), self.hidden_size).cuda()
		#context_c0 = torch.zeros(2, context_vector.size(0), self.hidden_size).cuda()

		#context_output, context_hidden = self.biLSTM1(context_vector, (context_h0, context_c0))
		context_output, context_hidden = self.biLSTM1(context_vector)
		if context_max_length is not None:
			pass
			#context_output, _ = nn.utils.rnn.pad_packed_sequence(context_output, batch_first=True)

		#response 부분 bilstm
		#rc0 = torch.zeros(2, response_vector.size(0), self.hidden_size).cuda()
		#rh0 = torch.zeros(2, response_vector.size(0), self.hidden_size).cuda()
		if response_max_length is not None:
			pass
			#response_vector = nn.utils.rnn.pack_padded_sequence(response_vector, response_max_length, batch_first=True, )
		# response_h0 = torch.zeros(2, response_vector.size(0), self.hidden_size).cuda()
		# response_c0 = torch.zeros(2, response_vector.size(0), self.hidden_size).cuda()
		# response_output, response_hidden = self.biLSTM2(response_vector, (response_h0, response_c0))
		response_output, response_hidden = self.biLSTM2(response_vector)
		if response_max_length is not None:
			pass
			#response_output, _ = nn.utils.rnn.pad_packed_sequence(response_output, batch_first=True)

		c_max = torch.max(context_output, dim=1)[0]
		c_mean = torch.mean(context_output, dim=1)
		r_max = torch.max(response_output, dim=1)[0]
		r_mean = torch.mean(response_output, dim=1)

		raw_final = torch.cat((c_max, c_mean, r_max, r_mean), dim=-1)
		raw_final = torch.tanh(self.fc(raw_final))
		raw_final = self.dropout1(raw_final)

		final = self.dropout2(self.final_layer(raw_final))
		#final = torch.sigmoid(self.final_layer(raw_final))

		return final

class ESIMModel(nn.Module):
	def __init__(self, weight_matrix1, weight_matrix2, weight_matrix3, weight_matrix4, weight_matrix5, vocab_size, output_embedding_size, hidden_size):
		super(ESIMModel, self).__init__()
		#self.context_encoding = InputEncoding(weight_matrix1, weight_matrix2, weight_matrix3, weight_matrix4, weight_matrix5, vocab_size, output_embedding_size, hidden_size)
		#self.response_encoding = InputEncoding(weight_matrix1, weight_matrix2, weight_matrix3, weight_matrix4, weight_matrix5, vocab_size, output_embedding_size, hidden_size)
		self.local_matching = LocalMatching(weight_matrix1, vocab_size, output_embedding_size, hidden_size)
		self.matching_composition = MatchingComposition(weight_matrix1, vocab_size, output_embedding_size, hidden_size)
		self.context_response_encoding = ContextResponseEncoding(weight_matrix1, weight_matrix2, weight_matrix3, weight_matrix4, weight_matrix5, vocab_size, output_embedding_size, hidden_size)

	def forward(self, context, response, context_max_length=None, response_max_length=None):
		#cs = self.context_encoding(context, context_max_length)
		#rs = self.response_encoding(response, response_max_length)
		cs, rs = self.context_response_encoding(context, response)
		context_local, response_local = self.local_matching(cs, rs)
		final = self.matching_composition(context_local, response_local)

		return final

if __name__=='__main__':
	context_input = [[1, 3, 2, 2], [1, 3, 2, 2], [1, 3, 2, 2], [1, 2, 2, 0], [1, 2, 2, 0], [1, 2, 2, 0]]
	response_input = [[2, 2], [3, 1], [1, 0], [3, 1], [2, 2], [1,3]]
	context_length = torch.tensor([4, 4])
	response_length = torch.tensor([[2, 2, 2], [2, 2, 2]])
	loss_func = CrossEntropyLoss()
	if torch.cuda.is_available():
		loss_func.cuda()
	model = ESIMModel(None, None, None, None, None, vocab_size=4, output_embedding_size=10, hidden_size=10)
	optimizer = optim.Adam(model.parameters())
	context_input = torch.tensor(context_input)
	response_input = torch.tensor(response_input)
	lengths = [4, 3, 2]
	outputs = model(context_input, response_input, context_length, response_length)
	print("Done!")