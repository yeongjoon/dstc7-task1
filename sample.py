import os
import argparse
import logging

import torch
from torch.nn import CrossEntropyLoss, BCELoss
import pickle
import random

from networks.dual_encoder import Encoder, DualEncoder
from trainers.supervised_trainer import SupervisedTrainer
from util.checkpoint import Checkpoint
from dataset.dataset import Dataset
from evaluator.evaluator import Evaluator

from networks.esim import ESIMModel
import copy

from dataset.vocabulary import load_embedding_weight

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Sample usage:
#     # training
#     python sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
					help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
					help='Path to dev data')
parser.add_argument('--test_path', action='store', dest='test_path',
					help='Path to test data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
					help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
					help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
					default=False,
					help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
					default='info',
					help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
	logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
	checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
	checkpoint = Checkpoint.load(checkpoint_path)
	dual_encoder = checkpoint.model
	vocab = checkpoint.vocab
else:
	# Prepare dataset
	#train = Dataset.from_file(opt.train_path, is_train=True)
	#train = Dataset.from_file(opt.dev_path, is_train=False)
	#with open('../data/augmented_dev_data.pkl', 'wb') as f:
	#    pickle.dump(train, f)
	#dev = Dataset.from_file(opt.dev_path, vocab=train.vocab, is_train=False)
	#with open('../data/augmented_dev_data.pkl', 'rb') as f:
	#    train = pickle.load(f)
	#dev = copy.deepcopy(train)
	#print("Changed1")
	#with open('../data/bosung_data.pkl', 'wb') as f:
	#    pickle.dump(train, f)
	#    pickle.dump(dev, f)
	#with open('../data/bosung_data.pkl', 'rb') as f:
	#    train = pickle.load(f)
	#    dev = pickle.load(f)
	#train = dev
	#with open('../data/tmp_test_data.pkl', 'wb') as f:
	#    pickle.dump(train, f)
	#    pickle.dump(dev, f)
	#with open('../data/tmp_test_data.pkl', 'rb') as f:
	#    train = pickle.load(f)
	#    dev = pickle.load(f)

	#train.data.sort(key=lambda s:len(s[0][1]), reverse=True)
	#with open('../data/augmented_data.pkl', 'wb') as f:
	#    pickle.dump(train, f)
	#    pickle.dump(dev, f)
	#with open('../data/augmented_data.pkl', 'rb') as f:
	#	train = pickle.load(f)
	#	dev = pickle.load(f)
	# with open('../data/lowercase_augmented_data.pkl', 'wb') as f:
	# 	pickle.dump(train, f)
	# 	pickle.dump(dev, f)
	with open('../data/lowercase_augmented_data.pkl', 'rb') as f:
		train = pickle.load(f)
		dev = pickle.load(f)
	#dev = copy.deepcopy(train)
	#train = copy.deecopy(dev)

	#train.data.sort(key=lambda s: len(s[0][1]), reverse=True)

	vocab = train.vocab
	max_len = 300

	# for i in range(10):
	# 	context_response = train.data[i][0]
	# 	target = train.data[i][1]
	# 	context = context_response[0]
	# 	response = context_response[1]
	# 	for idx in context:
	# 		print(vocab.itos[idx],' ',end="")
	# 	print()
	# 	for idx in response:
	# 		print(vocab.itos[idx],' ', end="")
	# 	print()
	#train = copy.deepcopy(dev)
	random.seed(1)
	random.shuffle(train.data)

	embedding_weight = None
	# embedding_weight1 = load_embedding_weight('/home/nlpgpu5/data/embeddings/glove.6B.300d.txt', vocab)         #300
	# embedding_weight2 = load_embedding_weight('/home/nlpgpu5/data/embeddings/glove.840B.300d.txt', vocab)       #300
	# embedding_weight3 = load_embedding_weight('/home/nlpgpu5/data/embeddings/glove.twitter.27B.200d.txt', vocab, dimension=200)    #200
	# embedding_weight4 = load_embedding_weight('/home/nlpgpu5/data/embeddings/wiki-news-300d-1M.vec', vocab)     #300
	# embedding_weight5 = load_embedding_weight('/home/nlpgpu5/data/embeddings/crawl-300d-2M.vec', vocab)         #300
	# with open('../data/subtask1_lowercase_embeddings.pkl', 'wb') as f:
	# 	pickle.dump(embedding_weight1, f)
	# 	pickle.dump(embedding_weight2, f)
	# 	pickle.dump(embedding_weight3, f)
	# 	pickle.dump(embedding_weight4, f)
	# 	pickle.dump(embedding_weight5, f)


	with open('../data/subtask1_lowercase_embeddings.pkl', 'rb') as f:
		embedding_weight1 = pickle.load(f)
		embedding_weight2 = pickle.load(f)
		embedding_weight3 = pickle.load(f)
		embedding_weight4 = pickle.load(f)
		embedding_weight5 = pickle.load(f)
	#with open('../data/subtask1_train_embedding.pkl', 'wb') as f:
	#    pickle.dump(embedding_weight, f)
	#with open('../data/subtask1_train_embedding.pkl', 'rb') as f:
	#    embedding_weight = pickle.load(f)
	# Prepare loss
	#import sys
	#sys.exit(1)
	loss_func = CrossEntropyLoss()
	if torch.cuda.is_available():
		loss_func.cuda()

	optimizer = None
	if not opt.resume:
		# Initialize model
		hidden_size = 300
		#hidden_size = 128
		bidirectional = True
		# context_encoder = Encoder(vocab.get_vocab_size(), max_len, hidden_size,
		#                      bidirectional=bidirectional, variable_lengths=True)
		# response_encoder = Encoder(vocab.get_vocab_size(), max_len, hidden_size,
		#                      bidirectional=bidirectional, variable_lengths=True)
		if torch.cuda.is_available() and embedding_weight1 is not None:
			embedding_weight1 = torch.tensor(embedding_weight1).cuda()
			embedding_weight2 = torch.tensor(embedding_weight2).cuda()
			embedding_weight3 = torch.tensor(embedding_weight3).cuda()
			embedding_weight4 = torch.tensor(embedding_weight4).cuda()
			embedding_weight5 = torch.tensor(embedding_weight5).cuda()

		#esim_model = ESIMModel(embedding_weight, vocab_size=vocab.get_vocab_size(), output_embedding_size=300, hidden_size=300)
		#esim_model = ESIMModel(weight_matrix=None, vocab_size=vocab.get_vocab_size(), output_embedding_size=300, hidden_size=300)
		esim_model = ESIMModel(embedding_weight1, embedding_weight2, embedding_weight3, embedding_weight4, embedding_weight5, vocab_size=vocab.get_vocab_size(), output_embedding_size=300, hidden_size=300)
		if torch.cuda.is_available():
			esim_model.cuda()

		#for param in esim_model.parameters():
		#    param.data.uniform_(-0.08, 0.08)

		# dual_encoder = DualEncoder(context_encoder, response_encoder)
		# if torch.cuda.is_available():
		#     dual_encoder.cuda()
		#
		# for param in dual_encoder.parameters():
		#     param.data.uniform_(-0.08, 0.08)

	# train
	t = SupervisedTrainer(loss_func=loss_func, batch_size=128,
						  checkpoint_every=30,
						  print_every=100, expt_dir=opt.expt_dir)

	t.train(esim_model, train, batch_size=128, num_epochs=30, dev_data=dev, optimizer=optimizer, resume=opt.resume)

	evaluator = Evaluator(batch_size=200)
	l, precision, recall = evaluator.evaluate(esim_model, dev)
	print("Precision: {}, Recall: {}".format(precision, recall))



