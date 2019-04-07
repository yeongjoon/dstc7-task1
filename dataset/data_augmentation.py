import numpy
import random
import spacy
from spacy.attrs import ORTH, LEMMA

def tokenize_string(nlp, s):
	doc = nlp(s)
	strs = ''
	for token in doc:
		strs += str(token)+' '
	strs=strs[:-1]
	return strs

def augmenting_data(pairs, ratio=4):

	random.seed(1)
	new_pairs = []
	for pair in pairs:
		whole_context = pair[0][0]
		whole_candidates = pair[0][1]

		target = pair[1]
		answer = whole_candidates[target]
		del(whole_candidates[target])

		new_candidates = []
		new_candidates.append(answer)
		for random_candidate in random.sample(whole_candidates, ratio):
			new_candidates.append(random_candidate)

		new_pairs.append(((whole_context, new_candidates), 0))          #여기서 0은 false의 의미가 아닌 index 0

		indexes = [index for index, value in enumerate(whole_context) if value == '__eou__']    #augmented가 아닌 정상적인 데이터는 따로 추가

		for i, idx in enumerate(indexes[:-1]):
			new_candidates = []
			if whole_context[idx+1] == '__eot__':
				new_context = whole_context[:idx+2]
				answer = whole_context[idx+2:indexes[i+1]]
			else:
				new_context = whole_context[:idx+1]
				answer = whole_context[idx+1:indexes[i+1]]

			new_candidates.append(answer)
			for random_candidate in random.sample(whole_candidates, ratio):
				new_candidates.append(random_candidate)

			new_pairs.append(((new_context, new_candidates), 0))

	return new_pairs

def slice_pairs(pairs):
	new_pairs = []
	for (context, candidates), target in pairs:
		new_context = context[-300:]
		new_candidates = []
		for candidate in candidates:
			new_candidates.append(candidate[:30])
		new_pairs.append(((new_context, new_candidates), target))
	return new_pairs

if __name__ == '__main__':
	context = ["fuck", "you", "__eou__", "__eot__", "love", "me", "__eou__", "hello", "boys", "__eou__", "__eot__"]
	candidates = [["wow", "!"], ["oh", "my"], ["god", "?"], ["sorry", "man"], ["What", "is", "up"], ["Please"]]
	#pairs = [((context, candidates), 0)]
	#augmenting_data(pairs)
	nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner"])
	nlp.tokenizer.add_special_case(u'__eou__', [{ORTH: u'__eou__'}])
	nlp.tokenizer.add_special_case(u'__eot__', [{ORTH: u'__eot__'}])
	s = "i don't forget about the case!!!"
	t = nlp(s)
	for token in t:
		print(str(token))

	for token in t.text:
		print(token)

	import pickle
	with open('/home/nlpgpu5/yeongjoon/dstc7-noesis/data/lowercase_pairs_train_final.pkl', 'rb') as f:
		pairs = pickle.load(f)

	new_pairs = augmenting_data(pairs)

	print(2)