import logging
import ijson
from tqdm import tqdm
import numpy as np
import spacy
from spacy.attrs import ORTH, LEMMA
from noesis.dataset.data_augmentation import tokenize_string
from noesis.dataset.data_augmentation import augmenting_data

import pickle

logger = logging.getLogger(__name__)


def space_tokenize(text):
    """
    Tokenizes a piece of text by splitting it up based on single spaces (" ").
    Args:
     text (str): input text as a single string
    Returns:
         list(str): list of tokens obtained by splitting the text on single spaces
    """
    return text.split(" ")


def read_json(input_file):
    json_objects_lst = list()
    json_objects = ijson.items(input_file, 'item')
    for obj in json_objects:
        json_objects_lst.append(obj)
    return json_objects_lst


def prepare_data(path, is_train, tokenize_func=space_tokenize, format='JSON'):
    """
    Reads a tab-separated data file where each line contains a source sentence and a target sentence. Pairs containing
    a sentence that exceeds the maximum length allowed for its language are not added.
    Args:
        path (str): path to the data file
        tokenize_func (func): function for splitting words in a sentence (default is single-space-delimited)
        format (str): data format for input file. Default is JSON.
    Returns:
        list((str, list(str)), str): list of ((context, list of candidates), target) string pairs
    """

    logger.info("Reading Lines from {}".format(path))
    # Read the file and split into lines
    pairs = []
    with open(path, 'r') as fin:
        if format == 'JSON':
            pairs = process(read_json(fin), tokenize_func, is_train)
        elif format == 'CSV':
            pairs = read(fin, ",", tokenize_func)
        elif format == 'TSV':
            pairs = read(fin, ",", tokenize_func)

    logger.info("Number of pairs: %s" % len(pairs))
    return sort(pairs)


def sort(pairs):
    records = []
    for (context, candidates), target in pairs:
        sorted_candidates = list()
        tmp = [(i, v) for i, v in enumerate(candidates)]
        tmp.sort(key=lambda s: len(s[1]), reverse=True)
        for i, (idx, cand) in enumerate(tmp):
            if idx == target:
                target = i
            sorted_candidates.append(cand)
        records.append(((context, sorted_candidates), target))

    records.sort(key=lambda s: len(s[0][0]), reverse=True)
    return records

def read(fin, delimiter, tokenize_func):
    pairs = []
    for line in tqdm(fin):
        try:
            src, dst = line.strip().split(delimiter)
            pair = map(tokenize_func, [src, dst])
            pairs.append(pair)
        except:
            logger.error("Error when reading line: {0}".format(line))
            raise
    return pairs


def process(records, tokenize_func, is_train=False):
    pairs = []
    nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner"])
    nlp.tokenizer.add_special_case(u'__eou__', [{ORTH: u'__eou__'}])
    nlp.tokenizer.add_special_case(u'__eot__', [{ORTH: u'__eot__'}])
    for record in tqdm(records[:25000]):
        context = ""
        speaker = None
        for msg in record['messages-so-far']:
            if speaker is None:
                context += msg['utterance'] + " __eou__ "
                speaker = msg['speaker']
            elif speaker != msg['speaker']:
                context += "__eot__ " + msg['utterance'] + " __eou__ "
                speaker = msg['speaker']
            else:
                context += msg['utterance'] + " __eou__ "

        context += "__eot__"
        #print(context)
        context = [str(tok) for tok in nlp(context.lower())]

        # Create the next utterance options and the target label
        candidates = []
        correct_answer = record['options-for-correct-answers'][0]
        target_id = correct_answer['candidate-id']
        tgt = None
        for i, candidate in enumerate(record['options-for-next']):
            if candidate['candidate-id'] == target_id:
                tgt = i
            candidates.append([str(tok) for tok in nlp(candidate['utterance'].lower())])
            #candidates.append(tokenize_func(candidate['utterance'])[:30])

        if tgt is None:
            logger.info(
                'Correct answer not found in options-for-next - example {}. Setting 0 as the correct index'.format(
                    record['example-id']))
            tgt = 0
        else:
            pairs.append(((context, candidates), tgt))

    file_token = 'train' if is_train else 'dev'
    filename = '/home/nlpgpu5/yeongjoon/dstc7-noesis/data/lowercase_pairs_' + file_token + '_final.pkl'
    print(filename)
    with open(filename, 'wb') as f:
        pickle.dump(pairs, f)
    import sys
    sys.exit(1)
    if is_train is True:
        pairs = augmenting_data(pairs)
    return pairs


def read_vocabulary(path, max_num_vocab=50000):
    """
    Helper function to read a vocabulary file.
    Args:
        path (str): filepath to raw vocabulary file
        max_num_vocab (int): maximum number of words to read from vocabulary file
    Returns:
        set: read words from vocabulary file
    """
    logger.info("Reading vocabulary from {}".format(path))
    # Read the file and create list of tokens in vocabulary
    vocab = set()
    with open(path) as fin:
        for line in fin:
            if len(vocab) >= max_num_vocab:
                break
            try:
                vocab.add(line.strip())
            except:
                logger.error("Error when reading line: {0}".format(line))
                raise

    logger.info("Size of Vocabulary: %s" % len(vocab))
    return vocab

def prepare_bosung_data(path):
    pairs = []
    nlp = spacy.load('en_core_web_sm')
    nlp.tokenizer.add_special_case(u'__eou__', [{ORTH: u'__eou__'}])
    nlp.tokenizer.add_special_case(u'__eot__', [{ORTH: u'__eot__'}])
    for line in tqdm(open(path, 'r').readlines()):
        if line.split('\t')[0] == 'index':
            continue
        context = tokenize_string(nlp, line.split('\t')[1])
        response = tokenize_string(nlp, line.split('\t')[2])
        target = 1 if line.split('\t')[3].strip() == 'entailment' else 0
        pairs.append(((space_tokenize(context)[-50:], space_tokenize(response)[:50]), target))

    return pairs
