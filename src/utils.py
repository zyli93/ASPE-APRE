"""Utilities file

Authors: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
import argparse
import random
import re
import csv
import gzip
import numpy as np
import torch
from torch.nn.init import _calculate_fan_in_and_fan_out
from torch import nn
from collections import namedtuple
import math

import sys
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from lxml import etree
from bs4 import BeautifulSoup
from collections import Counter
from collections import defaultdict
from datetime import datetime

from tqdm import tqdm

import spacy
import nltk


try:
    import _pickle as pickle
except ImportError:
    import pickle

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

# System functions
def dump_pkl(path, obj):
    """helper to dump objects"""
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)


def load_pkl(path):
    """helper to load objects"""
    with open(path, "rb") as fin:
        return pickle.load(fin)


def make_dir(path):
    """helper for making dir"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_time():
    time = datetime.now().isoformat()[5:24]
    return time

def check_memory():
    print('GPU memory: %.1f' % (torch.cuda.memory_allocated() // 1024 ** 2))


def tensor_to_numpy(tensor, use_gpu):
    if use_gpu:
        tensor = tensor.cpu()
    return tensor.detach().numpy()


def move_batch_to_gpu(batch):
    not_cuda = set(['uid_list', 'iid_list', 'u_split', 'i_split'])
    for tensor_name, tensor in batch.items():
        if tensor_name not in not_cuda:
            batch[tensor_name] = tensor.cuda()
    return batch


# Training model helper functions

def print_parameters(model):
    """Prints model parameters"""
    total = 0
    total_wo_embed = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        total_wo_embed += np.prod(p.shape) if "embed" not in name else 0
        print("{:30s} {:14s} requires_grad={}".format(name, str(list(p.shape)),
                                                      p.requires_grad))
    print("\nTotal parameters: {}".format(total))
    print("Total parameters (w/o embed): {}\n".format(total_wo_embed))

def load_embeddings(path, vocab, dim=200):
    """
    Load word embeddings and update vocab.
    :param path: path to word embedding file
    :param vocab:
    :param dim: dimensionality of the pre-trained embeddings
    :return:
    """
    if not os.path.exists(path):
        raise RuntimeError("You need to download the word embeddings. "
                           "See `data/beer` for the download script.")
    vectors = []
    w2i = {}
    i2w = []

    # Random embedding vector for unknown words
    vectors.append(np.random.uniform(
        -0.05, 0.05, dim).astype(np.float32))
    w2i[UNK_TOKEN] = 0
    i2w.append(UNK_TOKEN)

    # Zero vector for padding
    vectors.append(np.zeros(dim).astype(np.float32))
    w2i[PAD_TOKEN] = 1
    i2w.append(PAD_TOKEN)

    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(u' ', 1)
            w2i[word] = len(vectors)
            i2w.append(word)
            v = np.array(vec.split(), dtype=np.float32)
            assert len(v) == dim, "dim mismatch"
            vectors.append(v)

    vocab.w2i = w2i
    vocab.i2w = i2w

    return np.stack(vectors)

def get_minibatch(data, batch_size=256, shuffle=False):
    """Return minibatches, optional shuffling"""

    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch

def pad(tokens, length, pad_value=1):
    """add padding 1s to a sequence to that it has the desired length"""
    return tokens + [pad_value] * (length - len(tokens))


def prepare_minibatch(mb, vocab, device=None, sort=True):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    # batch_size = len(mb)
    lengths = np.array([len(ex.tokens) for ex in mb])
    maxlen = lengths.max()
    reverse_map = None

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]
    y = [ex.scores for ex in mb]

    x = np.array(x)
    y = np.array(y, dtype=np.float32)

    if sort:  # required for LSTM
        sort_idx = np.argsort(lengths)[::-1]
        x = x[sort_idx]
        y = y[sort_idx]

        # create reverse map
        reverse_map = np.zeros(len(lengths), dtype=np.int32)
        for i, j in enumerate(sort_idx):
            reverse_map[j] = i

    x = torch.from_numpy(x).to(device)
    y = torch.from_numpy(y).to(device)

    words = [ex.tokens for ex in mb]
    # print(y)

    return x, y, reverse_map, words, maxlen


nltk.data.path.append("/workspace/nltk_data")
ps = PorterStemmer()

def clean_str2(s):
    """Clean up the string
    * New version, removing all punctuations
    Cleaning strings of content or title
    Original taken from 
    [https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py]
    Args:
        string - the string to clean
    Return:
        _ - the cleaned string
    
    Things that are done here:
        1. get rid of
            a. contents in brackets
            b. punctuations
            c. words containing numbers in them
            d. double quotes
        2. replace
            a. 's to s (user's -> users)
            b. 've to ve (I've -> Ive)
            c. 't, 're, 'd, 'll
            d. more than one (2+) spaces -> one space
    
    The right order to process the signs:
    1. 's/'ve/... to no "'"
    2. 
    """
    ss = s

    # if there are square brackets with characters in them, get rid of the whole thing
    # ss = re.sub('\[.*?\]', '', ss)

    # if there are strings concatenated by puncts get rid of them
    ss = re.sub( r'([,.!?])([a-zA-Z0-9])', r'\1 \2', ss)

    # gets rid of words containing numbers in them
    ss = re.sub('\w*\d\w*', '', ss)
    ss = re.sub('[\'\"]', '', ss)  # removing double quotes
    ss = re.sub('\n', '', ss)  # removing line breaks
    # ss = re.sub("re-", "re", ss) # specifically fix re- to re
    ss = re.sub(r"\'s", "s", ss)
    ss = re.sub(r"\'ve", "ve", ss)
    ss = re.sub(r"n\'t", "nt", ss)
    ss = re.sub(r"\'re", "re", ss)
    ss = re.sub(r"\'d", "d", ss)
    ss = re.sub(r"\'ll", "ll", ss)
    ss = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", ss)

    # if anything is any of those punctuation marks, get rid of it
    ss = re.sub('[%s]' % re.escape(string.punctuation), '', ss)
    ss = re.sub(r"\s{2,}", " ", ss)
    # ss = re.sub(r"  ", " ", ss)
    return ss.strip().lower()


def remove_stopwords(string, stopword_set):
    """Removing Stopwords
    Args:
        string - the input string to remove stopwords
        stopword_set - the set of stopwords
    Return:
        _ - the string that has all the stopwords removed
    """
    word_tokens = word_tokenize(string)
    filtered_string = [word for word in word_tokens
                       if word not in stopword_set]
    return " ".join(filtered_string)


def clean_html(x):
    return BeautifulSoup(x, "lxml").get_text()


def stemmed_string(s):
    # tokens = word_tokenize(s)
    # stemmed = ""
    # for word in tokens:
    #     stemmed += ps.stem(word)
    #     stemmed += " "
    # return stemmed.rstrip()
    tokens = word_tokenize(s)
    return " ".join([ps.stem(word) for word in tokens])


lemmatizer = WordNetLemmatizer()


def lemmatized_string(s):
    ls = s.split()
    ns = ''
    for token in ls:
        ns += lemmatizer.lemmatize(token)
        ns += ' '
    return ns[:-1]


def print_args(args):
    not_print = set(
        ['padded_length', 
        "random_seed", 
        "disable_cf", "aspemb_max_norm", "eval_as_cls"])

    print("")
    print("="* 70)
    for arg in vars(args):
        if arg not in not_print:
            print("\t" + arg + " = " + str(getattr(args, arg)))
    print("="* 70)
    print("")