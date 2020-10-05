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

from vocabulary import UNK_TOKEN, PAD_TOKEN

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

from tqdm import tqdm

import spacy
import nltk


try:
    import _pickle as pickle
except ImportError:
    import pickle

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
        
def initialize_model_(model):
    """
    Model initialization.

    :param model:
    :return:
    """
    print("Glorot init")
    for name, p in model.named_parameters():
        if name.startswith("embed") or "lagrange" in name:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
        elif "lstm" in name and len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier", name, p.shape))
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            print("{:10s} {:20s} {}".format("zeros", name, p.shape))
            torch.nn.init.constant_(p, 0.)
        else:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))

def evaluate_loss(model, data, batch_size=256, device=None, cfg=None):
    """
    Loss of a model on given data set (using minibatches)
    Also computes some statistics over z assignments.
    """
    model.eval()  # disable dropout
    total = defaultdict(float)
    total_examples = 0
    total_predictions = 0

    for mb in get_minibatch(data, batch_size=1, shuffle=False):
        x, targets, reverse_map, words, _ = prepare_minibatch(mb, model.vocab, device=device)
        print(words)
        print(targets)
        mask = (x != 1)

        batch_examples = targets.size(0)
        batch_predictions = np.prod(list(targets.size()))

        total_examples += batch_examples
        # print("Test examples:")
        # print(total_examples)
        total_predictions += batch_predictions

        with torch.no_grad():
            output, z_matrix = model(x)
            for ii in range(z_matrix.shape[1]):
                rationle = ""
                for jj in range(z_matrix.shape[2]):
                    if z_matrix[0][ii][jj] == 1:
                        rationle = rationle + words[0][jj] + " "
                print(rationle)
                print('------------')

            classification_loss, total_loss, z_loss, z_matrix_loss = model.get_loss(output, targets, z_matrix, mask=mask)
            output = output * 6
            output = torch.round(output)
            output = torch.clamp(output, min=1, max=5)
            correct = (output == targets).sum().item()
            total["classification_loss"] += classification_loss.item() * batch_examples
            total["total_loss"] += total_loss.item() * batch_examples
            total["z_loss"] += z_loss * batch_examples
            total["accuracy"] += correct 
            #  # print info to console
            # classification_loss_str = "%.4f" % classification_loss.item()
            # total_loss_str = "%.4f" % total_loss.item()
            # # opt_str = make_kv_string(loss_optional)
            # print("Test classification loss %s total loss %s" %
            #         (classification_loss_str, total_loss_str))
            # e.g. mse_loss, loss_z_x, sparsity_loss, coherence_loss
            # for k, v in loss_opt.items():
            #     total[k] += v * batch_examples

    result = {}
    for k, v in total.items():
        if not k.startswith("z_num"):
            result[k] = v / float(total_examples)

    if "z_num_1" in total:
        z_total = total["z_num_0"] + total["z_num_c"] + total["z_num_1"]
        selected = total["z_num_1"] / float(z_total)
        result["p1r"] = selected
        result["z_num_0"] = total["z_num_0"]
        result["z_num_c"] = total["z_num_c"]
        result["z_num_1"] = total["z_num_1"]

    return result
