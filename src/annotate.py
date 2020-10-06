"""
    Annotation for aspects in natural languages

    Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
import sys
import argparse
import spacy
import pandas as pd
import ujson as json

from collections import Counter
from itertools import permutations
from utils import clean_str2
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from math import log

from utils import dump_pkl


def load_train_file(args):
    """load train data file"""
    print("[Annotate] loading training df ...", end=" ")
    in_file = args.path + "/train_data.csv"
    df = pd.read_csv(in_file)
    print("Done!")
    return df


def load_seed_words():
    """load seed works for auto aspect detection"""
    print("[Annotate] loading seed words ...", end=" ")
    seed_words_file = "./configs/seed_words.json"
    if not os.path.exists(seed_words_file):
        raise FileNotFoundError(
            "Seed words doesn't exist! Please refer to README for details.")
    with open(seed_words_file, "r") as fin:
        seed_words = json.load(fin)
    if not all([x in seed_words for x in ["POS", "NEG"]]):
        raise KeyError(
            "Config has to include all POS and NEG")
    print("Done!")
    return seed_words


def compute_pmi(args, df):
    """compute vocabulary and PMI
    Args:
        df - the input dataframe

    Returns:
        total_window_counter - the total number of windows in the corpus
        single_counter - the counter of single tokens
        pair_counter - the counter of pair of tokens
        pmi_vocab - vocabulary of pmi processing
    """
    # TODO: see if vocab should be returned.

    if not hasattr(args, "pmi_window_size"):
        raise AttributeError("--pmi_window_size has to be specified!")
    print("[Annotate] compute P(i) and P(i,j) ...", end=" ")

    # TODO: to discard the rare tokens as they are incorrect!

    # initialize counters
    ws = args.pmi_window_size
    single_counter, pair_counter = Counter(), Counter()
    total_window_counter = 1

    text_col = df["text"]
    for line in tqdm(text_col):
        line = line.lower()
        for sent in sent_tokenize(line):
            word_tokens = word_tokenize(sent)
            for i in range(0, len(word_tokens) - ws):
                token_window = word_tokens[i: i+ws]
                single_counter.update(token_window)
                pair_counter.update(permutations(token_window, 2))
                total_window_counter += 1
    print("Done!")

    # build p_i and p_ij and divide them by total_window_counter    
    print("[Annotate] compute PMI ...", end=" ")

    # delete rare tokens as they usually contain misspellings
    for word in list(single_counter.keys()):
        freq = single_counter[word]
        if freq < args.token_min_count:
            del single_counter[word]
    
    # delete pairs with rare tokens
    for pair in list(pair_counter.keys()):
        if pair[0] not in single_counter or pair[1] not in single_counter:
            del pair_counter[pair]

    p_i = single_counter.copy()
    p_ij = pair_counter.copy()
    p_i = {key: value / total_window_counter for key, value in p_i.items()}
    p_ij = {key: value / total_window_counter for key, value in p_ij.items()}

    # compute PMI, PMI[i,j] = log(P(i,j) / (P(i)*P(j))
    pmi = {key: log(value/(p_i[key[0]] * p_i[key[1]]))
           for key, value in p_ij.items()}
    dump_pkl(args.path + "/pmi_matrix.dict.pkl", pmi)

    # save pmi vocabularies
    pmi_vocabs = list(single_counter.keys())
    dump_pkl(args.path + "/pmi_vocabs.list.pkl", pmi_vocabs)

    print("Done!")
    print("[Annotate] PMI matrix and vocabulary saved in {}".format(args.path))
    
    # TODO: to delete dump_pkl later after verification
    dump_pkl("pmi.pkl", pmi)  # TODO: how to verify this is correct?

    dump_pkl("pij.pkl", p_ij)
    dump_pkl("pi.pkl", p_i)

    dump_pkl("single_counter.pkl", single_counter)
    dump_pkl("pair_counter.pkl", pair_counter)

    return pmi, pmi_vocabs


def get_vocab_pos(args, df, vocab):
    """Get the Part-of-Speech tagging for vocabulary
    * What are `pos` and `tag`?
        - "pos": Coarse-grained part-of-speech from the Universal POS tag set.
        - "tag": Fine-grained part-of-speech.
    
    Args:
        args - input arguments
        df - training dataframe
        vocab - the filtered vocabulary of PMI
    
    Return:
        vocab_pos - [Dict] word -> Dict{"pos": Counter(), "tag": Counter()}
    """
    print("[Annotate] processing vocabulary part-of-speech ...", end = " ")
    vocab_pos = {
        word: {"pos": Counter(), "tag": Counter()} for word in vocab}
    vocab = set(vocab)
    sp = spacy.load("en_core_web_sm")
    text_col = df["text"]
    # text -> sentences -> words
    for line in tqdm(text_col):
        line = line.lower()
        for sent in sent_tokenize(line):
            sp_sent = sp(sent)  # pos tagging with spacy
            for word in sp_sent:
                if word.text.lower() in vocab:
                    vocab_pos[word.text]["pos"].update(word.pos_)
                    vocab_pos[word.text]["tag"].update(word.tag_)
    print("Done!")

    print("[Annotate] majority voting for coarse- and fine-grained ...", end=" ")
    vocab_pos_coarse = {word: vocab_pos[word]["pos"].most_common(1)[0][0]
                        for word in vocab_pos.keys()}
    vocab_pos_fine = {word: vocab_pos[word]["tag"].most_common(1)[0][0]
                        for word in vocab_pos.keys()}
    print("Done!")
    return vocab_pos, vocab_pos_coarse, vocab_pos_fine


def main(args):
    # check args.path
    if not os.path.exists(args.path):
        raise ValueError("Invalid path {}".format(args.path))
    if args.path[-1] == "\\":
        args.path = args.path[:-1]

    seeds = load_seed_words()
    train_df = load_train_file(args)
    pmi_matrix, pmi_vocab = compute_pmi(args, train_df)
    get_vocab_pos(args, train_df, pmi_vocab)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset goodreads")

    parser.add_argument(
        "--pmi_window_size",
        type=int,
        default=3,
        help="The window size of PMI cooccurance relations. Default=3.")
    
    parser.add_argument(
        "--token_min_count",
        type=int,
        default=20,
        help="Minimum number of token occurences in corpus. Rare tokens are discarded")

    args = parser.parse_args()
    main(args)
