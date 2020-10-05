"""
    Annotation for aspects in natural languages

    Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
import sys
import argparse
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
    for i, line in enumerate(tqdm(text_col)):
        try:
            line = line.lower()
        except:
            print(i)
            print(line)
            sys.exit()
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
    p_i = single_counter.copy()
    p_ij = pair_counter.copy()
    p_i = {key: value / total_window_counter for key, value in p_i.items()}
    p_ij = {key: value / total_window_counter for key, value in p_ij.items()}

    # TODO: first process p_i and p_ij, then discard some rare tokens
    # TODO: the threshold of rare tokens are still to be decided

    # compute PMI, PMI[i,j] = log(P(i,j) / (P(i)*P(j))
    pmi = {key: log(value/(p_i[key[0]] * p_i[key[1]]))
           for key, value in p_ij.items()}
    print("Done!")
    
    # TODO: to delete dump_pkl later after verification
    dump_pkl("pmi.pkl", pmi)  # TODO: how to verify this is correct?

    dump_pkl("pij.pkl", p_ij)
    dump_pkl("pi.pkl", p_i)
    return pmi


def main(args):
    seeds = load_seed_words()
    train_df = load_train_file(args)
    compute_pmi(args, train_df)


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

    args = parser.parse_args()
    main(args)
