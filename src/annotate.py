"""
    Annotation for aspects in natural languages

    Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
import sys
import argparse
import pandas as pd
from collections import Counter
from itertools import permutations
from utils import clean_str2
from nltk.tokenize import sent_tokenize, word_tokenize

def load_train_file(args):
    """load train data file"""`
    in_file = args.path + "/train_data.csv"
    df = pd.read_csv(in_file)
    return df


def load_seed_words():
    """load seed works for auto aspect detection"""
    pass


def compute_vocab_and_pmi(df, args)):
    """compute vocabulary and PMI"""
    if not hasattr(args.pmi_window_size):
        raise AttributeError("--pmi_window_size has to be specified!")

    # set counter
    ws = args.pmi_window_size
    single_counter, pair_counter = Counter(), Counter()
    total_window_counter = 1

    text_col = df["text"]
    for line in text_col:
        line = line.lower()
        for sent in sent_tokenize(line):
            word_tokens = word_tokenize(sent)
            for i in range(0, len(word_tokens) - args.window_size):
                token_window = l[i: i+ws]
                single_counter.update(token_window)
                pair_counter.update(permutations(token_window, 2))
                total_window_counter += 1



def main(args):
    pass

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