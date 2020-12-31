"""
    Filter aspects

    File name: filter_aspects.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/28/2020
    Date Last Modified: TODO
    Python Version: 3.6+

    An example to run this file:
        python src/filter_aspects.py --data_path=./data/amazon/digital_music/ --count_threshold=30
"""

# TODO: remove the run this file cmd to readme later

import os
import sys

import argparse
import pandas as pd
from tqdm import tqdm

from nltk.stem import WordNetLemmatizer
from nltk.corpus import words


from utils import load_pkl
from annotate import COL_AS_PAIRS

COL_AS_PAIRS_FIL = "review_as_aspairs_filtered"


def process_aspect_pipeline(aspect_term, wnl, valid_word_set, vocab_pos):
    aspect_term = aspect_term.split(" ")
    ret = []
    for w in aspect_term:
        w = w.lower()
        w = wnl.lemmatize(w)
        ret.append(w)

    if len(ret) == 1 \
        and (ret[0] not in valid_word_set 
            or ret[0] not in vocab_pos 
            or nn_proportion(ret[0], vocab_pos) < 0.15):
        return None
    if any([len(x) < 3 for x in ret]):
        return None
    if any([len(x) < 5 and x not in valid_word_set for x in ret]):
        return None
    return " ".join(ret)


def nn_proportion(term, vocab_pos):
    nn_count = sum([vocab_pos[term].get(pos, 0) for pos in ['NN', 'NNP', 'NNS', 'NNPS']])
    total_count = sum(vocab_pos[term].values())
    return nn_count / total_count


def get_aspect_above(counter_, threshold):
    tmp_count = {k:v for k,v in counter_.items() if v >= threshold}
    return tmp_count


def load_useful_pickles(args):

    data_dir = args.data_path
    aspairs = load_pkl(data_dir + "/as_pairs.pkl")
    aspairs_counter = load_pkl(data_dir + "/as_pairs_counter.pkl")
    vocab_pos = load_pkl(data_dir + "/postag_of_vocabulary_full.pkl")

    return aspairs, aspairs_counter, vocab_pos


def filter_data(args, aspects_filters):
    def filter_aspairs(row):
        aspairs_cand = row[COL_AS_PAIRS]
        return [x for x in aspairs_cand if x[0] in aspects_filters]
        

    print("[filter aspects] loading pickle ...")
    df = pd.read_pickle(args.data_path + "train_data_dep.pkl")
    tqdm.pandas()
    df[COL_AS_PAIRS_FIL] = df.progress_apply(filter_aspairs, axis=1)

    # check problems on local notebook

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--count_threshold", type=int, required=True, 
        help="Threshold of the count.")
    args = parser.parse_args()

    if args.data_path[-1] == "\/":
        args.data_path = args.data_path[:-1]

    aspairs, aspair_counter, vocab_pos = load_useful_pickles(args)

    wnl = WordNetLemmatizer()
    valid_word_set = set(words.words())

    # get aspect counter
    aspect_counter = {}
    for aspair, count in aspair_counter.items():
        aspect = aspair[0].lower()
        aspect_counter[aspect] = aspect_counter.get(aspect, 0) + count

    # process aspect_counter:
    sing_aspect_counter = {}
    for k, v in aspect_counter.items():
        new_asp = process_aspect_pipeline(k, wnl, valid_word_set, vocab_pos)
        if new_asp:
            sing_aspect_counter[new_asp] = sing_aspect_counter.get(new_asp, 0) + v
    
    # get results
    result = {k:v for k,v in sing_aspect_counter.items() 
                  if v >= args.count_threshold}
    
    print("[filter aspect] using {} aspects".format(len(result)))

    data = filter_data(args, result)

    data = data[["user_id", "item_id", "rating", "text", "original_text",
        COL_AS_PAIRS_FIL]]

    print("[filter aspect] saving dataframe with filtered aspects at {}".format(
        "train_data_aspairs.csv"))
    
    data.to_csv(args.data_path + "/train_data_aspairs.csv")

