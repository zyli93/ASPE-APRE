"""
    Extract 

    File name: extract.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/28/2020
    Date Last Modified: TODO
    Python Version: 3.6+

    An example to run this file:
        python src/extract.py --data_path=./data/amazon/digital_music/ --count_threshold=30
"""

# TODO: move the run this file cmd to readme later

import os
import sys

import argparse
import pandas as pd
from tqdm import tqdm

from nltk.stem import WordNetLemmatizer
from nltk.corpus import words


from utils import load_pkl, dump_pkl
from annotate import COL_AS_PAIRS

COL_AS_PAIRS_FIL = "review_aspairs_filtered"
COL_AS_PAIRS_IDX = "review_aspairs_index"


def process_aspect_pipeline(aspect_term, wnl, valid_word_set, vocab_pos):
    if aspect_term == "ItemTok":
        return "itemtok"
    aspect_term_split = aspect_term.split(" ")
    proc_asp_term_list = []
    for w in aspect_term_split:
        w = w.lower()
        w = wnl.lemmatize(w)
        proc_asp_term_list.append(w)

    # single term aspect
    if len(proc_asp_term_list) == 1 \
        and (proc_asp_term_list[0] not in valid_word_set
            or nn_proportion(aspect_term, vocab_pos, wnl) < 0.15):
        return None
    
    # e.g.: "an apple", "eg"
    if any([len(x) < 3 for x in proc_asp_term_list]):
        return None
    
    # e.g.: "xml lmlx"
    if any([len(x) < 5 and x not in valid_word_set for x in proc_asp_term_list]):
        return None
    return aspect_term


def nn_proportion(term, vocab_pos, wnl):
    nn_proportion_list = []
    for x in [term, term.lower(), wnl.lemmatize(term.lower())]:
        if x in vocab_pos:
            nn_count = sum([vocab_pos[x].get(pos, 0) for pos in ['NN', 'NNP', 'NNS', 'NNPS']])
            total_count = sum(vocab_pos[x].values())
            nn_proportion_list.append(nn_count / total_count)
        else:
            nn_proportion_list.append(0)
    return max(nn_proportion_list)


def load_useful_pickles(args):
    data_dir = args.data_path
    aspairs = load_pkl(data_dir + "/as_pairs.pkl")
    aspairs_counter = load_pkl(data_dir + "/as_pairs_counter.pkl")
    vocab_pos = load_pkl(data_dir + "/postag_of_vocabulary_full.pkl")

    return aspairs, aspairs_counter, vocab_pos


def filter_aspects(args, aspects_filters, vocab_pos):
    def filter_aspairs(row):
        aspairs_cand = row[COL_AS_PAIRS]
        ret = []
        for cand in aspairs_cand:
            asp_term, senti_term = cand
            if asp_term == "ItemTok":
                ret.append(cand)
            asp_sing = wnl.lemmatize(asp_term.lower())
            if asp_sing in aspects_filters:
                ret.append(cand)
        return ret
        

    print("[filter aspects] loading pickle ...")
    df = pd.read_pickle(args.data_path + "train_data_dep.pkl")
    tqdm.pandas()
    df[COL_AS_PAIRS_FIL] = df.progress_apply(filter_aspairs, axis=1)

    # check problems on local notebook
    # next problem:
    #   aspect with 2+ words

    return df


def build_aspect_category_dict(aspair_counter, valid_aspairs):
    """build aspect category list
    
    Args:
        aspair_counter - Aspect-Sentiment occurrence counter 
            {raw_aspect: occurrence}
    
    Variables:
        aspect_counter - {processed_aspect: occurrence}
        aspcat_to_idx - {aspect cateogory: index}
        aspect_to_aspcat - {processed_aspect: aspect category}
    
    Returns:

    """
    # get aspect counter
    aspcat_counter = {}
    aspcat_to_idx = {}
    aspect_to_aspcat = {}

    mult_token_aspect = []

    # single token aspects
    for aspair in valid_aspairs:
        count = aspair_counter[aspair]
        aspect_term, sentiment_term = aspair

        # first round avoid multi-token aspect
        if " " in aspect_term:
            mult_token_aspect.append(aspair)
            continue

        # single token aspect, processed aspect
        proc_aspect = wnl.lemmatize(aspect_term.lower())
        aspcat_counter[proc_aspect] = aspcat_counter.get(proc_aspect, 0) + count

        # add raw aspect term to aspect cat
        if aspect_term not in aspect_to_aspcat:
            aspect_to_aspcat[aspect_term] = proc_aspect
    
    # multi token aspects
    for aspair in mult_token_aspect:
        aspect_term, sentiment_term = aspair
        count = aspair_counter[aspair]

        resolved_aspect = None
        resolved_proc_aspect = None
        resolved_count = 10000000
        resolved_nn_prop = 0

        for sub_term in aspect_term.split(" "):
            # TODO: use procxxx_func
            # TODO: ItemTok
            proc_sub_term = wnl.lemmatize(sub_term.lower())
            if proc_sub_term in aspcat_counter and \
                nn_proportion(proc_sub_term, vocab_pos, wnl) > resolved_nn_prop:
                resolved_aspect = sub_term
                resolved_proc_aspect = proc_sub_term
                resolved_count = count
                resolved_nn_prop = nn_proportion(proc_sub_term, vocab_pos, wnl)
        
        if resolved_aspect:
            aspcat_counter[resolved_proc_aspect] += resolved_count
            aspect_to_aspcat[aspect_term] = resolved_proc_aspect

    return aspcat_counter, aspect_to_aspcat


def convert_aspect_to_idx(data, asp2aspcat, aspcat2idx):
    def replace_text_by_id(row):
        aspair_list = row[COL_AS_PAIRS_FIL]
        ret = []
        for aspair in aspair_list:
            aspect_term, sentiment_term = aspair
            try:
                id_ = aspcat2idx[asp2aspcat[aspect_term]]
                ret.append((id_, sentiment_term))
            except:
                print(aspect_term)
        return ret
    
    data[COL_AS_PAIRS_IDX] = data.apply(replace_text_by_id, axis=1)
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--count_threshold", type=int, required=True, 
        help="Threshold of the count.")
    parser.add_argument("--run_mapping", action="store_true", default=False, 
        help="If off, only get aspairs but do not work on df. For viewing use, cheaper.")
    args = parser.parse_args()

    if args.data_path[-1] == "\/":
        args.data_path = args.data_path[:-1]

    # raw aspairs, raw aspair_counter
    aspairs, aspair_counter, vocab_pos = load_useful_pickles(args)

    # tools for regularization
    wnl = WordNetLemmatizer()
    valid_word_set = set(words.words())

    # Step 0: filter valid setting
    #  only aspairs that can pass process_aspect_pipeline will be added to 
    #  `valid_aspairs`
    print("[filter aspects] step 0: filter out invalid aspairs")
    valid_aspairs = []
    for aspair, count in aspair_counter.items():
        aspect_term, sentiment_term = aspair
        proc_aspect_term = process_aspect_pipeline(
            aspect_term, wnl, valid_word_set, vocab_pos)
        if proc_aspect_term:
            valid_aspairs.append(aspair)

    # Step 1: get counting information
    # Method 1: Simple Aspect to Aspect ID
    aspcat_counter, asp2aspcat = build_aspect_category_dict(
        aspair_counter, valid_aspairs)
    
    # Method 2: TODO.

    # Step 2: remove aspect under threshold
    # aspects_filters = {aspect: aspcat} we wanted
    aspects_filters = {k:v for k,v in asp2aspcat.items()
        if aspcat_counter[asp2aspcat[k]] >= args.count_threshold}
    
    # create aspcat2idx
    aspcat2idx = {}
    idx2aspcat = {}
    print("[filter aspects] note that aspcat idx starts from 0")
    for aspcat in aspects_filters.values():
        if aspcat == "filler":
            print(aspcat)
        if aspcat not in aspcat2idx:
            idx2aspcat[len(aspcat2idx)] = aspcat
            aspcat2idx[aspcat] = len(aspcat2idx)
    
    print("[filter aspects] using {} raw aspects".format(len(aspects_filters)))
    print("[filter aspects] using {} aspects".format(len(aspcat2idx)))
    
    dump_pkl(args.data_path+"/aspect_filters_test.pkl", aspects_filters)
    
    print("[filter aspects] save aspcat_counter, aspcat2idx, and asp2aspcat")
    dump_pkl(args.data_path+"/aspcat_counter.pkl", aspcat_counter)
    dump_pkl(args.data_path+"/aspcat2idx.pkl", aspcat2idx)
    dump_pkl(args.data_path+"/idx2aspcat.pkl", idx2aspcat)
    dump_pkl(args.data_path+"/asp2aspcat.pkl", asp2aspcat)


    if args.run_mapping:
        # use raw aspect to filter
        print("[filter aspects] run_mapping is True, filtering aspects")
        data = filter_aspects(args, aspects_filters, vocab_pos)

        print("[filter aspects] coverting aspect to index")
        data = convert_aspect_to_idx(data, asp2aspcat, aspcat2idx)

        data = data[["user_id", "item_id", "rating", "text", "original_text",
            COL_AS_PAIRS_FIL, COL_AS_PAIRS_IDX]]

        print("[filter aspects] saving dataframe with filtered aspects at {}".format(
            "train_data_aspairs.csv"))
        
        data.to_csv(args.data_path + "/train_data_aspairs.csv", index=False)
        data.to_pickle(args.data_path + "/train_data_aspairs.pkl")



