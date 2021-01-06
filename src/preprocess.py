"""Preprocessing the dataset.

Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
import sys
import argparse
import pandas as pd
import networkx as nx
import numpy as np
import ujson as json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from autocorrect import Speller
from nltk.tokenize import sent_tokenize, word_tokenize

from utils import make_dir, dump_pkl, clean_str2

CONFIG_DIR = "./configs/"
INPUT_DIR = "./raw_datasets/"
OUTPUT_DIR = "./data/"

correcter = Speller(lang="en")


def process_text(text, do_spell_check):
    """pipeline to process string
    
    Args:
        text - text snippet consisting of multiple sentences
        do_spell_check - [Bool] Whether to use spell check
    Returns:
        cleaned string with typos fixed by SpellChecker
    """
    sentences = sent_tokenize(text)
    ret_sentences = []
    for sent in sentences:
        if do_spell_check:
            sent = correcter(sent)
        sent = clean_str2(sent)
        ret_sentences.append(sent)
    return "\n".join(ret_sentences)


def output_aggregate_reviews(df, relative_path, for_user=True, output_single_file=False):
    """output aggregate review for user or for item
    [Moved to annotation process]
    Args:
        df - the input dataframe
        path - the path to the output file
        for_user - [Bool] create for user or for item 
        output_single_file - [Bool] whether to generate everything in a single file 
    """
    pass


def parse_amazon(args):
    """Parse Amazon subsets
       Note: Amazon datasets are downloaded 5-cored.
    """
    if not hasattr(args, "amazon_subset"):
        raise AttributeError("Please set `amazon_subset` for Amazon dataset.")
    print("[Amazon] parsing data from Amazon {}".format(args.amazon_subset))

    in_dir = INPUT_DIR + "amazon/"
    out_dir = OUTPUT_DIR + "amazon/{}/".format(args.amazon_subset)

    iid2idx, uid2idx = dict(), dict()
    uniq_iids, uniq_uids = [], []

    in_file = in_dir + args.amazon_subset + ".json"

    if not os.path.exists(in_file):
        raise FileNotFoundError(
            "Cannot find the dataset {}".format(args.amazon_subset))

    dataset_entries = []
    with open(in_file, "r") as fin:
        for jsonl in tqdm(fin):
            data = json.loads(jsonl)

            # make sure it has all domains of interest
            req_domains = ["asin", "overall", "reviewerID", "reviewText"]
            if not all([x in data for x in req_domains]) \
                    or not all([bool(data[x]) for x in req_domains]):
                continue

            iid, uid = data["asin"], data["reviewerID"]

            # filter out extremely short reviews
            processed_text = process_text(
                data['reviewText'], do_spell_check=args.use_spell_check)
            # processed_text = data['reviewText']
            if len(processed_text.strip().split(" ")) < args.min_review_len:
                continue

            # change string `asin` and `reviewerID` to `iid` and `uid`
            if iid not in iid2idx:
                new_iid = "i_"+str(len(uniq_iids))
                uniq_iids.append(iid)
                assert uniq_iids[-1] == uniq_iids[int(new_iid[2:])]
                iid2idx[iid] = new_iid
            else:
                new_iid = iid2idx[iid]

            if uid not in uid2idx:
                new_uid = "u_"+str(len(uniq_uids))
                uniq_uids.append(uid)
                assert uniq_uids[-1] == uniq_uids[int(new_uid[2:])]
                uid2idx[uid] = new_uid
            else:
                new_uid = uid2idx[uid]


            entry = {
                "item_id": new_iid, "user_id": new_uid,
                "rating": float(data['overall']),
                "text": processed_text,
                "original_text": data['reviewText']}

            dataset_entries.append(entry)

    # TODO: whether to re-apply kcore for amazon should be decided later!

    df = pd.DataFrame(dataset_entries)
    df = df.dropna(axis=0)  # dropping rows with NaN in it.
    print("[Amazon] \t num of nodes [{}] and num of edges [{}]".format(
        len(uniq_uids)+len(uniq_iids), df.shape[0]))
    print("[Amazon] \t unique number of users [{}], items [{}]".format(
        df["user_id"].nunique(), df["item_id"].nunique()))

    print("[Amazon] saving source data to disk ...")
    make_dir(out_dir)
    df.to_csv(out_dir+"data.csv", index=False,
              columns=["user_id", "item_id", "rating", "text"])
    dump_pkl(out_dir+"item_id_2_index.pkl", iid2idx)
    dump_pkl(out_dir+"user_id_2_index.pkl", uid2idx)
    dump_pkl(out_dir+"unique_item_ids.pkl", uniq_iids)
    dump_pkl(out_dir+"unique_user_ids.pkl", uniq_uids)

    print("[Amazon] splitting train and test data ...")
    train, test = train_test_split(df, test_size=args.test_split_ratio)
    train.to_csv(out_dir+'train_data.csv', index=False,
                 columns=["user_id", "item_id", "rating", "text", "original_text"])
    test.to_csv(out_dir+'test_data.csv', index=False,
                columns=["user_id", "item_id", "rating", "text", "original_text"])

    print("[Amazon] aggregating users' and items' reviews ...")

    print("[Amazon] done saving data to {}!".format(out_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--amazon_subset",
        type=str,
        required=True,
        help="Subset name of Amazon dataset")

    parser.add_argument(
        "--test_split_ratio",
        type=float,
        default=0.1,
        help="Ratio of test split to main dataset. Default=0.1.")

    parser.add_argument(
        "--k_core",
        type=int,
        default=5,
        help="The number of cores of the dataset. Default=5.")
    
    parser.add_argument(
        "--min_review_len",
        type=int,
        default=5,
        help="Minimum num of words of the reviews. Default=5.")

    parser.add_argument(
        "--use_spell_check",
        action="store_true",
        help="Whether to use spell check and correction. Turning this on will SLOW DOWN the process.")



    args = parser.parse_args()

    parse_amazon(args)
