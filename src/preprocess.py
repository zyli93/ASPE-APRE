"""Preprocessing the dataset.

Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
import sys
import argparse
import pandas as pd
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import ujson as json
except ImportError:
    import json

from utils import make_dir, dump_pkl

CONFIG_DIR = "./configs/"
INPUT_DIR = "./raw_datasets/"
OUTPUT_DIR = "./data/"


def load_yelp_categories():
    """Load Yelp categories for business filter"""
    fname = CONFIG_DIR + "yelp.categories"
    cats = []
    with open(fname, "r") as fin:
        for line in fin:
            line = line.strip().split(", ")
            cats += [x.strip() for x in line]
    return set(cats)


def process_string(string):
    """pipeline to process string"""
    pass


def parse_yelp(args):
    """draw review from `review.json` """

    assert hasattr(args, "min_cat_num"), "Please set `min_cat_num` for yelp."
    assert hasattr(args, "k_core"), "Please set `k_core` for yelp."

    in_dir = INPUT_DIR + "yelp/"
    out_dir = OUTPUT_DIR + "yelp/"

    print("[Yelp] processing yelp dataset ...")

    print("[Yelp] loading business ...")
    food_cats = load_yelp_categories()

    business_profiles = dict()
    with open(in_dir + "business.json", "r") as fin:
        for ind, ln in enumerate(fin):
            data = json.loads(ln)
            # entry must have below fields
            if not all([bool(data[x]) for x in
                        ['business_id', 'review_count', 'categories']]):
                continue
            if data['review_count'] < args.k_core:
                continue

            categories = [x.strip().lower() for x in
                          data['categories'].strip().split(", ")]
            filter_cats_num = sum([x in food_cats for x in categories])

            # throw away the business in two cases:
            #    1. all its cat(s) not present in food_cats
            #    2. >=1 cats not in, but >args.min_cat_num in food_cats
            if (not filter_cats_num) or \
                (len(categories) != filter_cats_num
                 and filter_cats_num < args.min_cat_num):
                continue

            bid = data['business_id']
            business_profiles[bid] = {
                'review_count': data['review_count'],
                'categories': categories}

    print("[Yelp] loading reviews ...")
    bid2idx, uid2idx = dict(), dict()
    uniq_bids, uniq_uids = [], []

    review_bids, review_uids = [], []
    review_set = dict()
    with open(in_dir + "review.json", "r") as fin:
        for ln in fin:
            data = json.loads(ln)

            # Make sure all four domains are not `None`.
            if not all([bool(data[x]) for x in
                        ['business_id', 'user_id', 'stars', 'text']]):
                continue

            bid, uid = data['business_id'], data['user_id']

            if bid not in business_profiles:
                continue

            if bid not in bid2idx:
                new_bid = "b_"+str(len(uniq_bids))
                uniq_bids.append(bid)
                assert uniq_bids[-1] == uniq_bids[int(new_bid[2:])]
                bid2idx[bid] = new_bid
            else:
                new_bid = bid2idx[bid]

            if uid not in uid2idx:
                new_uid = "u_"+str(len(uniq_uids))
                uniq_uids.append(uid)
                assert uniq_uids[-1] == uniq_uids[int(new_uid[2:])]
                uid2idx[uid] = new_uid
            else:
                new_uid = uid2idx[uid]

            review_bids.append(new_bid)
            review_uids.append(new_uid)

            # NOTE: new_uid and new_bid are `u_[user_idx]` and `b_[bus_idx]`.
            review_set[(new_uid, new_bid)] = {
                "user_id": new_uid,
                "item_id": new_bid,
                "rating": data['stars'],
                "review": data["text"]}

    assert len(review_bids) == len(review_uids)

    print("[Yelp] building k_core graph, k={} ...".format(args.k_core))
    G = nx.Graph()
    G.add_edges_from(zip(review_uids, review_bids))
    print("[Yelp]\t num of nodes [{}] and edges [{}] before k_core.".format(
        G.number_of_nodes(), G.number_of_edges()))

    G_kcore = nx.algorithms.core.k_core(G, k=args.k_core)

    # Check if all edges are "ub" or "bu"
    assert all([x[0]+y[0] in ["bu", "ub"] for x, y in G_kcore.edges()]),\
        "NOT all edges are u-b or b-u!"

    # Unify edges from "u-b" or "b-u" to "u-b" to query `review_set`
    G_kcore_edges = [(x, y) if x[0] == "u" else (y, x)
                     for x, y in G_kcore.edges()]

    kcore_dataset = [review_set[tp] for tp in G_kcore_edges]
    print("[Yelp]\t num of nodes [{}] and edges [{}] after k_core.".format(
        G_kcore.number_of_nodes(), G_kcore.number_of_edges()))

    # create a dataframe to save/view/...
    kcore_df = pd.DataFrame(kcore_dataset)
    print("[Yelp] \t number of unique users [{}] and businesses [{}]".format(
        kcore_df["user_id"].nunique(), kcore_df['item_id'].nunique()))
    print("[Yelp] \t unique ratings {}".format(kcore_df['rating'].unique()))

    print("[Yelp] dumping data and four ref pickles ...")
    make_dir(out_dir)
    kcore_df.to_csv(out_dir+"data.csv", index=False,
                    columns=['user_id', 'item_id', 'rating', 'review'])

    dump_pkl(out_dir+"bid2idx.pkl", bid2idx)
    dump_pkl(out_dir+"uniq_bids.pkl", uniq_bids)
    dump_pkl(out_dir+"uid2idx.pkl", uid2idx)
    dump_pkl(out_dir+"uniq_uids.pkl", uniq_uids)

    print("[Yelp] preprocessing done, files saved to {}".format(out_dir))

    train, test = train_test_split(kcore_df, test_size=test_split_ratio)
    train.to_csv('sTrainData.csv')
    test.to_csv('sTestData.csv')


def parse_amazon(args):
    """Parse Amazon subsets

        Note: Amazon datasets are downloaded 5-cored.
    """
    assert hasattr(
        args, "amazon_subset"), "Please set `amazon_subset` for Amazon dataset."
    print("[Amazon] parsing data from Amazon {}".format(args.amazon_subset))

    in_dir = INPUT_DIR + "amazon/"
    out_dir = OUTPUT_DIR + "amazon/{}/".format(args.amazon_subset)

    iid2idx, uid2idx = dict(), dict()
    uniq_iids, uniq_uids = [], []

    in_file = in_dir + args.amazon_subset + ".json"
    assert os.path.exists(in_file),\
        "Cannot find the dataset {}".format(args.amazon_subset)

    dataset_entries = []
    with open(in_file, "r") as fin:
        for jsonl in fin:
            data = json.loads(jsonl)

            # make sure it has all domains of interest
            req_domains = ["asin", "overall", "reviewerID", "reviewText"]
            if not all([x in data for x in req_domains]) \
                    or not all([bool(data[x]) for x in req_domains]):
                continue

            iid, uid = data["asin"], data["reviewerID"]

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
                "text": data['reviewText'].replace("\n", " ")}

            dataset_entries.append(entry)

    df = pd.DataFrame(dataset_entries)
    print("[Amazon] \t num of nodes [{}] and num of edges [{}]".format(
        len(uniq_uids)+len(uniq_iids), df.shape[0]))
    print("[Amazon] \t unique number of users [{}], items [{}]".format(
        df["user_id"].nunique(), df["item_id"].nunique()))

    print("[Amazon] saving data to disk ...")
    make_dir(out_dir)
    df.to_csv(out_dir+"data.csv", index=False,
              columns=["user_id", "item_id", "rating", "text"])
    dump_pkl(out_dir+"iid2idx.pkl", iid2idx)
    dump_pkl(out_dir+"uid2idx.pkl", uid2idx)
    dump_pkl(out_dir+"uniq_iids.pkl", uniq_iids)
    dump_pkl(out_dir+"uniq_uids.pkl", uniq_uids)
    print("[Amazon] done saving data to {}!".format(out_dir))

    train, test = train_test_split(df, test_size=args.test_split_ratio)
    train.to_csv(out_dir+'train_data.csv')
    test.to_csv(out_dir+'test_data.csv')


def parse_goodreads(args):
    print("[Goodreads] processing yelp dataset ...")

    json_data = []
    # replace "goodreads.json" with the path of your file
    with open('goodreads.json', 'r') as handle:
        for line in handle:
            json_data.append(json.loads(line))

    data_raw = pd.DataFrame(json_data)
    data = data_raw.drop(['review_id', 'date_added', 'date_updated',
                          'read_at', 'started_at', 'n_votes', 'n_comments'], axis=1)

    def normalizeNulls(a):
        a.replace('N/A', np.NaN, inplace=True)
        a.replace('Null', np.NaN, inplace=True)
        a.replace('NULL', np.NaN, inplace=True)
        a.replace('null', np.NaN, inplace=True)
        a.replace('', np.NaN, inplace=True)
        a.replace('None', np.NaN, inplace=True)
        a.replace('none', np.NaN, inplace=True)

    normalizeNulls(data)
    data.dropna(inplace=True)  # dropping n/a or null values

    # the 2 blocks of code below assign a unique id to each book and author
    data['id'] = data.groupby(['user_id']).ngroup()
    for i in range(len(data.index)):
        data.loc[i, 'id'] = 'u_' + str(data.loc[i, 'id'])
    data['user_id'] = data['id']
    data = data.drop('id', axis=1)

    data['id'] = data.groupby(['book_id']).ngroup()
    for i in range(len(data.index)):
        data.loc[i, 'id'] = 'b_' + str(data.loc[i, 'id'])
    data['book_id'] = data['id']
    data = data.drop('id', axis=1)

    review_set = dict()
    for index, row in data.iterrows():
        review_set[(row['user_id'], row['book_id'])] = {
            "user_id": row['user_id'],
            "book_id": row['book_id'],
            "rating": row['rating'],
            "review": row['review_text']
        }

    # 5-core has to be applied here
    G = nx.Graph()
    # uids for authors, bids for books
    G.add_edges_from(zip(data['user_id'], data['book_id']))
    print("[Goodreads]\t num of nodes [{}] and edges [{}] before k_core.".format(
        G.number_of_nodes(), G.number_of_edges()))

    G_kcore = nx.algorithms.core.k_core(G, k=5)
    print("[Goodreads]\t num of nodes [{}] and edges [{}] after k_core.".format(
        G_kcore.number_of_nodes(), G_kcore.number_of_edges()))

    # Unify edges from "u-b" or "b-u" to "u-b" to query `review_set`
    G_kcore_edges = [(x, y) if x[0] == "u" else (y, x)
                     for x, y in G_kcore.edges()]

    kcore_user = []
    kcore_book = []
    for tp in G_kcore_edges:
        kcore_user.append(tp[0])
        kcore_book.append(tp[1])

    dict = {'user_id': kcore_user, 'book_id': kcore_book}

    # create a dataframe to save/view/...
    kcore_df = pd.DataFrame(data=dict)
    # our main data is now in 'kcore_df'˜ dataframe and not the 'data' dataframe

    def clean_text(unclean_text):
        clean_text = wc.clean_html(unclean_text)
        clean_text = wc.clean_str2(clean_text)
        stop_words = text.ENGLISH_STOP_WORDS
        clean_text = wc.remove_stopwords(clean_text, stop_words)
        clean_text = wc.lemmatized_string(clean_text)
        return clean_text

    def cleanFunction(x): return clean_text(x)
    kcore_df['review'] = pd.DataFrame(data.review_text.apply(cleanFunction))

    kcore_df.to_csv('cleanData.csv')

    def stem_text(unstemmed_text):
        stemmed_text = wc.stemmed_string(unstemmed_text)
        return stemmed_text

    def stemFunction(x): return stem_text(x)
    kcore_df['review_text'] = pd.DataFrame(
        data.review_text.apply(stemFunction))

    # count authors and books here
    book_count = kcore_df['book_id'].nunique()
    user_count = kcore_df['user_id'].nunique()

    num_reviews = (kcore_df.count())[3]

    # saving stats of the dataset in stats.json
    stats_dict = {'Number of books': str(book_count),
                  'Number of users': str(user_count), 'Number of reviews': str(num_reviews)}
    with open('stats.json', 'w') as fp:
        json.dump(stats_dict, fp)

    kcore_df.to_csv('stemmedCleanData.csv')

    train, test = train_test_split(kcore_df, test_size=test_split_ratio)
    train.to_csv('sTrainData.csv')
    test.to_csv('sTestData.csv')

    uniq_bids = kcore_df['book_id'].unique()
    uniq_uids = kcore_df['user_id'].unique()
    dump_pkl(out_dir+"uniq_bids.pkl", uniq_bids)
    dump_pkl(out_dir+"uniq_uids.pkl", uniq_uids)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="yelp",
        help="Dataset: yelp, amazon, goodreads")

    parser.add_argument(
        "--k_core",
        type=int,
        default=5,
        help="The number of cores of the dataset.")

    parser.add_argument(
        "--min_cat_num",
        type=int,
        default=2,
        help="Minimum number of category labels in the filter set.")

    parser.add_argument(
        "--amazon_subset",
        type=str,
        help="Subset name of Amazon dataset")

    parser.add_argument(
        "--test_split_ratio",
        type=float,
        default=0.05,
        help="Ratio of test split to main dataset")

    args = parser.parse_args()

    if args.dataset == "yelp":
        parse_yelp(args)
    elif args.dataset == "goodreads":
        parse_goodreads(args)
    elif args.dataset == "amazon":
        parse_amazon(args)
    else:
        raise ValueError("Invalid --dataset value")
