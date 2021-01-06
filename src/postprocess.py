'''
    Data postprocessing, aiming to generate data.

    TODO:
        maybe we should do everything in the postprocess 
        with the BERT tokenizer.

    File name: postprocess.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/07/2020
    Date Last Modified: TODO
    Python Version: 3.6

    Today I learned:
        1. csv is only suitable for non-object saving
        2. 
'''

import sys
import os
import argparse

from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm

from nltk.tokenize import sent_tokenize
from scipy.sparse import csr_matrix
from transformers import BertTokenizerFast
from pandarallel import pandarallel

import torch

from utils import dump_pkl

from extract import COL_AS_PAIRS_IDX

COL_REV_TEXT = "original_text"
COL_ASPAIRS = COL_AS_PAIRS_IDX

# Arguments parser
parser = argparse.ArgumentParser()

# E.g.: ./data/amazon/home_kitchen
parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Path to the dataset.")

parser.add_argument(
    "--num_aspects",
    type=int,
    required=True,
    help="Number of aspect categories in total")

parser.add_argument(
    "--max_pad_length",
    type=int,
    default=100,
    help="Max length of padding. Default=100.")

parser.add_argument(
    '--num_workers',
    type=int,
    default=8,
    help="Number of multithread workers")

parser.add_argument(
    "--build_nbr_graph",
    action="store_true",
    default=False,
    help="Whether to build neighborhood graph.")

args = parser.parse_args()


class EntityReviewAggregation:
    def __init__(self, reviews, aspairs, pad_len, num_asp):
        """Save aggregated reviews for users and items

        to keep as attributes:
            self.tokenized_revs - PyTorch style tokenization
            self.revs_asp_mention_locs - (list[csr_matrix]) Locations of aspect 
                mentions. sparse-mat dim: [num_asp, pad_len]
            self.reviews - (list[str]) list of text review
            self.reviews_count - (int) number of reviews
        """
        reviews = reviews.tolist()
        aspairs = aspairs.tolist() 
        # test sizes of reviews and aspairs
        if len(reviews) != len(aspairs):
            raise ValueError("Reviews and Aspairs sizes don't match!")
        
        if not all([isinstance(text, str) for text in reviews]):
            raise TypeError("There exists reviews that are NOT str")
            
        self.pad_len = pad_len
        self.num_asp = num_asp
        self.reviews = reviews

        # create tokenizer
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        # first tokenize reviews
        # only returns input_ids and attention_mask
        # no token_type_ids needed as we are handling single sentence/doc.
        self.tokenized_revs = tokenizer(
            reviews, return_tensors="pt", padding="max_length", 
            max_length=pad_len, truncation=True, return_token_type_ids=False)
        
        # get shape and pad to `pad_len`
        tokenizer_padded_shape = self.tokenized_revs['input_ids'].shape
        if tokenizer_padded_shape[1] < pad_len:
            for field in ['input_ids', 'attention_mask']:
                self.tokenized_revs[field] = self.__pad_to_fixedlen(
                    self.tokenized_revs[field])
            
        input_ids_tensor = self.tokenized_revs['input_ids']
        
        # tokenize aspairs
        self.revs_asp_mention_locs = []
        for i, aspair_list in enumerate(aspairs):
            senti_id_to_asp = defaultdict(list)
            for asp, senti in aspair_list:
                senti_id = tokenizer(senti, 
                    add_special_tokens=False)['input_ids'][0]
                senti_id_to_asp[senti_id].append(asp)

            # asp_id_loc is (num_asp * pad_len) matrix, saves location of asp
            asp_mention_loc = np.zeros(
                (self.num_asp, self.pad_len), dtype=np.int32)
            for senti_id in senti_id_to_asp:
                loc_vector = (input_ids_tensor[i] == senti_id)\
                             .numpy().astype(np.int32)
                for asp_i in senti_id_to_asp[senti_id]:
                    asp_mention_loc[asp_i] += loc_vector
            self.revs_asp_mention_locs.append(csr_matrix(asp_mention_loc))
    
    def get_review_text(self):
        return self.reviews
    
    def get_anno_tkn_revs(self):
        """
        Get annotated tokenized reviews

        Return:
            self.tokenized_revs - PyTorch style tokenization. 
                Two fields: `input_ids`, `attention_mask`
            self.revs_asp_mention_locs - list of sp matrixs
        """
        return self.tokenized_revs, self.revs_asp_mention_locs
    
    def set_anno_tkn_revs(self, new_anno_tkn_revs):
        self.tokenized_revs = new_anno_tkn_revs
    
    def get_rev_size(self):
        return self.tokenized_revs['input_ids'].shape[0]
    
    def __pad_to_fixedlen(self, tensor):
        """pad to fixed length.

        Args:
            tensor - tensor already padded to a certain flexible length 
        """
        height = tensor.shape[0]
        width = self.pad_len - tensor.shape[1]
        return torch.cat(
            (tensor, torch.zeros(size=(height, width), dtype=tensor.dtype)), 
            dim=1)


def agg_tokenized_data(df, n_partition=5):
    """from annotation to user/item annotations
    
    Args:
        df - the dataframe of reviews and annotations
    Returns:
        user_revs - (dict{user_id: EntityReviewAggregation}) tokenized review 
            and aspect locations aggregated by users
        item_revs - same but aggregated by items
    """

    def merge_dict(dict1, dict2):
        return (dict2.update(dict1))

    def process(row):
        return EntityReviewAggregation(
            reviews=row.original_text, aspairs=row[COL_ASPAIRS],
            pad_len=args.max_pad_length, num_asp=args.num_aspects)

    if not all([x in df.columns for x in [COL_REV_TEXT, COL_ASPAIRS]]):
        raise KeyError("Missing a column from review text and aspairs")

    pandarallel.initialize(nb_workers=args.num_workers, progress_bar=True,
        use_memory_fs=False)
    
    revs_to_return = []
    # groupby users
    useful_cols = ["original_text", COL_ASPAIRS]
    for task in ['user_id', 'item_id']:
        print("task {}".format(task))
        uniq_ids = list(df[task])
        partition_ids = defaultdict(list)
        result_revs_list = []
        for id_ in uniq_ids:
            partition_ids[int(id_[2:]) % n_partition].append(id_)
        
        for partition in partition_ids.keys():
            print("partition {}".format(partition))
            sub_df = df[df[task].isin(partition_ids[partition])]
            # sub_revs = sub_df.groupby(task)[useful_cols].progress_apply(process)
            sub_revs = sub_df.groupby(task)[useful_cols].parallel_apply(process)
            # result_revs_list.append(dict(sub_revs))
            dump_pkl("./temp/p{}_{}".format(partition, task), dict(sub_revs))
        
        result_revs = dict()
        for _revs in result_revs_list:
            result_revs = merge_dict(result_revs, _revs)
        revs_to_return.append(result_revs)

        
    # user_revs = df.groupby('user_id')[useful_cols].progress_apply(process)
    # item_revs = df.groupby('item_id')[useful_cols].progress_apply(process)

    # old code
    # user_revs = df.groupby('user_id')[useful_cols].parallel_apply(process)
    # item_revs = df.groupby('item_id')[useful_cols].parallel_apply(process)

    # convert to dict
    return dict(user_revs), dict(item_revs)

def main():
    """
    Two parts for post processing. 
    Part I:
        Sentiment term labeling
        TODO: to be done after modeling is done!
    
    Part II:
        CF (user graphs), which has been solved by `graph.py` and build_graph.
        It takes care of the dumping.
    """

    # fix path to be ./data/amazon/home_kitchen/
    args.data_path += '/' if args.data_path[-1] != '/' else ""

    # =====================
    # Part I - labeling
    # =====================

    # load dataframe
    # TODO: put in the correct csv file of train data with aspairs
    # df = pd.read_csv(args.data_path + "train_data_aspairs.csv")
    df = pd.read_pickle(args.data_path + "train_data_aspairs.pkl")

    print("[postprocess] processing user/item annotation reviews ...")
    user_anno_tkn_rev, item_anno_tkn_rev= agg_tokenized_data(df)

    dump_pkl(args.data_path + "user_anno_tkn_revs.pkl", user_anno_tkn_rev)
    dump_pkl(args.data_path + "item_anno_tkn_revs.pkl", item_anno_tkn_rev)

    # =====================
    # Part II - user/item graph
    # =====================

    if args.build_nbr_graph:
        from graph import build_graph
        print("[postprocess] processing user/item neighbor data ...")
        user_nbr, item_nbr = build_graph(path=args.data_path)
        dump_pkl(args.data_path + "user_nbr_item.pkl", user_nbr)
        dump_pkl(args.data_path + "item_nbr_user.pkl", item_nbr)

    print("[postprocess] done! Save four files `anno_tkn_revs` and `nbr`")


if __name__ == "__main__":
    main()
