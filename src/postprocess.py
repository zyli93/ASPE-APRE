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
'''

import sys
import os
import argparse

from collections import defaultdict
import numpy as np
import pandas as pd

from nltk.tokenize import sent_tokenize
from scipy.sparse import csr_matrix
from transformers import BertTokenizerFast
from pandarallel import pandarallel

from graph import build_graph
from utils import dump_pkl

COL_REV_TEXT = "original_text"  # TODO: fix this
COL_ASPAIRS = "as_pairs"
COL_ANNO_REV = "annotated_review"

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

args = parser.parse_args()


class AnnotatedReviewInID:
    def __init__(self, review_text, aspairs, num_asp, max_pad_len=100):
        self.total_num_asp = num_asp
        self.pad_len = max_pad_len

        # =====================================================================
        # get a tokenizer
        #   tokenizer will process two things:
        #       1. Review text. Produces: input_ids, attn_mask, token_type_ids.
        #       2. The sentiment term inside the aspairs
        # =====================================================================
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        # =====================================================================
        # handle review in different formats.
        #   for str, use nltk.sent_tokenize to split it into list of sentences
        #   for list of sentences, leave it be
        # =====================================================================
        if isinstance(review_text, str):  # string of review
            self.review_sents = sent_tokenize(review_text)
        elif isinstance(review_text, list):  # list of sentences
            self.review_sents = review_text
        else:  # TypeError of compatible review_text
            raise TypeError(
                "review_text can only be [str] or [list] of sentences. " +
                "But a {} received!".format(str(type(review_text))))
        
        # return PyTorch-styled ("pt") padded tensors
        self.tokenized_revsents = tokenizer.tokenize(
            self.review_sents, return_tensors="pt", padding=True)
        # get the shape of tokenized_review_sents
        tkn_revsents_input_ids = self.tokenized_revsents['input_ids']
        input_ids_shape = tuple(tkn_revsents_input_ids)
        
        # =====================================================================
        # handle aspairs (int, str). Here's the logic
        #   1. senti_id_to_asp is a defaultdict(list) where keys are 
        #      sentiment term tokenized ids and values are lists of corresponding
        #      aspects modified by that sentiment. 
        #   2. For each id, we use masking matrices to record their positions
        #      (coordinates) and add the masking matrices to the corresponding
        #      self.asp_senti_coord matrix. self.asp_senti_coord is a dictionary 
        #      of aspect_id to csr_matrices where the sparse matrices are 
        #      the masks.
        # =====================================================================

        senti_id_to_asp = defaultdict(list)

        # save aspect-sentiment term coordinate mask
        def new_csr_matrix():
            return csr_matrix(input_ids_shape, dtype=np.int32)
        self.asp_senti_coord = defaultdict(new_csr_matrix)
        
        # Step 1: build senti_id --> asp_id mapping
        for asp_id, senti_term in aspairs:
            senti_id = tokenizer.tokenize(senti_term, 
                add_special_tokens=False, return_attention_mask=False)
            senti_id = senti_id['input_ids'][0]
            senti_id_to_asp[senti_id].append(asp_id)
        
        # Step 2: aspect senti location
        for senti_id in senti_id_to_asp:
            senti_coord = csr_matrix(
                (tkn_revsents_input_ids == senti_id)
                    .numpy().astype(np.int32))
            inv_aspects = senti_id_to_asp[senti_id]
            for inv_asp in inv_aspects:
                self.asp_senti_coord[inv_asp] += senti_coord
        
        # after the processing, reset defaultdict to dict
        self.asp_senti_coord = dict(self.asp_senti_coord)

        # =====================================================================
        # attribute variables to keep:
        #     * tokenizer                - to del
        #     * tkn_revsents_input_ids   - to del
        #     * senti_id_to_asp          - to del
        #     * self.review_sents        - to keep
        #     * self.tokenized_revsents  - to keep
        #     * self.asp_senti_coord     - to keep
        # =====================================================================
    
    def get_tokenized_revsents(self):
        return self.tokenized_revsents
    
    def get_asp_coord(self):
        return self.asp_senti_coord
    
    def get_review_text(self):
        return self.review_sents
    
    def get_num_asp(self):
        return self.total_num_asp


def agg_tokenized_data(df):
    """convert data into AnnotatedReviewInID objects
    Args:
        df - the input dataframe
    Return:
        converted user_anno_reviews, item_anno_reviews
    """
    def process(x):
        """func to apply"""
        annotated_review = AnnotatedReviewInID(
            review_text=x[COL_REV_TEXT],
            aspairs=x[COL_ASPAIRS],
            num_asp=args.num_aspects,
            max_pad_len=args.max_pad_length)
        return annotated_review

    pandarallel.initialize(
        nb_workers=args.num_workers, progress_bar=True, verbose=1)
    df[COL_ANNO_REV] = df.parallel_apply(process)

    user_anno_reviews = df.groupby('user_id')[COL_ANNO_REV].agg(list)
    item_anno_reviews = df.groupby('item_id')[COL_ANNO_REV].agg(list)

    # convert user anno reviews to dictionary for easier store
    user_anno_reviews = dict(user_anno_reviews)
    item_anno_reviews = dict(item_anno_reviews)

    return user_anno_reviews, item_anno_reviews


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
    df = pd.read_csv(args.data_path + "train_data.csv")

    print("[postprocess] processing user/item annotation reviews ...")
    user_anno_rev_in_id, item_anno_rev_in_id = agg_tokenized_data(df)
    dump_pkl(args.data_path + "user_anno_idrev.pkl", user_anno_rev_in_id)
    dump_pkl(args.data_path + "item_anno_idrev.pkl", item_anno_rev_in_id)

    # =====================
    # Part II - user/item graph
    # =====================

    print("[postprocess] processing user/item neighbor data ...")
    user_nbr, item_nbr = build_graph(path=args.data_path)
    dump_pkl(args.data_path + "user_nbr_item.pkl", user_nbr)
    dump_pkl(args.data_path + "item_nbr_user.pkl", item_nbr)


if __name__ == "__main__":
    main()
