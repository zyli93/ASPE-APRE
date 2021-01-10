"""
    Build user and item graph for CF part

    File name: graph.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/07/2020
    Date Last Modified: TODO
    Python Version: 3.6
"""

import os
import argparse
import pandas as pd

from utils import load_pkl, dump_pkl

# Arguments parser
parser = argparse.ArgumentParser()

# E.g.: ./data/amazon/home_kitchen
parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Path to the dataset.")

args = parser.parse_args()

def build_graph(path):
    data_path = path + "train_data.csv"
    train_df = pd.read_csv(data_path)

    # obtain user to item dictionary
    # user_id,item_id,rating,text,original_text
    print("\t[graph] computing user neighbors (which are items) ...")
    user_nbr = train_df.groupby("user_id")['item_id'].apply(list)
    user_nbr = user_nbr.to_dict()

    # obtain item to user dictionary
    print("\t[graph] computing item neighbors (which are users) ...")
    item_nbr = train_df.groupby("item_id")['user_id'].apply(list)
    item_nbr = item_nbr.to_dict()

    return user_nbr, item_nbr

if __name__ == "__main__":
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError("Directory not found")

    # load dataframe 
    args.data_path += '/' if args.path_path[-1] != '/' else ""
    build_graph(path=args.data_path)