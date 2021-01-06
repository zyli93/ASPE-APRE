'''
    Extracting the terms in the document

    File name: extract.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 11/23/2020
    Date Last Modified: TODO
    Python Version: 3.6+
'''

import os
import sys
import argparse
import ujson as json

import pandas as pd


def load_aspect_term_map(path):
    """load Fine-grained aspect term to Coarse-grained aspect category mapping

    Args:
        path - the path to the mapping
    Return:
        map_ - the map
    """
    print("\t[Extract] loading aspect term mapping words ...", end=" ")

    dataset = args.path.split("\/")[-1]  # get dataset, e.g., `digital music`
    aspect_map_file = "./configs/aspect_term_maps/{}.json".format(dataset)
    if not os.path.exists(aspect_map_file):
        raise FileNotFoundError(
            "Aspect words doesn't exist! Please refer to README for details.")

    with open(aspect_map_file, "r") as fin:
        map_ = json.load(fin)
    print("Done!")

    return map_


def main(args):
    # check args.path
    if not os.path.exists(args.path):
        raise ValueError("Invalid path {}".format(args.path))
    if args.path[-1] == "\/":
        args.path = args.path[:-1]

    # load training data
    train_df = pd.read_pickle(args.data_path + "train_data_dep.pkl")

    # load aspect mapping: {cat: [list of terms]}
    asp_cat2term = load_aspect_term_map(path=args.aspect_map_path)
    asp_cat2term_list = [(x, cat) 
        for cat in asp_cat2term  for x in asp_cat2term[cat]]
    asp_term2cat = dict(asp_cat2term_list)  #  {term: cat}
    cat_to_idx = dict(zip(asp_cat2term.keys(), 
                      range(1, len(asp_cat2term.keys()) + 1)))
    

    # annotate aspect`
    
    
    # TODO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the dataset.")

    parser.add_argument(
        "--aspect_map_path",
        type=str,
        required=True,
        help="Path to aspect term to aspect category mapping.")

    args = parser.parse_args()
    main(args)