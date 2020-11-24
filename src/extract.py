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

import pandas as pd

from annotate import load_train_file

def load_aspect_term_map(path):
    """load Fine-grained aspect term to Coarse-grained aspect category mapping

    Args:
        path - the path to the mapping
    Return:
        map_ - the map
    
    TODO: to implement
    """
    map_ = dict()

    return map_


def main(args):
    # load training data
    df = load_train_file(path=args.data_path)

    # load aspect mapping
    aspect_map = load_aspect_term_map(path=args.aspect_map_path)

    # annotate aspect
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