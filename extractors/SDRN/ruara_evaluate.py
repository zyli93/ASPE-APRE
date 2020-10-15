"""
Evaluate script for RUARA

Author:
    Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>


TODO:
    1. load model
    2. build datasets
        1. transfer to torch.load format
        2. save datasets
    3. apply model to our dataset
    4. decipher models output
"""

import pandas as pd
from tqdm import tqdm
from main import InputFeatures
from ...src.utils import clean_str2

DATA = "../../data/amazon/"
VOCAB = "./bert_model/vocab.txt"

def load_vocab():
    with open(VOCAB, "r") as fin:
        lines = fin.readlines()
    return {line.strip(): i for i, line in enumerate(lines)}

def process_data():
    df = pd.read_csv(DATA, index=False)
    sentences = []
    text_col = df['text']  # each cell has multiple lines, need to append [CLS] and [SEP]
    # TODO: check [CLS] and [SEP]




def load_model():
    pass


def build_input_features():
    pass


if __name__ == "__main__":
    subset = "digital_music"
    DATA = DATA + subset + "/train_data.csv"
