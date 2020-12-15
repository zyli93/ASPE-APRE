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

import pickle
import re
import string
import sys
import os
import torch
import pandas as pd
from tqdm import tqdm
from main import InputFeatures, read_data, evaluate
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import pickle


DATA = "../../data/amazon/"
VOCAB = "./bert_model/vocab.txt"

def dump_pkl(path, obj):
    """helper to dump objects"""
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)


def load_pkl(path):
    """helper to load objects"""
    with open(path, "rb") as fin:
        return pickle.load(fin)


def new_read_data(train_features, type, batchsize):
    assert type in ["train", "test"]
    all_input_ids = torch.tensor([f.token_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.token_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segmentId for f in train_features], dtype=torch.long)
    # all_relations = torch.tensor([f.relations for f in train_features], dtype=torch.long)
    # all_labels = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
    # all_labels[:, :1] = torch.ones(all_labels.size(0), 1).long()
    if type == "train":
        # train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_relations, all_labels)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        train_sampler = RandomSampler(train_data)
    else:
        # train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_relations, all_labels)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batchsize)
    return train_dataloader


def clean_str2(s):
    """Clean up the string
    * New version, removing all punctuations
    Cleaning strings of content or title
    Original taken from 
    [https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py]
    Args:
        string - the string to clean
    Return:
        _ - the cleaned string
    
    Things that are done here:
        1. get rid of
            a. contents in brackets
            b. punctuations
            c. words containing numbers in them
            d. double quotes
        2. replace
            a. 's to s (user's -> users)
            b. 've to ve (I've -> Ive)
            c. 't, 're, 'd, 'll
            d. more than one (2+) spaces -> one space
    
    The right order to process the signs:
    1. 's/'ve/... to no "'"
    2. 
    """
    ss = s

    # if there are square brackets with characters in them, get rid of the whole thing
    # ss = re.sub('\[.*?\]', '', ss)

    # if there are strings concatenated by puncts get rid of them
    ss = re.sub( r'([,.!?])([a-zA-Z0-9])', r'\1 \2', ss)

    # gets rid of words containing numbers in them
    ss = re.sub('\w*\d\w*', '', ss)
    ss = re.sub('[\'\"]', '', ss)  # removing double quotes
    ss = re.sub('\n', '', ss)  # removing line breaks
    # ss = re.sub("re-", "re", ss) # specifically fix re- to re
    ss = re.sub(r"\'s", "s", ss)
    ss = re.sub(r"\'ve", "ve", ss)
    ss = re.sub(r"n\'t", "nt", ss)
    ss = re.sub(r"\'re", "re", ss)
    ss = re.sub(r"\'d", "d", ss)
    ss = re.sub(r"\'ll", "ll", ss)
    ss = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", ss)

    # if anything is any of those punctuation marks, get rid of it
    ss = re.sub('[%s]' % re.escape(string.punctuation), '', ss)
    ss = re.sub(r"\s{2,}", " ", ss)
    # ss = re.sub(r"  ", " ", ss)
    return ss.strip().lower()

def load_vocab():
    with open(VOCAB, "r") as fin:
        lines = fin.readlines()
    return {line.strip(): i for i, line in enumerate(lines)}


def load_and_process_data(head=None):
    df = pd.read_csv(DATA)
    if head:
        df = df.head(head)
    sentences = []
    text_col = list(df['text']) # each cell has multiple lines
    sentences = [
        ["[CLS]"] + clean_str2(sent).split(" ") + [".", "[SEP]"] 
        for text in tqdm(text_col) for sent in text.split("\n") ]
    return sentences


def build_input_features(sentences, mapping, pad_length):
    instances = []
    for sentence in tqdm(sentences):
        if len(sentence) > pad_length:
            continue
        try:
            token_ids = [mapping[x] for x in sentence]
            padded_tokens_ids = token_ids + (pad_length - len(token_ids)) * [0]
            instances.append(
                InputFeatures(
                    tokens=sentence,
                    token_ids=padded_tokens_ids,
                    token_mask=[1] * len(sentence) + [0] * (pad_length-len(sentence)),
                    segmentId=[0] * pad_length,
                    labels=['O'] * len(sentence),
                    # label_ids=[0] * pad_length,
                    # relations=[[0] * pad_length for _ in range(pad_length)],
                    label_ids=None,
                    relations=None,
                    gold_relations=None,
                    token_to_orig_map=None))
        except:
            continue
    return instances


def recover_label_inference(targetPredict, all_input_mask):
    pred_variable = targetPredict
    mask_variable = all_input_mask
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    pred_label = []
    for idx in range(batch_size):
        pred = [pred_tag[idx][idy] - 1 for idy in range(seq_len) if mask[idx][idy] != 0]
        pred_label.append(pred)
    return pred_label


def inference(dataloader, inf_set, model, output_file_path, ifgpu=True):
    pred_results = []

    model.eval()

    for step, batch in tqdm(enumerate(dataloader)):
        if ifgpu:
            batch = tuple(t.cuda() for t in batch)  # multi-gpu does scattering it-self
        all_input_ids, all_input_mask, all_segment_ids = batch
        max_seq_len = torch.max(torch.sum(all_input_mask, dim=1))
        all_input_ids = all_input_ids[:, :max_seq_len].contiguous()
        all_input_mask = all_input_mask[:, :max_seq_len].contiguous()
        all_segment_ids = all_segment_ids[:, :max_seq_len].contiguous()
        targetPredict, _ = model(all_input_ids, all_segment_ids, all_input_mask)

        # get real label
        pred_label = recover_label_inference(targetPredict, all_input_mask)
        pred_results += pred_label

    # write to file
    labelDic = ["O", "B-T", "I-T", "B-P", "I-P", "O"]  # len=6
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for k in tqdm(range(len(inf_set))):
            words = inf_set[k].tokens
            pred = pred_results[k]
            # gold = inf_set[k].labels
            for j in range(len(words)):
                try:
                    output_file.write(words[j] + "\t" + labelDic[pred[j]] + "\n")
                except:
                    print("j {} len words {} pred {}".format(
                        j, len(words), len(pred)))
                    print(words)
                    print(pred)
            output_file.write("\n")

if __name__ == '__main__':
    if len(sys.argv) != 5 + 1:
        print("Usage\n\t" + 
              "python {} [do_process] [training data] [annotate subset] [head]".format(sys.argv[0]))
        sys.exit()

    to_process = eval(sys.argv[1])
    training_set = sys.argv[2]  # 2014Lap, 2015Res, 2014Res
    subset = sys.argv[3]
    model_path = "./model/{}/modelFinal.model".format(training_set)
    batch_size = 100
    head = int(sys.argv[4])
    gpu_id = sys.argv[5]

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    if head < 0:
        head = None
    output_file_dir = "./infer_on_ruara_{}/".format(training_set)

    os.makedirs(output_file_dir, exist_ok=True)

    DATA = DATA + subset + "/train_data.csv"

    if to_process:
        print("load & process data + load bert vocab ...")
        sentences = load_and_process_data(head)  # list of list of tokens
        vocab_to_id = load_vocab()

        print("build input features ...")
        instances = build_input_features(sentences, mapping=vocab_to_id, pad_length=100)

        # dump_pkl("instances.pkl", instances)
        # sys.exit()

        print("create dataloader ...")
        dataloader = new_read_data(instances, "test", batchsize=batch_size)

        print("save processe data ...")
        dump_pkl("./data/dataloader_{}.pkl".format(subset), dataloader)
        dump_pkl("./data/instances_{}.pkl".format(subset), instances)
    
    else:
        print("load dataloader and instances ...")
        dataloader = load_pkl("./data/dataloader_{}.pkl".format(subset))
        instances = load_pkl("./data/instances_{}.pkl".format(subset))
        # dataloader, instances = None, None

    print("load model ...")
    model = torch.load(model_path, map_location="cuda:0")

    print("running inference!")
    with torch.no_grad():
        fname = "annotation_{}.txt".format(subset)
        inference(dataloader, instances, model, output_file_dir+fname, True)

