"""
    parse output file

    Author: Zeyu Li <zeyuli@g.ucla.edu> or <zyli@cs.ucla.edu>
"""

import sys
import os
import pickle

ALL_TRAIN_SETS = ['2014Lap', '2014Res', '2015Res']

def load_output_annotation_file(train_set, subset):
    ANN_DIR = "infer_on_ruara_{}/".format(train_set)
    with open(ANN_DIR + "annotation_{}.txt".format(subset), "r") as fin:
        return fin.readlines()

def parse_annotation(data):
    aspect_list, sentiment_list = [], []

    total_length = len(data)
    i = 0
    while i < total_length:
        line = data[i].strip()
        if len(line) == 0:  # empty line
            i += 1
            continue
        token, label = line.split("\t")
        if not label in ["B-T", "B-P"]:
            i += 1
            continue
        j = 1
        while data[i+j].strip().split("\t")[1] in ["I-T", "I-P"]:
            j += 1
        token_seq = [x.strip().split("\t")[0] for x in data[i: i+j]]

        i += j
        if label == "B-T":
            aspect_list.append(" ".join(token_seq))
        else:
            sentiment_list.append(" ".join(token_seq))
    return set(aspect_list), set(sentiment_list)


def dump_pkl(path, obj):
    with open(path, "wb") as fout:
        pickle.dump(obj, fout)

def load_pkl(path):
    with open(path, "rb") as fin:
        return pickle.load(fin)


if __name__ == "__main__":
    if len(sys.argv) != 3 + 1:
        print("Usage\n\tpython {} [task] [training_set] [subset]".format(sys.argv[0]))
        sys.exit()
    
    task = sys.argv[1]
    train_set = sys.argv[2]
    subset = sys.argv[3]

    if task == "parse":
        print("load file {} {}".format(train_set, subset))
        data = load_output_annotation_file(train_set, subset)
        aspect_set, sentiment_set = parse_annotation(data)

        annotation_dir = "./data/anno_{}_{}/".format(train_set, subset)
        os.makedirs(annotation_dir, exist_ok=True)
        dump_pkl(annotation_dir + "aspect_terms.pkl", aspect_set)
        dump_pkl(annotation_dir + "sentiment_terms.pkl", sentiment_set)

        print("Minded {} unique aspect terms and {} unique sentiment terms".format(
                len(aspect_set), len(sentiment_set)))
        print("Results have been saved to {}".format(annotation_dir))

    elif task == "merge":
        print("merge three parsed sentiment sets for {}".format(subset))
        senti_sets_sizes = []
        new_set = set()
        for train_set in ALL_TRAIN_SETS:
            annotation_dir = "./data/anno_{}_{}/".format(train_set, subset)
            cur_set = load_pkl(annotation_dir+"sentiment_terms.pkl")
            new_set = new_set.union(cur_set)
            senti_sets_sizes.append(len(cur_set))
        
        print("\tSizes = ", senti_sets_sizes)
        print("\tSize of merge set: {}".format(len(new_set)))
        
        dump_path = "./data/senti_term_{}_merged.pkl".format(subset)
        print("Merged results go to {}".format(dump_path))
        dump_pkl(dump_path, new_set)
    
    else:
        raise ValueError(
            "Cannot process task=={}. Must be parse/merge/both".format(task))

        


