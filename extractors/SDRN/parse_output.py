"""
    parse output file

    Author: Zeyu Li <zeyuli@g.ucla.edu> or <zyli@cs.ucla.edu>
"""

import sys
import os
import pickle

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


if __name__ == "__main__":
    if len(sys.argv) != 2 + 1:
        print("Usage\n\tpython {} [training_set] [subset]".format(sys.argv[0]))
        sys.exit()
    
    train_set = sys.argv[1]
    subset = sys.argv[2]
    print("load file")
    data = load_output_annotation_file(train_set, subset)
    aspect_set, sentiment_set = parse_annotation(data)

    annotation_dir = "./data/anno_{}/".format(subset)
    os.makedirs(annotation_dir, exist_ok=True)
    dump_pkl(annotation_dir + "aspect_terms.pkl", aspect_set)
    dump_pkl(annotation_dir + "sentiment_terms.pkl", sentiment_set)

    print("Minded {} unique aspect terms and {} unique sentiment terms".format(
            len(aspect_set), len(sentiment_set)))
    print("Results have been saved to {}".format(annotation_dir))