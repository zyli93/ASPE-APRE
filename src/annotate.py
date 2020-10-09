"""
    Annotation for aspects in natural languages

    Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
import sys
import argparse
import spacy
import pandas as pd
import ujson as json
from collections import Counter
from itertools import permutations
from utils import clean_str2
from tqdm import tqdm
from math import log

from nltk import pos_tag_sents
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import words

from utils import dump_pkl, load_pkl


def load_train_file(args):
    """load train data file"""
    print("[Annotate] loading training df ...", end=" ")
    in_file = args.path + "/train_data.csv"
    df = pd.read_csv(in_file)
    print("Done!")
    return df


def load_seed_words():
    """load seed works for auto aspect detection"""
    print("[Annotate] loading seed words ...", end=" ")
    seed_words_file = "./configs/seed_words.json"
    if not os.path.exists(seed_words_file):
        raise FileNotFoundError(
            "Seed words doesn't exist! Please refer to README for details.")
    with open(seed_words_file, "r") as fin:
        seed_words = json.load(fin)
    if not all([x in seed_words for x in ["POS", "NEG"]]):
        raise KeyError(
            "Config has to include all POS and NEG")
    print("Done!")
    return seed_words


def load_postag_filters(args):
    """load pos tag filters"""
    grain = "fine"
    pos_json = "./configs/fine_grained.pos.json"
    with open(pos_json, "r") as fin:
        pos_filters = json.load(fin)
    print("[Annotate] using {}-grained POS tags".format(grain))
    return pos_filters


def compute_pmi(args, df):
    """compute vocabulary and PMI
    Args:
        df - the input dataframe

    Returns:
        total_window_counter - the total number of windows in the corpus
        single_counter - the counter of single tokens
        pair_counter - the counter of pair of tokens
        pmi_vocab - vocabulary of pmi processing
    """
    # TODO: see if vocab should be returned.

    if not hasattr(args, "pmi_window_size"):
        raise AttributeError("--pmi_window_size has to be specified!")
    print("[Annotate] compute P(i) and P(i,j) ...", end=" ")

    # TODO: to discard the rare tokens as they are incorrect!

    # initialize counters
    ws = args.pmi_window_size
    single_counter, pair_counter = Counter(), Counter()
    total_window_counter = 0

    text_col = df["text"]
    for line in tqdm(text_col):
        line = line.lower()
        for sent in sent_tokenize(line):
            word_tokens = word_tokenize(sent)
            # win_size=3, sent="a b c d" => windows = ["a b c", "b c d"]
            # win_size=3, sent="a b" => windows = ["a b"]
            for i in range(0, max(len(word_tokens)+1-ws, 1)):
                end = min(i+ws, len(word_tokens))
                token_window = word_tokens[i: end]
                single_counter.update(token_window)
                pair_counter.update(permutations(token_window, 2))
                total_window_counter += 1
    print("Done!")

    # build p_i and p_ij and divide them by total_window_counter
    print("[Annotate] compute PMI ...", end=" ")

    # delete rare tokens as they usually contain misspellings
    for word in list(single_counter.keys()):
        freq = single_counter[word]
        if freq < args.token_min_count:
            del single_counter[word]

    # delete pairs with rare tokens
    for pair in list(pair_counter.keys()):
        if pair[0] not in single_counter or pair[1] not in single_counter:
            del pair_counter[pair]

    p_i = single_counter.copy()
    p_ij = pair_counter.copy()
    p_i = {key: value / total_window_counter for key, value in p_i.items()}
    p_ij = {key: value / total_window_counter for key, value in p_ij.items()}

    # compute PMI, PMI[i,j] = log(P(i,j) / (P(i)*P(j))
    pmi = {key: log(value/(p_i[key[0]] * p_i[key[1]]))
           for key, value in p_ij.items()}
    dump_pkl(args.path + "/pmi_matrix.dict.pkl", pmi)

    # save pmi vocabularies
    pmi_vocabs = list(single_counter.keys())
    dump_pkl(args.path + "/pmi_vocabs.list.pkl", pmi_vocabs)

    print("Done!")
    print("[Annotate] PMI matrix and vocabulary saved in {}".format(args.path))

    # TODO: to delete dump_pkl later after verification
    dump_pkl("pmi.pkl", pmi)  # TODO: how to verify this is correct?

    dump_pkl("pij.pkl", p_ij)
    dump_pkl("pi.pkl", p_i)

    dump_pkl("single_counter.pkl", single_counter)
    dump_pkl("pair_counter.pkl", pair_counter)

    return pmi, pmi_vocabs


def get_vocab_postags(args, df, vocab):
    """Get the Part-of-Speech tagging for vocabulary

    * Q: Why use nltk rather than spacy?
      A: Lessons learned that nltk.pos_tag() is around 18 times faster than 
         spacy.load("en_core_web_sm").

    * Q: Why use nltk.pos_tag_sents rather than nltk.pos_tag?
      A: In nltk version 3.5, the doc of nlkt.pos_tag() mentioned that 
         "NB. Use `pos_tag_sents()` for efficient tagging of more than one sentence."

    * Q: Difference between the outputs of nltk.pos_tag (or nltk.pos_tag_sents) and spacy?
      A: Spacy generously provides four attributes in the tagged object: .pos, .pos_, .tag, .tag_.
         "pos" and "tag" refer to two different level of granularities. "pos" is coarse-grained
         and "tag" is fine-grained. In contrary, nltk only gives fine-grained.

    Based on above three QA's, we use nltk.pos_tag_sents for POS tagging.

    Args:
        args - cmd line input arguments
        df - training dataframe
        vocab - the filtered vocabulary of PMI

    Return:
        vocab_pos_majvote - [Dict] majority vote of most popular word pos: {"word": "NN"}.
    """
    def update_vocab_counter(vocab_pos, token_tuple):
        token, pos = token_tuple
        if token in vocab:
            vocab_pos[token].update([pos])

    print("[Annotate] processing vocabulary part-of-speech ...", end=" ")

    vocab_pos = {word: Counter() for word in vocab}
    vocab = set(vocab)
    text_col = df["text"]
    # text -> sentences -> words
    for text in tqdm(text_col):
        sents = sent_tokenize(text.lower())
        # tagged_sents is a list of lists, tokens look like: ("apple", "NN").
        tagged_sents = pos_tag_sents(map(word_tokenize, sents))
        # tagged_tokens flattens the list of lists
        tagged_tokens = [tag_tuple for sublist in tagged_sents 
                                   for tag_tuple in sublist] 
        # apply update op on all tagged tokens, use `list()` to make map execute.
        list(map(lambda tpl: update_vocab_counter(vocab_pos, tpl), tagged_tokens))
    print("Done!")

    print("[Annotate] majority voting for pos-tagging ...", end=" ")
    # TODO: to be uncommented later
    # vocab_pos_majvote = {word: vocab_pos[word].most_common(1)[0][0] 
    #                      for word in vocab_pos.keys()}  # majvote: majority vote
    vocab_pos_majvote = {}
    for word in vocab_pos:
        try:
            vocab_pos_majvote[word] = vocab_pos[word].most_common(1)[0][0]
        except:
            print(vocab_pos[word])
        
    dump_pkl(args.path + "/postag_of_vocab_full.pkl", vocab_pos)  # TODO: to remove this

    dump_pkl(args.path + "/postag_of_vocabulary.pkl", vocab_pos_majvote)
    print("Done!")
    print("[Annotate] results saved at {}/postag_of_vocabulary.pkl".format(args.path))
    return vocab_pos_majvote


def compute_vocab_polarity_from_seeds(args, seeds, postag_filters, vocab_postags, pmi_matrix):
    """ "Grow" (generate) aspect words from seeds for both aspects (pos/neg). 
    Get token polarity based on PMI values.

    Current plan: 
        (1) filter the needed POS (given by `postag_filters`)
        (2) sort only with certain POS.

    TODO: If a lot of tokens are not in PMI-matrix, we need to increase the window size.

    Args:
        args - cmd line input arguments
        seeds - [Dict] the seed words. {"POS": ["pos_1", ...], "NEG": ["neg_1", ...]}
        quota - the number of related words to return
        postag_filters - [Set] the filters of POS tags. POS tags to keep only. 
            `None` for not using filters.
        vocab_postags - the POS tags of vocabulary
        pmi_matrix - the PMI matrix
    Returns:

    """
    print("[Annotate] computing aspect-sentiment words polarity from seeds ...", end=" ")
    if postag_filters and not isinstance(postag_filters, set):
        postag_filters = set(postag_filters)  # postag_filters contains the POS tags to keep

    # candidate sentiment word polarity
    cand_senti_pol = []

    pos_seeds, neg_seeds = seeds["POS"], seeds["NEG"]

    # guaranteed that words in vocab_postags also in pmi_matrix
    for word in tqdm(vocab_postags.keys()):
        if postag_filters and vocab_postags[word] not in postag_filters:
            continue
        pos_pmi = [pmi_matrix.get((pos_seed, word), 0) for pos_seed in pos_seeds]
        neg_pmi = [pmi_matrix.get((neg_seed, word), 0) for neg_seed in neg_seeds]
        mean_pos_pmi = sum(pos_pmi) / len(pos_pmi) 
        mean_neg_pmi = sum(neg_pmi) / len(neg_pmi)
        polarity = mean_pos_pmi - mean_neg_pmi

        # remove words with little references with both sides of seeds
        if pos_pmi.count(0) + neg_pmi.count(0) > 0.8 * (len(pos_pmi) + len(neg_pmi)):
            continue

        cand_senti_pol.append((word, vocab_postags[word], polarity, mean_pos_pmi, mean_neg_pmi))
    cand_senti_pol.sort(reverse=True, key=lambda x: x[2])
    cand_df = pd.DataFrame(cand_senti_pol, 
        columns=["word", "POS", "polarity", "Mean Positive PMI", "Mean Negative PMI"])
    
    cand_df.to_csv(args.path + "/cand_senti_pol.csv", index=False)

    # TODO: remove this or save it elsewhere
    print("Done!")
    print("[Annotate] vocab polarity saved at {}/cand_senti_pol.csv".format(args.path))

    return cand_df


# TODO: merge compute polarity words and remove invalid words later!
def remove_invalid_words(args, word_list):
    """Remove invalid words in the word list dataframe 
    
    Args:
        word_list - the dataframe of words
    """
    print("[Annotate] removing invalid words ...", end=" ")
    assert isinstance(word_list, pd.DataFrame), "word_chart should be a dataframe"
    valid_word_list = word_list[word_list["word"].isin(words.words())]
    valid_word_list.to_csv(args.path + "/valid_cand_senti_pol.csv", index=False)
    print("Done!")
    print("[Annotate] valid vocab saved at {}/valid_cand_senti_pol.csv".format(args.path))
    return valid_word_list


def main(args):
    # check args.path
    if not os.path.exists(args.path):
        raise ValueError("Invalid path {}".format(args.path))
    if args.path[-1] == "\/":
        args.path = args.path[:-1]

    # load config files
    seeds = load_seed_words()
    train_df = load_train_file(args)
    postag_filters = load_postag_filters(args)

    # compute PMI and POS tag
    # TODO: uncomment later
    pmi_matrix, pmi_vocab = compute_pmi(args, train_df)
    word_to_postag = get_vocab_postags(args, train_df, pmi_vocab)

    # pmi_matrix = load_pkl(args.path + "/pmi_matrix.dict.pkl")
    # pmi_vocab = load_pkl(args.path + "/pmi_vocabs.list.pkl")
    # word_to_postag = load_pkl(args.path + "/postag_of_vocabulary.pkl")

    # generate opinion words
    word_pol_df = compute_vocab_polarity_from_seeds(
        args,
        seeds=seeds,
        postag_filters=postag_filters['keep'],
        vocab_postags=word_to_postag,
        pmi_matrix=pmi_matrix)
    remove_invalid_words(args, word_pol_df)

    # TODO: read the PMI post, think about relate between PMI and actual word

    
    # get polarity here!
    # maybe add spell check!
    # quota = args.aspect_candidate_quota_per_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset goodreads")

    parser.add_argument(
        "--pmi_window_size",
        type=int,
        default=3,
        help="The window size of PMI cooccurance relations. Default=3.")

    parser.add_argument(
        "--token_min_count",
        type=int,
        default=20,
        help="Minimum token occurences in corpus. Rare tokens are discarded. Default=20.")

    parser.add_argument(
        "--aspect_candidate_quota_per_seed",
        type=int,
        default=100,
        help="Number of candidate aspect opinion word to extract per seed. Default=3.")

    args = parser.parse_args()
    main(args)
