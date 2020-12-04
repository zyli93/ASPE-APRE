"""
    Annotation for aspects in natural languages

    Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import os
import sys
import argparse
import ujson as json
import re
from collections import Counter
from itertools import permutations
from tqdm import tqdm
from math import log

import pandas as pd
from pandarallel import pandarallel

import spacy
from spacy.symbols import amod
from nltk import pos_tag_sents
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import words
from gensim.models import KeyedVectors

from utils import dump_pkl, load_pkl
from utils import clean_str2


def load_train_file(path):
    """load train data file"""
    print("\t[Annotate] loading training df ...", end=" ")
    in_file = path + "/train_data.csv"
    df = pd.read_csv(in_file)
    print("Done!")
    return df


def load_pmi_seed_words():
    """load seed words for auto aspect detection"""
    print("\t[Annotate] loading seed words ...", end=" ")
    seed_words_file = "./configs/pmi_seed_words.json"
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
    print("\t[Annotate] using {}-grained POS tags".format(grain))
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
    if not hasattr(args, "pmi_window_size"):
        raise AttributeError("--pmi_window_size has to be specified!")
    print("\t[Annotate] compute P(i) and P(i,j) ...", end=" ")

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
    print("\t[Annotate] compute PMI ...", end=" ")

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
    print("\t[Annotate] PMI matrix and vocabulary saved in {}".format(args.path))

    # TODO: to delete dump_pkl later after verification
    dump_pkl("./temp/pmi.pkl", pmi)  # TODO: how to verify this is correct?

    dump_pkl("./temp/pij.pkl", p_ij)
    dump_pkl("./temp/pi.pkl", p_i)

    dump_pkl("./temp/single_counter.pkl", single_counter)
    dump_pkl("./temp/pair_counter.pkl", pair_counter)

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

    def rough_clean(s):
        """change ^-=+/\\ to " " in text as the pos tagger doesn't handle it"""
        s = re.sub(r"\^|-|=|\+|\/", " ", s)
        s = re.sub(r"\\", " ", s)
        return s

    print("\t[Annotate] processing vocabulary part-of-speech ...", end=" ")

    vocab_pos = {word: Counter() for word in vocab}
    vocab = set(vocab)

    # use original text (with cases) to do pos tagging for better accuracy
    original_text_col = df["original_text"]

    # text -> sentences -> words
    for text in tqdm(original_text_col):

        # sents is un-lowered and roughly processed (certain punctation removed)
        #   text for more accurate POS tagging
        # tagged_sents is a list of lists, tokens look like: ("apple", "NN").
        # tagged_tokens flattens the list of lists, lower it before save
        sents = list(map(rough_clean, sent_tokenize(text)))
        tagged_sents = pos_tag_sents(map(word_tokenize, sents))
        tagged_tokens = [(tag_tuple[0].lower(), tag_tuple[1])
                         for sublist in tagged_sents for tag_tuple in sublist]

        # apply update op on all tagged tokens, use `list()` to make map execute.
        list(map(lambda tpl: update_vocab_counter(vocab_pos, tpl), tagged_tokens))
    print("Done!")

    print("\t[Annotate] majority voting for pos-tagging ...", end=" ")
    vocab_pos_majvote = {}
    for word in vocab_pos:
        if len(vocab_pos[word]):
            vocab_pos_majvote[word] = vocab_pos[word].most_common(1)[0][0]

    dump_pkl(args.path + "/postag_of_vocabulary_full.pkl", vocab_pos)
    dump_pkl(args.path + "/postag_of_vocabulary.pkl", vocab_pos_majvote)
    print("Done!")
    print("\t[Annotate] results saved at {}/postag_of_vocabulary.pkl".format(args.path))
    return vocab_pos_majvote


def compute_vocab_polarity_from_seeds(
        args, seeds, postag_filters, vocab_postags, pmi_matrix):
    """ "Grow" (generate) aspect words from seeds for both aspects (pos/neg). 
    Get token polarity based on PMI values.

    Current plan: 
        (1) filter the needed POS (given by `postag_filters`)
        (2) only sort words with certain POS.
        (3) remove tokens that are not words via nltk.corpus.words.words()

    Args:
        args - cmd line input arguments
        seeds - [Dict] the seed words. {"POS": ["pos_1", ...], "NEG": ["neg_1", ...]}
        quota - the number of related words to return
        postag_filters - [Set] the filters of POS tags. POS tags to keep only. 
            `None` for not using filters.
        vocab_postags - the POS tags of vocabulary
        pmi_matrix - the PMI matrix
    Returns:
        cand_df - candidate sementic terms dataframe with all PMI information
    """
    print("\t[Annotate] computing aspect-sentiment words polarity from seeds ...", end=" ")
    if postag_filters and not isinstance(postag_filters, set):
        # postag_filters contains the POS tags to keep
        postag_filters = set(postag_filters)

    # candidate sentiment word polarity
    cand_senti_pol = []
    valid_word_vocab = set(words.words())

    pos_seeds, neg_seeds = seeds["POS"], seeds["NEG"]

    # guaranteed that words in vocab_postags also in pmi_matrix
    for word in tqdm(vocab_postags.keys()):
        if postag_filters and vocab_postags[word] not in postag_filters:
            continue
        if word not in valid_word_vocab:
            continue
        pos_pmi = [pmi_matrix.get((pos_seed, word), 0)
                   for pos_seed in pos_seeds]
        neg_pmi = [pmi_matrix.get((neg_seed, word), 0)
                   for neg_seed in neg_seeds]
        mean_pos_pmi = sum(pos_pmi) / len(pos_pmi)
        mean_neg_pmi = sum(neg_pmi) / len(neg_pmi)
        polarity = mean_pos_pmi - mean_neg_pmi

        # remove words with little references with both sides of seeds
        if pos_pmi.count(0) + neg_pmi.count(0) > 0.8 * (len(pos_pmi) + len(neg_pmi)):
            continue

        cand_senti_pol.append(
            (word, vocab_postags[word], polarity, mean_pos_pmi, mean_neg_pmi))
    cand_senti_pol.sort(reverse=True, key=lambda x: x[2])
    cand_df = pd.DataFrame(cand_senti_pol,
                           columns=["word", "POS", "polarity", "Mean Positive PMI", "Mean Negative PMI"])

    cand_df.to_csv(args.path + "/cand_senti_pol.csv", index=False)

    print("Done!")
    print("\t[Annotate] vocab polarity saved at {}/cand_senti_pol.csv".format(args.path))

    return cand_df


def filter_senti_terms_by_glove(args, df):
    """rectify polarity with GloVe

    df - the dataframe for PMI generated terms
    quota - the num of words to use in each polarity
    """
    with open("./configs/glove_seed.json", "r") as fin:
        glove_seeds = json.load(fin)
    pos_seeds, neg_seeds = glove_seeds['POS'], glove_seeds['NEG']

    def avg_similarity_to_seeds(target, seeds):
        return sum([glove.similarity(target, seed) 
                    for seed in seeds]) / len(seeds)
    
    def get_terms_in_polarity(iter_, senti_terms, func):
        while len(senti_terms) < args.num_senti_terms_per_pol:
            try:
                t = next(iter_)
            except:
                print("\tStopped early. Unable to get {} terms.".format(
                    args.num_senti_terms_per_pol))
                break
            pos_sim = avg_similarity_to_seeds(t, pos_seeds)
            neg_sim = avg_similarity_to_seeds(t, neg_seeds)
            score = 2 * (pos_sim - neg_sim) / (pos_sim + neg_sim)
            if func(score):
                senti_terms.append(t)

    # load GloVe vectors
    print("\t[Annotate] loading GloVe {}-d".format(args.glove_dimension), end=" ")
    glove = glove = KeyedVectors.load_word2vec_format(
        "./glove/glove.6B.{}d.word2vec_format.txt".format(args.glove_dimension))
    print("Done!")
    
    pos_senti_terms, neg_senti_terms = [], []
    all_words = list(df['word'])

    head_iter = iter(all_words)
    bott_iter = iter(all_words[::-1])

    # mine positive tokens
    print("\t[Annotate] getting terms in positive & negative polarity ...")
    get_terms_in_polarity(head_iter, pos_senti_terms, lambda x: x > 0.0)  # positive
    get_terms_in_polarity(bott_iter, neg_senti_terms, lambda x: x < 0.0)  # negative

    return pos_senti_terms, neg_senti_terms


def load_sdrn_sentiment_annotation_terms(args):
    """load SDRN sentiment annotation terms

    Args:
        args.sdrn_anno_path - the path of SDRN annotation text.
    Return:
        sdrn_senti_words - [Set] of SDRN annotated sentiment words.
    """
    if not os.path.exists(args.sdrn_anno_path):
        raise FileNotFoundError(
            "SDRN annotation file NOT found here: {}".format(args.sdrn_anno_path))
    anno_file = args.sdrn_anno_path
    sdrn_senti_words = set()
    with open(anno_file, "r") as fin:
        for i, line in enumerate(fin.readlines()):
            if len(line) > 1:
                token, tag = line.strip().split("\t")
                if tag in ['B-P', 'I-P']:
                    sdrn_senti_words.add(token)
    
    return sdrn_senti_words


def load_senti_wordlist_terms(args):
    """load Sentiment words from Bing Liu's knowledge base.

    Returns:
        senti_wl_terms - [Set] sentiment word list of terms
    """
    pos_words_path = "./configs/opinion-lexicon-English/positive-words.txt"
    neg_words_path = "./configs/opinion-lexicon-English/negative-words.txt"
    if not os.path.exists(pos_words_path):
        raise FileNotFoundError("positive-words file not found")
    if not os.path.exists(neg_words_path):
        raise FileNotFoundError("negative-words file not found")

    senti_wl_terms = []
    for wl_path in [pos_words_path, neg_words_path]:
        with open(wl_path) as fin:
            terms = fin.readlines()
            terms = [term.strip() for term in terms if term[0].isalpha()]
            senti_wl_terms += terms
    senti_wl_terms = set(senti_wl_terms)
    return senti_wl_terms


def get_sentiment_terms(args):
    """Mine Sentiment words by PMI, SDRN, SentiWordTable

    Args:
        args - the input with multiple arguments
    Returns:
        pmi, sdrn terms - if args.use_senti_word_list is False
        pmi, sdrn, senti_wordlist terms - if args.use_senti_word_list is True
    """

    # ===============================
    #   Mine Sentiment words by PMI
    # ===============================

    # check args.path
    if not os.path.exists(args.path):
        raise ValueError("Invalid path {}".format(args.path))
    if args.path[-1] == "\/":
        args.path = args.path[:-1]

    # load config files
    pmi_seeds = load_pmi_seed_words()
    train_df = load_train_file(path=args.path)
    postag_filters = load_postag_filters(args)

    # compute PMI and POS tag
    pmi_matrix, pmi_vocab = compute_pmi(args, train_df)
    word_to_postag = get_vocab_postags(args, train_df, pmi_vocab)

    # generate opinion words
    word_pol_df = compute_vocab_polarity_from_seeds(
        args,
        seeds=pmi_seeds,
        postag_filters=postag_filters['keep'],
        vocab_postags=word_to_postag,
        pmi_matrix=pmi_matrix)
    
    # get pmi sentiment word terms
    pos_senti_terms, neg_senti_terms = filter_senti_terms_by_glove(args, word_pol_df)
    pmi_senti_terms = set(pos_senti_terms).union(set(neg_senti_terms))

    # ===============================
    #   Parse SDRN output
    # ===============================
    sdrn_senti_terms = load_sdrn_sentiment_annotation_terms(args)

    # if choose not to use sentiment word list, exit here
    if not args.use_senti_word_list:
        return pmi_senti_terms, sdrn_senti_terms

    # ===============================
    #   (Optional) SentiWordTable
    # ===============================

    senti_wl_terms = load_senti_wordlist_terms(args)
    return pmi_senti_terms, sdrn_senti_terms, senti_wl_terms


def get_aspect_senti_pairs(args, senti_term_set):
    """generate (aspect, sentiment) pairs
    Args:
        args - the input arguments
        senti_term_set - the set of sentiment terms to use
    Returns:
        as_pair_set - Aspect-Sentiment pair set
    
    Note:
    1.  In this section, we are using the nlp.pipe api in spaCy to process the strings
        in batches because of its superior efficiency. Default spaCy pipeline involves 
        tokenizer, pos tagger, dependency parser, and name entity recognizer. 
        Therefore, we disable `tagger` and `ner`.
    2.  In this function, we use `to_pickle` to save dataframe as pickle since the parsed
        Doc list will be used later. We don't want to waste that time again. Just in case
        Pickle wouldn't be able to handle the size, we can always switch to HDF5 by using
        `to_hdf()`.
    """ 

    def process(s):
        """helper func: sentiment tokenizer, dependency parser by batches
        Arg:
            s - string of un-sentence-tokenized reviews.
        Return:
            generator of Doc objects of processed review 
        """
        sentences = sent_tokenize(s)
        # gen = nlp.pipe(sentences, disable=["tagger", "ner"])
        doc_list = [_ for _ in nlp.pipe(sentences, disable=['tagger', 'ner'])]
        selected_dep_rel = set([amod]) 
        for doc in doc_list:
            for tk in doc:
                if tk.dep in selected_dep_rel and tk.lower_ in senti_term_set:
                    # tk.head [spacy Token], tk.head.text [str]
                    as_pair_set.add((tk.head.text, tk.text))
        return doc_list
    
    def process_complex(s):
        """helper func. Same as `process` but implements more complex extraction method
        for aspect and sentiments"""
        sentences = sent_tokenize(s)
        doc_list = [_ for _ in nlp.pipe(sentences, disable=['tagger', 'ner'])]
        # TODO: finish here!


    # load training data
    train_df = load_train_file(path=args.path)

    # processing original text
    nlp = spacy.load("en_core_web_sm")
    as_pair_set = set()  # will be modified in the `progress_apply`.
    if not args.multi_proc_dep_parsing:
        tqdm.pandas()  # use tqdm pandas to enable progress bar for `progress_apply`.
        train_df['dep_review_Doc_list'] = train_df.original_text.progress_apply(process)
    else:
        print("\t[Annotate] Using parallel dep parsing." + 
              "Spinning off {} parallel workers ...".format(args.num_workers_mp_dep))
        pandarallel.initialize(
            nb_workers=args.num_workers_mp_dep,
            progress_bar=True,
            verbose=1)
        train_df['dep_review_Doc_list'] = train_df.original_text.parallel_apply(process)

    # save the processed train_df for later use.
    train_df.to_pickle(args.path + "/train_data_dep.pkl")

    return as_pair_set


def main(args):
    print("[Annotate] getting sentiment terms ...")
    # term_sets = get_sentiment_terms(args)  # uncomment afterward

    # ==== temp code blow =====

    # dump_pkl("temp/term_set.pkl", term_sets)

    term_sets = load_pkl("temp/term_set.pkl")
    term_sets = list(term_sets)
    term_sets[0] = set(term_sets[0])
    
    sdrn_set = load_sdrn_sentiment_annotation_terms(args)
    term_sets[1] = sdrn_set

    print(type(term_sets))
    print(len(term_sets))
    print([type(x) for x in term_sets])
    print("[Annotate] unioning {} sets ...".format(len(term_sets)))

    term_set = set.union(*term_sets)  # merge all term sets, 2 or 3
    dump_pkl("temp/term_set.pkl", term_set)

    # ===== temp code above ====

    print("[Annotate] getting aspect sentiment pairs ...")
    # as_pairs = get_aspect_senti_pairs(args, term_set)

    # save things
    print("[Annotate] saving extracted aspect sentiment paris ...")
    # dump_pkl(path=args.path + "/as_pairs.pkl", obj=as_pairs)

    # write as pairs to file
    print("[Annotate] output aspects to temp folder ...")
    # for 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the dataset.")

    parser.add_argument(
        "--sdrn_anno_path",
        type=str,
        required=True,
        help="Path to SDRN annotation results")

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
        "--num_senti_terms_per_pol",
        type=int,
        default=300,
        help="Number of sentiment terms per seed. Default=300.")
    
    parser.add_argument(
        "--use_senti_word_list",
        action="store_true",
        help="If used, sentiment word table will be used as well.")

    parser.add_argument(
        "--glove_dimension",
        type=int,
        default=100,
        help="The dimension of glove to use in the PMI parsing. Default=100.")
    
    parser.add_argument(
        "--multi_proc_dep_parsing",
        action="store_true",
        default=False,
        help="If used, parallel processing of dependency parsing will be enabled.")
    
    parser.add_argument(
        "--num_workers_mp_dep",
        type=int,
        default=8,
        help="Number of workers to be spinned off for multiproc dep parsing.")

    args = parser.parse_args()
    main(args)
