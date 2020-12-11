'''
    Get the statistics of datasets.

    File name: data_statistics.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/10/2020
    Date Last Modified: TODO
    Python Version: 3.6
'''
from scipy.stats import describe
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from pandarallel import pandarallel

def load_df(name):
    return pd.read_csv("./data/amazon/{}/train_data.csv".format(name))

def word_per_review(df):
    def word_count(text):
        return len(word_tokenize(text))
    df["word_count"] = df.original_text.parallel_apply(word_count)
    return df

def sent_per_review(df):
    def sent_count(text):
        return len(sent_tokenize(text))
    df['sent_count'] = df.original_text.parallel_apply(sent_count)
    return df

def review_per_user(df):
    return df.groupby('user_id')['item_id'].count()

def sent_length(df):
    def sent_len(text):
        sents = sent_tokenize(text)
        return [len(x) for x in sents]
    df['sents_len'] = df.original_text.parallel_apply(sent_len)
    lens = [n for subarr in df['sent_len'].tolist() for n in subarr]
    return df, lens


if __name__ == "__main__":
    names = ["video_games", "sports_outdoors", "office_products", 
             "home_kitchen", "pet_supplies"]
    for name in names:
        pandarallel.initialize(
            nb_workers=16,
            progress_bar=True,
            verbose=1)
        print("loading {}".format(name))
        df = load_df(name)

        print("review per user")
        rpu = review_per_user(df)

        print("word per view")
        df = word_per_review(df)
 
        print("sent per review")
        df = sent_per_review(df)

        print("Dataset {}".format(name))
        total_words = df.word_count.sum()
        print("total words = {}".format(total_words))
        print("review/user = {}".format(rpu.sum() / len(rpu)))
        print("words/review = {}".format(df.word_count.mean()))
        print("sents/review = {}".format(df.sent_count.mean()))
