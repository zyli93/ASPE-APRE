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
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from pandarallel import pandarallel

def load_df(name):
    return pd.read_csv("./data/amazon/{}/train_data.csv".format(name)), \
        pd.read_csv("./data/amazon/{}/test_data.csv".format(name))

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

def review_per_item(df):
    return df.groupby('item_id')['user_id'].count()

def sent_length(df):
    def sent_len(text):
        sents = sent_tokenize(text)
        return [len(x) for x in sents]
    df['sents_len'] = df.original_text.parallel_apply(sent_len)
    lens = [n for subarr in df['sent_len'].tolist() for n in subarr]
    return df, lens


if __name__ == "__main__":
    # names = ["video_games", "sports_outdoors", "office_products", 
            #  "home_kitchen", "pet_supplies"]
    names = ["automotive", "sports_outdoors", "digital_music", 
             "toys_games", "pet_supplies"]
    if False:
        for name in names:
            pandarallel.initialize(
                nb_workers=16,
                progress_bar=True,
                verbose=1)
            print("loading {}".format(name))
            df, _ = load_df(name)

            n_u, n_t = df.user_id.nunique(), df.item_id.nunique()
            n_r = df.shape[0]

            print("review per user")
            rpu = review_per_user(df)
            rpu = rpu.sum() / n_u

            print("review per item")
            rpt = review_per_item(df)
            rpt = rpt.sum() / n_t

            print("word per view")
            df = word_per_review(df)
    
            print("sent per review")
            df = sent_per_review(df)

            total_words = df.word_count.sum()

            print("{} {} & {} & {} & {} & {} & {} & {} & {}".format(
                name, n_r, n_u, n_t, n_r/(n_u*n_t), 
                total_words, rpu, rpt, df.word_count.mean()
            ))
    elif False:
        for name in names:
            print(name)
            df, test_df = load_df(name)
            # check if covered:
            trn_user = set(df['user_id'].unique())
            tst_user = set(test_df['user_id'].unique())

            if len(trn_user.intersection(tst_user)) - len(tst_user) != 0:
                print(len(trn_user.intersection(tst_user)) - len(tst_user))
                print("Oh no for user!")

            # check if covered:
            trn_item = set(df['item_id'].unique())
            tst_item = set(test_df['item_id'].unique())

            if len(trn_item.intersection(tst_item)) - len(tst_item) != 0:
                print(len(trn_item.intersection(tst_item)) - len(tst_item))
                print("Oh no for item!")


            print("num user {}, num_item {}".format(len(trn_user), len(trn_item)))
    elif False:
        name = "digital_music"
        df, test_df = load_df(name)
        print(df['user_id'].nunique())
        print(test_df['user_id'].nunique())
        trn_user_ser = df['user_id'].apply(lambda x: int(x[2:]))
        print(trn_user_ser.max(), trn_user_ser.min())

        print(df['item_id'].nunique())
        print(test_df['item_id'].nunique())
        trn_item_ser = df['item_id'].apply(lambda x: int(x[2:]))
        print(trn_item_ser.max(), trn_item_ser.min())
    
    elif True:
        for name in names:
            print(name)
            df, test_df = load_df(name)
            rpu = review_per_user(df)
            print(list(range(50, 101, 5)))
            print(np.percentile(rpu.to_numpy(), range(50, 101, 5)))
            rpt = review_per_item(df)
            print(list(range(50, 101, 5)))
            print(np.percentile(rpt.to_numpy(), range(50, 101, 5)))
            print("")
    else:
        for name in names:
            df, test_df = load_df(name)
            print("test dataframe, before shape {}".format(test_df.shape))
            trn_user = set(df['user_id'].unique())
            trn_item = set(df['item_id'].unique())

            test_df = test_df[test_df.user_id.isin(trn_user)]
            test_df = test_df[test_df.item_id.isin(trn_item)]
            print("test dataframe, after shape {}".format(test_df.shape))

            test_df.to_csv("./data/amazon/{}/test_data_new.csv".format(name), 
                index=False)
            print("processing {} done!".format(name))

