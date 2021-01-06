import pandas as pd
import sys

from extract import COL_AS_PAIRS_IDX

dataset_names = [
    "office_products",
    "home_kitchen",
    "video_games",
    "pet_supplies"]

# dataset_names = [
#     "digital_music"]


for dataset in dataset_names:
    print("dataset {}".format(dataset))
    df = pd.read_pickle("./data/amazon/{}/train_data_aspairs.pkl".format(dataset))

    # compute #aspairs/r
    aspair_count = df[COL_AS_PAIRS_IDX].apply(len)
    print("# aspair/r = {}".format(sum(aspair_count)/df.shape[0]))

    def count_unique(x):
        x = x[COL_AS_PAIRS_IDX]
        flat = [x__ for x_ in x for x__ in x_]
        try:
            num_a = set([y[0] for y in flat])
        except:
            print(flat)
            sys.exit()
            
        return len(num_a)

    for gb_item in ['user_id', 'item_id']:
        df_gb = df.groupby(gb_item)[COL_AS_PAIRS_IDX].agg(list).reset_index()
        df_gb['n_unique_a'] =  df_gb.apply(count_unique, axis=1, result_type="expand")
        print("#a/{} {}".format(gb_item, sum(df_gb['n_unique_a'])/df_gb.shape[0]))