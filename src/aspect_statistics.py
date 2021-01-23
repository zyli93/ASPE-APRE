import pandas as pd
import sys

from extract import COL_AS_PAIRS_IDX

# dataset_names = [
#     "office_products",
#     "home_kitchen",
#     "video_games",
#     "pet_supplies"]

# dataset_names = [
#     "automotive",
#     "digital_music",
#     "pet_supplies",
#     "sports_outdoors",
#     "toys_games"]

dataset_names = [
    "musical_instruments", "tools_home"]

# dataset_names = [
#     "digital_music"]


for dataset in dataset_names:
    print("dataset {}".format(dataset))
    df = pd.read_pickle("./data/amazon/{}/train_data_aspairs.pkl".format(dataset))

    # compute #aspairs/r
    aspair_count = df[COL_AS_PAIRS_IDX].apply(len)
    n_aspair_per_review = sum(aspair_count)/df.shape[0]

    def count_unique(x):
        x = x[COL_AS_PAIRS_IDX]
        flat = [x__ for x_ in x for x__ in x_]
        try:
            num_a = set([y[0] for y in flat])
            # num_s = set([y[1] for y in flat])
        except:
            print(flat)
            sys.exit()
            
        return len(num_a)

    res = []
    for gb in ['user_id', 'item_id']:
        df_gb = df.groupby(gb)[COL_AS_PAIRS_IDX].agg(list).reset_index()
        df_gb['n_unique_a'] = df_gb.apply(count_unique, axis=1, result_type="expand")
        res.append(sum(df_gb['n_unique_a'])/df_gb.shape[0])

    apu, apt = res[0], res[1]
    
    print("{} & {:.3f} & {:.3f} & {:.3f}".format(dataset, n_aspair_per_review, apu, apt))