'''
    The file for dataloader.

    File name: dataloader.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/04/2020
    Date Last Modified: TODO
    Python Version: 3.6
'''

from random import shuffle
import pandas as pd

from utils import load_pkl

class ReviewInstance:
    def __init__(self, uid, iid, rtn):
        self.user_id = uid
        self.item_id = iid
        self.rating  = rtn

class DataLoader:
    
    def __init__(self,
                 dataset,
                 shuffle=True,
                 batch_size=128
                 ):
        """initialize DataLoader
        
        Args:
            dataset - the amazon dataset to use load data
            shaffle - whether to shaffle input training data for each epoch
            batch_size - the batch size of input data
        """
        # initialize dataloader
        self.ds = dataset
        self.shuffle_train = shuffle
        self.bs = batch_size
        self.recurrent_iteration = False

        self.user_rev, self.item_rev = None, None
        self.user_nbr, self.item_nbr = None, None
        self.__load_info_data()
        
        self.train_data = None
        self.train_instances = None
        self.__load_train_data()

    
    def __load_user_item_reviews(self):
        """[private] load aggregated reviews for user/item
        
        Returns:
            user_annotated_reviews: [dict] {uid: [list of AnnotatedReview]}
            item_annotated_reviews: [dict] {iid: [list of AnnotatedReview]}
        """
        user_review_path = "./data/amazon/{}/user_anno_reviews.pkl".format(self.ds)
        item_review_path = "./data/amazon/{}/item_anno_reviews.pkl".format(self.ds)
        return load_pkl(user_review_path), load_pkl(item_review_path)

    def __load_user_item_nbr(self):
        """[private] load neighbors for user and item"""
        user_nbr_path = "./data/amazon/{}/user_nbr_item.pkl".format(self.ds)
        item_nbr_path = "./data/amazon/{}/item_nbr_user.pkl".format(self.ds)
        return load_pkl(user_nbr_path), load_pkl(item_nbr_path)
    
    def __load_info_data(self):
        """load following information for user/item:
             review, neighbors
        """
        print("[DataLoader] initialize, load datasets ...")
        
        print("[DataLoader] loading user/item nbrs ...")
        self.user_nbr, self.item_nbr = self.__load_user_item_nbr()

        print("[DataLoader] loading user/item reviews ...")
        self.user_rev, self.item_rev = self.__load_user_item_reviews()

        print("[DataLoader] data loading done!")
    
    def __load_train_data(self):
        """load training data (aka <user, item, rating> triplets) 
           and convert them into ReviewInstance objects"""

        def load_instance(row):
            return ReviewInstance(
                uid=row['user_id'], iid=row['item_id'], rtn=row['rating'])

        print("[DataLoader] initialize, load train datasets ...")
        train_data_path = "./data/amazon/{}/train_data.csv.".format(self.ds)
        self.train_data = pd.read_csv(train_data_path)

        # series to list
        self.train_instances = self.train_data.apply(load_instance, axis=1)
        self.train_instances = list(self.train_instances)

    def get_train_batch(self):
        """Batch getter
        Note: didn't do sampling here to a fixed size for user/item nbrs
        """

        if self.recurrent_iteration:
            self.__shuffle_train_data()
            self.recurrent_iteration = True

        bs = self.bs
        tail_batch = 1 if len(self.train_instances) % bs else 0
        total_num_batches = len(self.train_instances) // bs + tail_batch

        for i in range(total_num_batches):
            end_idx = min(len(self.train_instances), (i+1) * bs)
            instances = self.train_instances[i * bs, end_idx]
            
            # get user/item ids
            users = [ins.user_id for ins in instances]
            items = [ins.item_id for ins in instances]

            # get user/item neighbors
            user_nbrs = [self.user_nbr[ins.user_id] for ins in instances]
            item_nbrs = [self.item_nbr[ins.item_id] for ins in instances]

            # get user/item reviews
            user_revs = [self.user_rev[ins.user_id] for ins in instances]
            item_revs = [self.item_rev[ins.item_id] for ins in instances]

            yield {"user_ids": users, "item_ids": items,
                   "user_nbr": user_nbrs, "item_nbr": item_nbrs,
                   "user_revs": user_revs, "item_revs": item_revs}
    
    def __shuffle_train_data(self):
        if self.shuffle_train:
            shuffle(self.train_instances)