'''
    The file for dataloader.

    File name: dataloader.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/04/2020
    Date Last Modified: TODO
    Python Version: 3.6
'''

import logging
from random import shuffle
import pandas as pd
from easydict import EasyDict as edict

from utils import load_pkl

class RateInstance:
    def __init__(self, uid, iid, rtg):
        self.user_id = uid
        self.item_id = iid
        self.rating  = rtg

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
        self.train_batch_num, self.test_batch_num = 0, 0
        self.__load_traintest_data()
    
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

        print("[DataLoader] info data loading done!")
    
    def __load_traintest_data(self):
        """load training/test data (aka <user, item, rating> triplets) 
           and convert them into RateInstance objects"""

        def load_instance(row):
            return RateInstance(
                uid=row['user_id'], iid=row['item_id'], rtg=row['rating'])

        print("[DataLoader] initialize, load train datasets ...")
        train_data_path = "./data/amazon/{}/train_data.csv.".format(self.ds)
        self.train_data = pd.read_csv(train_data_path)

        print("[DataLoader] initialize, process train datset ...")
        self.train_instances = self.train_data.apply(load_instance, axis=1)
        self.train_instances = list(self.train_instances)

        tail_batch = 1 if len(self.train_instances) % self.bs else 0
        self.train_batch_num = len(self.train_instances) // self.bs + tail_batch

        print("[DataLoader] initialize, load test datasets ...")
        test_data_path = "./data/amazon/{}/test_data.csv.".format(self.ds)
        self.test_data = pd.read_csv(test_data_path)

        print("[DataLoader] initialize, process train datset ...")
        self.test_instances = self.test_data.apply(load_instance, axis=1)
        self.test_instances  = list(self.test_data)

        tail_batch = 1 if len(self.test_instances) % self.bs else 0
        self.test_batch_num = len(self.test_instances) // self.bs + tail_batch

        print("[DataLoader] train/test data loading done!")

    def get_batch_iterator(self, for_train=True):
        """Batch getter
        Note: didn't do sampling here to a fixed size for user/item nbrs
        """

        if self.recurrent_iteration and for_train:
            logging.info("shuffling training data")
            self.__shuffle_train_data()
            logging.info("shuffling done.")
            self.recurrent_iteration = True
        
        if for_train:
            num_batches = self.train_batch_num
            data_instances = self.train_instances
        else:
            num_batches = self.test_batch_num
            data_instances = self.test_instances

        bs = self.bs

        for i in range(num_batches):
            end_idx = min(len(data_instances), (i+1) * bs)
            instances = data_instances[i * bs, end_idx]
            
            # get user/item ids
            users = [ins.user_id for ins in instances]
            items = [ins.item_id for ins in instances]
            ratings = [ins.rating for ins in instances]

            # get user/item neighbors
            user_nbrs = [self.user_nbr[ins.user_id] for ins in instances]
            item_nbrs = [self.item_nbr[ins.item_id] for ins in instances]

            # get user/item reviews
            user_revs = [self.user_rev[ins.user_id] for ins in instances]
            item_revs = [self.item_rev[ins.item_id] for ins in instances]

            yield edict({"batch_idx": i,
                         "uid": users, "iid": items,
                         "unbr": user_nbrs, "inbr": item_nbrs,
                         "urev": user_revs, "irev": item_revs,
                         "rtg": ratings})
    
    def __shuffle_train_data(self):
        if self.shuffle_train:
            shuffle(self.train_instances)