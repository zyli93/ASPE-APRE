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
import numpy as np
from easydict import EasyDict as edict

import torch

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
                 batch_size=128,
                 use_nbr=False
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
        self.use_neighbor = use_nbr

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
            user_annotated_reviews: [dict] {uid: EntityReviewAggregation}
            item_annotated_reviews: [dict] {iid: EntityReviewAggregation}
        """
        user_review_path = "./data/amazon/{}/user_anno_tkn_revs.pkl".format(self.ds)
        item_review_path = "./data/amazon/{}/item_anno_tkn_revs.pkl".format(self.ds)
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
        
        # TODO: neighbor not showing up on cuda might be another problem
        if self.use_neighbor:
            print("[DataLoader] loading user/item nbrs ...")
            self.user_nbr, self.item_nbr = self.__load_user_item_nbr()

        print("[DataLoader] loading user/item reviews ...")
        self.user_rev, self.item_rev = self.__load_user_item_reviews()

        print("[DataLoader] info data loading done!")
    
    def __load_traintest_data(self):
        """load training/test data (aka <user, item, rating> triplets) 
           and convert them into RateInstance objects"""

        def load_instance(row):
            return RateInstance(uid=row['user_id'],  
                iid=row['item_id'], rtg=row['rating'])

        print("[DataLoader] initialize, load train datasets ...")
        train_data_path = "./data/amazon/{}/train_data.csv".format(self.ds)
        self.train_data = pd.read_csv(train_data_path)

        print("[DataLoader] initialize, process train datset ...")
        self.train_instances = self.train_data.apply(load_instance, axis=1)
        self.train_instances = list(self.train_instances)

        tail_batch = 1 if len(self.train_instances) % self.bs else 0
        self.train_batch_num = len(self.train_instances) // self.bs + tail_batch

        print("[DataLoader] initialize, load test datasets ...")
        test_data_path = "./data/amazon/{}/test_data_new.csv".format(self.ds)
        self.test_data = pd.read_csv(test_data_path)

        print("[DataLoader] initialize, process train datset ...")
        self.test_instances = self.test_data.apply(load_instance, axis=1)
        self.test_instances  = list(self.test_instances)

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
            instances = data_instances[i * bs: end_idx]
            
            print([ins.user_id[2:] for ins in instances])  # DEBUG 
            # get user/item ids
            users = np.array([int(ins.user_id[2:]) for ins in instances])
            items = np.array([int(ins.item_id[2:]) for ins in instances])
            ratings = np.array([ins.rating for ins in instances])

            # change to tensor
            users = torch.from_numpy(users)
            items = torch.from_numpy(items)
            ratings = torch.from_numpy(ratings)

            # get user/item neighbors
            if self.use_neighbor:
                user_nbrs = [self.user_nbr[ins.user_id] for ins in instances]
                item_nbrs = [self.item_nbr[ins.item_id] for ins in instances]
            else:
                user_nbrs, item_nbrs = None, None

            # get user/item reviews
            user_revs = [self.user_rev[ins.user_id] for ins in instances]
            item_revs = [self.item_rev[ins.item_id] for ins in instances]

            # === new below ===
            # batch.urev, batch.irev list of EntityReviewAggregation
            u_split = torch.tensor([x.get_rev_size() for x in user_revs], dtype=torch.int)
            i_split = torch.tensor([x.get_rev_size() for x in item_revs], dtype=torch.int)

            # contextualized encoder (bert) input_ids
            urevs_input_ids = [x.get_anno_tkn_revs()[0]['input_ids'] for x in user_revs] # list of (num_revs*pad_len)
            irevs_input_ids = [x.get_anno_tkn_revs()[0]['input_ids'] for x in item_revs] # list of (num_revs*pad_len)
            urevs_input_ids = torch.cat(urevs_input_ids, dim=0) # ttl_u_n_rev, padlen
            irevs_input_ids = torch.cat(irevs_input_ids, dim=0) # ttl_i_n_rev, padlen
            
            # contextualized encoder (bert) attention masks
            urevs_attn_mask = [x.get_anno_tkn_revs()[0]['attention_mask'] for x in user_revs] #(num_revs*pad_len)
            irevs_attn_mask = [x.get_anno_tkn_revs()[0]['attention_mask'] for x in item_revs] #(num_revs*pad_len)
            urevs_attn_mask = torch.cat(urevs_attn_mask, dim=0) # ttl_u_n_rev, padlen
            irevs_attn_mask = torch.cat(irevs_attn_mask, dim=0) # ttl_i_n_rev, padlen


            # aspect locations
            urevs_loc = torch.as_tensor(np.concatenate(
                [x.toarray() for _rev in user_revs for x in _rev.get_anno_tkn_revs()[1]], 
                axis=0), dtype=torch.float)
            irevs_loc = torch.as_tensor(np.concatenate(
                [x.toarray() for _rev in item_revs for x in _rev.get_anno_tkn_revs()[1]], 
                axis=0), dtype=torch.float)
            # === new above ===

            # move to devices
            # if self.use_gpu:
            #     self.__move_rev_to_device(user_revs)
            #     self.__move_rev_to_device(item_revs)

            batch = {"uid": users, 
                     "iid": items,
                     "rtg": ratings,
                     "urevs_input_ids": urevs_input_ids, 
                     "irevs_input_ids": irevs_input_ids,
                     "urevs_attn_mask": urevs_attn_mask,
                     "irevs_attn_mask": irevs_attn_mask,
                     "urevs_loc": urevs_loc,
                     "irevs_loc": irevs_loc,
                     "u_split": u_split,
                     "i_split": i_split
            }

            # if self.use_gpu:
            #     self.__move_rev_to_device(batch)
            
            yield batch
    
    def __shuffle_train_data(self):
        if self.shuffle_train:
            shuffle(self.train_instances)

    def get_train_batch_num(self):
        return self.train_batch_num

    def get_test_batch_num(self):
        return self.test_batch_num
    
    def __move_rev_to_device(self, batch):
        """deprecated"""
        for rev_list in [batch.urev, batch.irev]:
            for _rev in rev_list:
                tkn, loc = _rev.get_anno_tkn_revs()
                tkn['input_ids'] = tkn['input_ids'].cuda()
                tkn['attention_mask'] = tkn['attention_mask'].cuda()
                _rev.set_anno_tkn_revs(tkn)
