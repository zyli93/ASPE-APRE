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
from transformers import BertModel

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
                 use_nbr=False,
                 use_gpu=True,
                 max_sample_num=30
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
        self.max_sample_num = max_sample_num
        self.use_gpu = use_gpu

        self.user_rev, self.item_rev = None, None
        self.user_nbr, self.item_nbr = None, None
        self.__load_info_data()

        self.train_data = None
        self.train_instances = None
        self.train_batch_num, self.test_batch_num = 0, 0
        self.__load_traintest_data()

        self.bert = None
        self.__load_bert()

        self.user_rev_enc = {}
        self.item_rev_enc = {}
    
    def __load_bert(self):
        self.bert = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
        self.bert.requires_grad_(False)
        
        if self.use_gpu:
            self.bert = self.bert.cuda()

    
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

    def get_batch_iterator(self, epoch=0, for_train=True):
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
            
            # get user/item ids
            users_list = [int(ins.user_id[2:]) for ins in instances]
            items_list = [int(ins.item_id[2:]) for ins in instances]
            ratings = np.array([ins.rating for ins in instances])

            # change to tensor
            users = torch.from_numpy(np.array(users_list))
            items = torch.from_numpy(np.array(items_list))
            ratings = torch.from_numpy(ratings)

            # get user/item reviews
            user_revs = [self.user_rev[ins.user_id] for ins in instances]
            item_revs = [self.item_rev[ins.item_id] for ins in instances]

            u_split = [self.__get_sample_size(x.get_rev_size()) for x in user_revs]
            i_split = [self.__get_sample_size(x.get_rev_size()) for x in item_revs]
            
            urevs_loc = []
            urevs_out_hid_list = []
            urevs_pooler_list = []
            for j, x in enumerate(user_revs):
                input_, loc = x.get_anno_tkn_revs()
                tmp_u_input_ids, tmp_u_attn_mask, tmp_u_loc = self.__get_sampled_reviews(
                    input_['input_ids'], input_['attention_mask'], loc)

                if users_list[j] not in self.user_rev_enc:
                    if self.use_gpu:
                        tmp_u_input_ids = tmp_u_input_ids.cuda()
                        tmp_u_attn_mask = tmp_u_attn_mask.cuda()
                    tmp_u_enc = self.bert(tmp_u_input_ids, tmp_u_attn_mask)
                    if self.ds != "automotive":
                        tmp_u_enc_pool = tmp_u_enc[1].cpu()
                        tmp_u_enc_outhid = tmp_u_enc[0].cpu()
                    else:
                        tmp_u_enc_pool = tmp_u_enc[1]
                        tmp_u_enc_outhid = tmp_u_enc[0]
                    self.user_rev_enc[users_list[j]] = {
                        "pooler": tmp_u_enc_pool, "out_hid": tmp_u_enc_outhid}
                    urevs_out_hid_list.append(tmp_u_enc_outhid)
                    urevs_pooler_list.append(tmp_u_enc_pool)
                else:
                    urevs_out_hid_list.append(
                        self.user_rev_enc[users_list[j]]['out_hid'])
                    urevs_pooler_list.append(
                        self.user_rev_enc[users_list[j]]['pooler'])
                urevs_loc.append(tmp_u_loc)
            

            urevs_out_hid = torch.cat(urevs_out_hid_list, dim=0)
            urevs_pooler = torch.cat(urevs_pooler_list, dim=0)
            urevs_loc = torch.as_tensor(np.concatenate(
                [x.toarray() for _rev in urevs_loc for x in _rev], axis=0), dtype=torch.float)

            irevs_out_hid_list = []
            irevs_pooler_list = []
            irevs_loc = []
            for j, x in enumerate(item_revs):
                input_, loc = x.get_anno_tkn_revs()
                tmp_i_input_ids, tmp_i_attn_mask, tmp_i_loc = self.__get_sampled_reviews(
                    input_['input_ids'], input_['attention_mask'], loc)
                if items_list[j] not in self.item_rev_enc:
                    if self.use_gpu:
                        tmp_i_input_ids = tmp_i_input_ids.cuda()
                        tmp_i_attn_mask = tmp_i_attn_mask.cuda()
                    tmp_i_enc = self.bert(tmp_i_input_ids, tmp_i_attn_mask)
                    if self.ds != "automotive":
                        tmp_i_enc_pool = tmp_i_enc[1].cpu()
                        tmp_i_enc_outhid = tmp_i_enc[0].cpu()
                    else:
                        tmp_i_enc_pool = tmp_i_enc[1]
                        tmp_i_enc_outhid = tmp_i_enc[0]
                    self.item_rev_enc[items_list[j]] = {
                        "pooler": tmp_i_enc_pool, "out_hid": tmp_i_enc_outhid}
                    irevs_out_hid_list.append(tmp_i_enc_outhid)
                    irevs_pooler_list.append(tmp_i_enc_pool)
                else:
                    irevs_out_hid_list.append(
                        self.item_rev_enc[items_list[j]]['out_hid'])
                    irevs_pooler_list.append(
                        self.item_rev_enc[items_list[j]]['pooler'])
                irevs_loc.append(tmp_i_loc)

            irevs_out_hid = torch.cat(irevs_out_hid_list, dim=0)
            irevs_pooler = torch.cat(irevs_pooler_list, dim=0)
            irevs_loc = torch.as_tensor(np.concatenate(
                [x.toarray() for _rev in irevs_loc for x in _rev], axis=0), dtype=torch.float)
            
            # u_pooler, i_pooler, u_out_hid, i_out_hid = None, None, None, None
            # if epoch > 0:
            #     u_pooler_list = [self.user_rev_enc[id_]['pooler'] for id_ in users_list]
            #     u_pooler = torch.cat(u_pooler_list, dim=0)
            #     i_pooler_list = [self.item_rev_enc[id_]['pooler'] for id_ in items_list]
            #     i_pooler = torch.cat(i_pooler_list, dim=0)

            #     u_out_hid_list = [self.user_rev_enc[id_]['out_hid'] for id_ in users_list]
            #     u_out_hid = torch.cat(u_out_hid_list, 0)
            #     i_out_hid_list = [self.item_rev_enc[id_]['out_hid'] for id_ in items_list]
            #     i_out_hid = torch.cat(i_out_hid_list, 0)
            
            batch = {"uid_list": users_list,
                     "iid_list": items_list,
                     "uid": users, 
                     "iid": items,
                     "rtg": ratings,
                     "urevs_loc": urevs_loc,
                     "irevs_loc": irevs_loc,
                     "u_split": u_split,
                     "i_split": i_split,
                     "u_pooler": urevs_pooler,
                     "i_pooler": irevs_pooler,
                     "u_out_hid": urevs_out_hid,
                     "i_out_hid": irevs_out_hid
            }
            
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
    

    def __get_sample_size(self, size):
        return size if size < self.max_sample_num else self.max_sample_num
    

    def __get_sampled_reviews(self, input_ids, attn_mask, loc):
        size = input_ids.shape[0]
        if size > self.max_sample_num:
            indices = np.random.choice(range(size), self.max_sample_num, True)
            input_ids = input_ids[indices]  # (sample_size, pad_len)
            attn_mask = attn_mask[indices]  # (sample_size, pad_len)
            new_loc = [loc[i] for i in indices]
            loc = new_loc
            
        return input_ids, attn_mask, loc
    

    def update_review_enc(self, id_list, pooler_list, out_hid_list, for_user=True):
        """
        id_list - 1-D numpy array of ID lists
        pooler_list - list of tensor of pooler state
        out_hid_list - output_hidden_state
        """
        dict_to_update = self.user_rev_enc if for_user else self.item_rev_enc
        
        for idx, id_ in enumerate(id_list):
            if id_ not in dict_to_update:
                dict_to_update[id_] = {"pooler": pooler_list[idx],
                                       "out_hid": out_hid_list[idx]}
    
    def get_cached_count(self):
        return len(self.user_rev_enc), len(self.item_rev_enc)
    