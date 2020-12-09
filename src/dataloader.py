'''
    The file for dataloader.

    File name: dataloader.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/04/2020
    Date Last Modified: TODO
    Python Version: 3.6
'''

from utils import load_pkl
from postprocess import AnnotatedReview

class DataLoader:
    
    def __init__(self,
                 dataset,
                 ):
        """initialize DataLoader
        
        Args:
            dataset - the amazon dataset to use load data
        """
        # initialize dataloader
        self.ds = dataset

        pass

    
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