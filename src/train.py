'''
    The trainer's file.

    File name: train.py
    Author: Zeyu Li 
    Email: <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
    Date Created: 12/07/2020
    Date Last Modified: TODO
    Python Version: 3.6
'''

import os
import sys
import argparse
import logging

from dataloader import DataLoader
from model import APRE

"""
don't forget to use model.train() and model.eval()
"""

# Arguments parser
parser = argparse.ArgumentParser()

# input and training configuration
parser.add_argument("--dataset", type=str, required=True, help="The dataset.")
parser.add_argument("--shuffle", action="store_true", default=False,
                    help="Whether to shuffle data before a new epoch")
parser.add_argument("--batch_size", type=int, default=128,
                    help="The batch size of each iteration")
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument("--num_epoch", type=int, default=20, help="Number of epoches")

# experiment configuration
parser.add_argument("--experimentID", type=str, required=True,
                    help="The ID of the experiments")
parser.add_argument("--log_iter_num", type=int, required=True, default=1000,
                    help="Number of iterations to write status log.")

# model configuration

# save model configeration
parser.add_argument("--save_model", action="store_true", default=False,
                    help="Whether to turn on model saving.")
parser.add_argument("--save_model_iter_num", type=int, default=1000,
                    help="Number of iterations per model saving." + 
                         "Only in effect when `save_model` is turned on.")

args = parser.parse_args()



def train(model, dataloader):

    for ep in args.num_epoch:
        break


if __name__ == "__main__":

    # config logging
    logging.basicConfig(filename='./log/{}.log'.format(args.experimentID), 
        filemode='w', level=logging.DEBUG, format='[%(asctime)s] %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info("shit")

    # model = APRE()  # TODO

    # dataloader = DataLoader(dataset=args.dataset, 
    #                         shuffle=args.shuffle,
    #                         batch_size=args.batch_size)
    
    # train(model, dataloader)
    # pass
