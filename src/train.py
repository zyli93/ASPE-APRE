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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
parser.add_argument("--learning_rate", type=float, default=0.001, 
                    help="Learning rate")
parser.add_argument("--num_epoch", type=int, default=20, help="Number of epoches")
parser.add_argument("--opimizer", type=str, help="Selection of optimizer")
parser.add_argument("--load_model", action="store_true", default=False,
                    help="Whether to resume training from an exisint ckpt")
parser.add_argument("--load_model_path", type=str, help="Path to load existing model")

# experiment configuration
parser.add_argument("--experimentID", type=str, required=True,
                    help="The ID of the experiments")
parser.add_argument("--log_iter_num", type=int, default=1000,
                    help="Number of iterations to write status log.")
parser.add_argument("--eval_iter_num", type=int, default=1000,
                    help="Number of iterations to evaluate current model")

# model configuration
parser.add_argument("--disable_explicit", action="store_true", default=False,
                    help="Flag to disable the explicit channel")
parser.add_argument("--disable_implicit", action="store_true", default=False,
                    help="Flag to disable the implicit channel")
parser.add_argument("--disable_cf", action="store_true", default=False,
                    help="Flag to disable the CF channel")
parser.add_argument("--num_aspects", type=int, required=True,
                    help="Number of close-domain aspects in total")

# save model configeration
parser.add_argument("--save_model", action="store_true", default=False,
                    help="Whether to turn on model saving.")
parser.add_argument("--save_model_iter_num", type=int, default=1000,
                    help="Number of iterations per model saving." + 
                         "Only in effect when `save_model` is turned on.")
parser.add_argument("--save_model_path", type=str, default="./ckpt/",
                    help="Path to directory to save models")

# evaluation configeration
parser.add_argument("--eval_as_cls", action="store_true", default=False,
                    help="Flag to round pred and eval as a classification task.")


args = parser.parse_args()


def get_optimizer(model):
    lr = args.learning_rate
    params = model.parameters()
    if args.opimizer == "adam":
        return optim.Adam(params, lr=lr)
    elif args.opimizer == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    else:
        raise ValueError("`learning rate` supported: adam/rmsprop")


def train(model, dataloader):

    # optimizer
    optimizer = get_optimizer(model)

    # loss function
    criterion = nn.MSELoss()

    # model training flag
    model.train()

    for ep in args.num_epoch:
        trn_iter = dataloader.get_train_batch_iterator()
        
        for batch in trn_iter:
            idx = batch.batch_idx
            # TODO: change everything to good shape
            pred = model(batch)
            target = batch.rtg # TODO: make it a torch.Tensor

            loss = criterion(pred, target)

            # optimization
            loss.backward()
            optimizer.step()

            # save model
            if args.save_model and idx and not idx % args.save_model_iter_num:
                # TODO save model
                # TODO: save path is known is arguments
                # TODO: add log for that
                pass
                
            # log model
            if idx and not idx % args.log_iter_num:
                # TODO: write log
                pass
                
            # run test
            if idx and not idx % args.eval_iter_num:
                # TODO: run test set
                # TODO: write log
                pass


if __name__ == "__main__":

    # config logging
    logging.basicConfig(filename='./log/{}.log'.format(args.experimentID), 
        filemode='w', level=logging.DEBUG, 
        format='[%(asctime)s][%(levelname)s][%(filename)s] %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S')
    
    # set random seeds

    # model saver
    
    # model = APRE(args, )  # TODO

    # dataloader = DataLoader(dataset=args.dataset, 
    #                         shuffle=args.shuffle,
    #                         batch_size=args.batch_size)
    
    # train(model, dataloader)
    # pass
