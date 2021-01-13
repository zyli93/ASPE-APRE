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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from sklearn.metrics import mean_squared_error
from collections import deque

from dataloader import DataLoader
from model import APRE
from postprocess import EntityReviewAggregation

from utils import get_time, check_memory, print_args, make_dir

"""
TODO: don't forget to use model.train() and model.eval()
"""

# TODO: initialize Linear!!!
# TODO: whether to use dropout!!!

# Arguments parser
parser = argparse.ArgumentParser()

# input and training configuration
parser.add_argument("--task", type=str, required=True, help="Train/Both?")
parser.add_argument("--dataset", type=str, required=True, help="The dataset.")
parser.add_argument("--shuffle", action="store_true", default=False,
                    help="Whether to shuffle data before a new epoch")
parser.add_argument("--batch_size", type=int, default=128, help="The batch size of each iteration")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--num_epoch", type=int, default=20, help="Number of epoches")
parser.add_argument("--opimizer", type=str, default="adam", help="Selection of optimizer")
parser.add_argument("--load_model", action="store_true", default=False,
                    help="Whether to resume training from an existing ckpt")
parser.add_argument("--load_model_path", type=str, help="Path to load existing model")
parser.add_argument("--padded_length", type=int, default=100, help="The padded length of input tokens")

# experiment configuration
parser.add_argument("--experimentID", type=str, required=True, help="The ID of the experiments")
parser.add_argument("--log_iter_num", type=int, default=1000,
                    help="Number of iterations to write status log.")
parser.add_argument("--eval_epoch_num", type=int, default=1000, 
                    help="Number of epochs to evaluate current model")
parser.add_argument("--eval_after_epoch_num", type=int, default=5,
                    help="Start to evaluate after num of epochs")
parser.add_argument("--gpu_id", type=int, default=5, help="ID of GPU to use.")
parser.add_argument("--random_seed", type=int, default=1993, help="Random seed of PyTorch and Numpy")

# model configuration
parser.add_argument("--disable_explicit", action="store_true", default=False,
                    help="Flag to disable the explicit channel")
parser.add_argument("--disable_implicit", action="store_true", default=False,
                    help="Flag to disable the implicit channel")
parser.add_argument("--disable_cf", action="store_true", default=False, help="Flag to disable the CF channel")
parser.add_argument("--num_aspects", type=int, required=True, help="Number of close-domain aspects in total")
parser.add_argument("--aspemb_max_norm", type=int, default=-1,
                    help="Max norm of aspect embedding. Set -1 for None.")
parser.add_argument("--num_user", type=int, required=True, help="Number of users.")
parser.add_argument("--num_item", type=int, required=True, help="Number of items.")
parser.add_argument("--feat_dim", type=int, default=128, help="Intermediate layer size of the final MLP")
parser.add_argument("--regularization_weight", type=float, default=0.0001, help="Regularization weight of the model")
parser.add_argument("--num_last_layers", type=int, default=4, help="Number of last layers embedding to use in BERT.")
parser.add_argument("--ex_attn_temp", type=float, default=1.0, help="Explicit temperature of review attention weights.")
parser.add_argument("--im_attn_temp", type=float, default=1.0, help="Implicit temperature of review attention weights.")
parser.add_argument("--max_review_num", type=int, default=30, help="Maximum number of reviews to sample")


# save model configeration
parser.add_argument("--save_model", action="store_true", default=False,
                    help="Whether to turn on model saving.")
parser.add_argument("--save_epoch_num", type=int, default=2, help="Save model per x epochs.")
parser.add_argument("--save_after_epoch_num", type=int, default=1000,
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
    reg_weight = args.regularization_weight

    params = model.parameters()
    if args.opimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=reg_weight)
    elif args.opimizer == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=reg_weight)
    else:
        raise ValueError("`optimizater` supported: adam/rmsprop")


def move_batch_to_gpu(batch):
    for tensor_name, tensor in batch.items():
        if tensor_name != "u_split" and tensor_name != "i_split":
            batch[tensor_name] = tensor.cuda()
    return batch


def train(model, dataloader):

    logging.info("[info] start training ID:[{}]".format(args.experimentID))
    print("[train] started training, ID:{}".format(args.experimentID))

    # sliding avg container
    dq = deque([], maxlen=50)
    clamp_dq = deque([], maxlen=50)

    # optimizer
    optimizer = get_optimizer(model)

    # loss function
    criterion = nn.MSELoss()

    # if True, load model and continue to train
    if args.load_model:
        print("[train] loading model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
    total_num_iter_counter = 0
    
    for ep in range(args.num_epoch):
        trn_iter = dataloader.get_batch_iterator()
        total_num_iter_per_epoch = dataloader.get_train_batch_num()
        model.train()  # model training flag

        total_loss = 0
        for idx, batch in enumerate(trn_iter):
            total_num_iter_counter += 1

            if torch.cuda.is_available():
                # if model on cuda, them move batch to cuda
                batch = move_batch_to_gpu(batch)
            
            # make pred
            pred = model(batch)


            # get ground truth, convert to float tensor
            target = batch['rtg'].float()

            # clean out existing gradients
            optimizer.zero_grad()

            # build loss term
            loss = criterion(pred, target)

            clamp_loss = torch.mean((torch.clamp(pred, 1.0, 5.0) - target)**2).item()

            dq.append(loss.item())
            sliding_perf = sum(dq) / len(dq)

            clamp_dq.append(clamp_loss)
            clamp_sliding_perf = sum(clamp_dq) / len(clamp_dq)

            # optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # log model
            if idx > 0 and idx % args.log_iter_num == 0:
                msg = "ep:[{}] iter:[{}/{}] loss:[{:.4f}] RawLoss:[{:.3f}] ClampLoss:[{:.3f}]".format(
                    ep, idx, total_num_iter_per_epoch, loss.item(), sliding_perf, clamp_sliding_perf)
                logging.info("[Perf][Iter] " + msg)
                print("{} [Perf][Iter] {}".format(get_time(), msg))
            
            # HARD CODE HERE
            if loss.item() > 2. and ep > args.eval_after_epoch_num:
                pred_str = ", ".join(["{:.2f}".format(x) for x in pred])
                target_str = ", ".join(["{:.2f}".format(x) for x in target])
                logging.info("[Perf][Iter][Abnormal] [{}], [{}]".format(pred_str, target_str))

                
        # avg loss this ep
        msg = "ep:[{}] iter:[{}] avgloss:[{:.6f}]".format(
            ep, total_num_iter_counter, total_loss / total_num_iter_per_epoch)
        logging.info("[Perf][Epoch] " + msg)
        print("{} [Perf][Epoch] {}".format(get_time(), msg))
        
        # run test with three conditions:
        #   1. dataset as "both"
        #   2. ep > a certain number & ep % eval_per_epoch == 0
        if not ep % args.eval_epoch_num and args.task == "both" \
            and ep >= args.eval_after_epoch_num - 1:
            test_mse = evaluate(model, 
                test_dl=dataloader.get_batch_iterator(for_train=False), rooted=False)
            msg = "ep:[{}] mse:[{}]".format(ep, test_mse)
            logging.info("[test] {}".format(msg))
            print("{} [Perf][Test] {}".format(get_time(), msg))

        # save model
        if args.save_model and not ep % args.save_epoch_num \
            and ep >= args.save_after_epoch_num - 1:
            model_name = "model_ExpID{}_EP{}".format(args.experimentID, ep)
            torch.save(model.state_dict(), args.save_model_path + model_name)
            logging.info("[save] saving model: {}".format(model_name))
            print("{} saving {}".format(get_time(), model_name))


def evaluate(model, test_dl, rooted=False, restore_model_path=None):
    model.eval()
    full_pred = []
    full_pred_clamp = []
    full_target = []
    if restore_model_path:
        model.load_state_dict(torch.load(restore_model_path))
    with torch.no_grad():
        for test_batch in test_dl:
            test_batch = move_batch_to_gpu(test_batch)
            test_pred = model(test_batch)
            test_pred_clamp = torch.clamp(test_pred, 1.0, 5.0)
            test_target = test_batch['rtg'].float()

            full_pred.append(test_pred.cpu().detach().numpy())
            full_pred_clamp.append(test_pred_clamp.cpu().detach().numpy())
            full_target.append(test_target.cpu().detach().numpy())
        
    full_test_pred = np.concatenate(full_pred)
    full_test_target = np.concatenate(full_target)
    full_test_pred_clamp = np.concatenate(full_pred_clamp)
    mse = mean_squared_error(full_test_target, full_test_pred)
    clamp_mse = mean_squared_error(full_test_target, full_test_pred_clamp)

    if rooted:
        return np.sqrt(mse), np.sqrt(clamp_mse)
    else:
        return mse, clamp_mse
    


if __name__ == "__main__":

    print("="* 20 + "\n  ExperimentID " + args.experimentID + "\n" + "="*20)
    # config logging
    logging.basicConfig(filename='./log/{}.log'.format(args.experimentID), 
        filemode='w', level=logging.DEBUG, 
        format='[%(asctime)s][%(levelname)s][%(filename)s] %(message)s', 
        datefmt='%m/%d/%Y %H:%M:%S')
    
    # setup gpu
    if torch.cuda.device_count() > 0:
        use_gpu = True
        assert torch.cuda.device_count() > args.gpu_id
        torch.cuda.set_device("cuda:"+str(args.gpu_id))
        msg = "[cuda] with {} gpus, using cuda:{}".format(
            torch.cuda.device_count(), args.gpu_id)
    else:
        use_gpu = False
        msg = "[cuda] no gpus, using cpu"

    print_args(args)
    make_dir("./ckpt/")
    
    logging.info(msg)
    print("{} {}".format(get_time(), msg))
    # set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    model = APRE(args)

    # move model to cuda device
    if torch.cuda.is_available():
        model = model.cuda()

    dataloader = DataLoader(dataset=args.dataset, shuffle=args.shuffle,
        batch_size=args.batch_size, max_sample_num=args.max_review_num)
    
    if args.task == "train" or args.task == "both":
        train(model, dataloader)
    elif args.task == "test":
        evaluate(model, test_dl=dataloader.get_batch_iterator(for_train=False),
            restore_model_path=args.load_model_path)
    else:
        raise ValueError("args.task must in `train`, `test`, or `both`")