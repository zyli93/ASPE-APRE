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

from dataloader import DataLoader
from model import APRE
from postprocess import EntityReviewAggregation

from utils import get_time, check_memory

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
        batch[tensor_name] = tensor.cuda()
    return batch


def train(model, dataloader):

    logging.info("[info] start training ID:[{}]".format(args.experimentID))
    print("[train] started training, ID:{}".format(args.experimentID))

    # optimizer
    optimizer = get_optimizer(model)

    # loss function
    criterion = nn.MSELoss()

    # if True, load model and continue to train
    if args.load_model:
        print("[train] loading model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
    total_num_iter = 0

    if args.task == "both":
        test_iter = dataloader.get_batch_iterator(for_train=False)
    
    for ep in range(args.num_epoch):
        trn_iter = dataloader.get_batch_iterator()
        model.train()  # model training flag

        total_loss = 0
        for idx, batch in enumerate(trn_iter):
            total_num_iter += 1

            """
            TODO: this comment delete later
                1. add process_batch function. move to device.
                2. change batch back to dictionary
                3. move data build from outside of model
            """
            
            for k, v in batch.items():
                print(k, v.element_size()*v.nelement())

            # TODO: to delete
            # sys.exit()

            print("before move_batch_to_gpu")
            check_memory()
            if torch.cuda.is_available():
                # if model on cuda, them move batch to cuda
                batch = move_batch_to_gpu(batch)
            print("after move_batch_to_gpu")
            check_memory()
            
            print(batch['u_split'].sum())

            # make pred
            pred = model(batch)

            # get ground truth
            target = torch.from_numpy(batch.rtg)

            # clean out existing gradients
            optimizer.zero_grad()

            # build loss term
            loss = criterion(pred, target)

            # optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # save model
            if args.save_model and not total_num_iter % args.save_model_iter_num:
                model_name = "model_id{}_ep{}_it{}".format(
                    args.experimentID, ep, total_num_iter)
                torch.save(model.state_dict(), path=args.save_model_path+"/"+model_name)
                logging.info("[save] saving model: {}".format(model_name))
                print("{} saving {}".format(get_time(), model_name))
                
            # log model
            if idx and not idx % args.total_num_iter:
                msg = "ep:[{}] iter:[{}/{}] loss:[{:.6f}]".format(
                    ep, idx, total_num_iter, loss.item())
                logging.info("[perf]-iter " + msg)
                print("{} perf-it {}".format(get_time(), msg))
                
        # avg loss this ep
        msg = "ep:[{}] iter:[{}] avgloss:[{:.6f}]".format(
            ep, total_num_iter, total_loss / trn_iter.get_train_batch_num())
        logging.info("[perf]-ep " + msg)
        print("{} perf-ep {}".format(get_time(), msg))
        
        # run test
        if not ep % args.eval_epoch_num and args.task == "both":
            test_mse = evaluate(model, test_iter, rooted=False)
            msg = "ep:[{}] mse:[{}]".format(ep, test_mse)
            logging.info("[test] {}".format(msg))
            print("{} perf-test {}".format(get_time(), msg))


def evaluate(model, test_dl, rooted=False, restore_model_path=None):
    model.eval()
    full_pred = []
    full_target = []
    if restore_model_path:
        model.load_state_dict(torch.load(restore_model_path))
    with torch.no_grad():
        for test_batch in test_dl:
            test_pred = model(test_batch)
            test_target = test_batch.rtg

            full_pred.append(test_pred.numpy())
            full_target.append(test_target)
        
    full_test_pred = np.concatenate(full_pred)
    full_test_target = np.concatenate(full_target)
    mse = mean_squared_error(full_test_target, full_test_pred)

    if rooted:
        return np.sqrt(mse)
    else:
        return mse
    


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
    
    logging.info(msg)
    print("{} {}".format(get_time(), msg))
    # set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    model = APRE(args)

    # move model to cuda device
    if torch.cuda.is_available():
        model = model.cuda()

    print(args.batch_size)
    dataloader = DataLoader(dataset=args.dataset, shuffle=args.shuffle,
        batch_size=args.batch_size)
    
    if args.task == "train" or args.task == "both":
        train(model, dataloader)
    elif args.task == "test":
        evaluate(model, test_dl=dataloader.get_batch_iterator(for_train=False),
            restore_model_path=args.load_model_path)
    else:
        raise ValueError("args.task must in `train`, `test`, or `both`")