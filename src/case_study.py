import sys
import torch
from easydict import EasyDict
import numpy as np

from tqdm import tqdm

from modelCaseStudy import APRE
from postprocess import EntityReviewAggregation
from dataloader import DataLoader
from utils import print_args, make_dir, get_time, dump_pkl, tensor_to_numpy, move_batch_to_gpu

N_U = 2928 
N_T = 1835 
DIM = 128
LOAD_PATH = "./ckpt/model_ExpID70004_EP11"
GPU = 3
DIR = "./case_study/"

args = {
    "dataset": "automotive",
    "shuffle": False,
    "batch_size": 28,
    "padded_length": 100,
    "random_seed": 1993,
    "disable_explicit":  False,
    "disable_implicit": False,
    "num_aspects": 291,
    "num_user": N_U,
    "num_item": N_T,
    "feat_dim": DIM,
    "regularization_weight": 1E-4,
    "dropout": 0.2,
    "cnn_out_channel": DIM,
    "transf_wordemb_func": "else",
    "im_kernel_size": 4,
    "load_model_path": LOAD_PATH,
    "gpu_id": GPU,
    "max_review_num": 30
}

args = EasyDict(args)

if torch.cuda.device_count() > 0:
    use_gpu = True
    assert torch.cuda.device_count() > args.gpu_id
    torch.cuda.set_device("cuda:"+str(args.gpu_id))
    msg = "[cuda] with {} gpus, using cuda:{}".format(
        torch.cuda.device_count(), args.gpu_id)
else:
    use_gpu = False
    msg = "[cuda] no gpus, using cpu"

print(msg)


# print arguments
print_args(args)
make_dir(DIR)

print("[case study] loading model from {}".format(args.load_model_path))
model = APRE(args)
model.load_state_dict(torch.load(args.load_model_path))

print("[case study] build dataloader")
dataloader = DataLoader(dataset=args.dataset, shuffle=args.shuffle,
    batch_size=args.batch_size, max_sample_num=args.max_review_num, 
    use_gpu=use_gpu)

trn_iter = dataloader.get_batch_iterator()
tst_iter = dataloader.get_batch_iterator(for_train=False)

user_rev_attn = {}
item_rev_attn = {}
trn_ui_info = {}
tst_ui_info = {}

# turn on evaluation mode for model
model.eval()
model.cuda()

with torch.no_grad():
    for idx, batch in tqdm(enumerate(trn_iter)):

        if use_gpu:
            # if model on cuda, them move batch to cuda
            batch = move_batch_to_gpu(batch)

        # make pred
        pred, urev_attn_w, irev_attn_w, ex_mlp = model(batch)

        uid_list, iid_list = batch['uid_list'], batch['iid_list']

        assert len(uid_list) == len(iid_list), "User and item lists, unequal length"

        # get ground truth, convert to float tensor
        target = batch['rtg'].float()
        pred = tensor_to_numpy(pred, True)
        target = tensor_to_numpy(target, True)

        urev_attn_w = [tensor_to_numpy(x, True) for x in urev_attn_w] # bs * [nrev, num_asp]
        irev_attn_w = [tensor_to_numpy(x, True) for x in irev_attn_w] # bs * [nrev, num_asp]
        ex_mlp = tensor_to_numpy(ex_mlp, True) # (bs, num_asp)

        for i in range(len(uid_list)):
            uid, iid = uid_list[i], iid_list[i]
            if uid not in user_rev_attn:
                user_rev_attn[uid] = urev_attn_w[i]
            
            if iid not in item_rev_attn:
                item_rev_attn[iid] = irev_attn_w[i]
            
            trn_ui_info[(uid, iid)] = {"mlp": ex_mlp[i], "pred": pred[i], "target": target[i]}

    for idx, batch in tqdm(enumerate(tst_iter)):

        if use_gpu:
            # if model on cuda, them move batch to cuda
            batch = move_batch_to_gpu(batch)
        
        # make pred
        pred, _, _, ex_mlp = model(batch)

        uid_list, iid_list = batch['uid_list'], batch['iid_list']
        assert len(uid_list) == len(iid_list), "User and item lists, unequal length (test)"
        
        target = batch['rtg'].float()

        # get ground truth, convert to float tensor
        ex_mlp = tensor_to_numpy(ex_mlp, True)
        pred = tensor_to_numpy(pred, True)
        target = tensor_to_numpy(target, True)

        for i in range(len(uid_list)):
            uid, iid = uid_list[i], iid_list[i]
            tst_ui_info[(uid, iid)] = {"mlp": ex_mlp[i], "pred": pred[i], "target": target[i]}
    
    gamma = tensor_to_numpy(model.gamma, True)

dump_pkl(DIR + "user_rev_attn.pkl", user_rev_attn)
dump_pkl(DIR + "item_rev_attn.pkl", item_rev_attn)
dump_pkl(DIR + "trn_ui_info.pkl", trn_ui_info)
dump_pkl(DIR + "tst_ui_info.pkl", tst_ui_info)
dump_pkl(DIR + "gamma.pkl", gamma)