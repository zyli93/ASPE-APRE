import os
from itertools import product
from datetime import datetime

CMD = """
CUDA_VISIBLE_DEVICES=2 python3 geapr/main.py \
    --trial_id {} \
    --ae_layers {} \
    --is_training \
    --epoch 20 \
    --batch_size 256 \
    --yelp_city lv \
    --nosave_model \
    --log_per_iter 400 \
    --negative_sample_ratio {} \
    --valid_set_size 256  --loss_type {} \
    --learning_rate {} \
    --regularization_weight {} \
    --embedding_dim {} \
    --hid_rep_dim {} \
    --tao {} \
    --num_total_item 17395 \
    --num_total_user 34389  \
    --ae_recon_loss_weight 0.01 \
    --gat_nheads 1 \
    --gat_ft_dropout {} \
    --gat_coef_dropout {} \
    --afm_use_dropout \
    --afm_dropout_rate {} \
    --afm_num_total_user_attr 140 \
    --afm_num_field 10 \
    --num_user_ctrd 16 \
    --num_item_ctrd 16 \
    --ctrd_corr_weight {} \
    --candidate_k 10,20,30,40,50,60,70,80,90,100
"""

# learning rate: 0.0005

CONFIGS = {
    "ae_layers": ["64,48"],  # ["32", "48,32"],  # 2, other options: "48"
    "negative_sample_ratio": [90, 105,120],  # [1, 3, 5],  # 3
    "loss_type": ["binary"],  # 2, binary is better than ranking
    "learning_rate": [0.001],  # [0.0001, 0.0003, 0.0005],  # 3, other options: 0.001
    "regularization_weight": [0.1],  #[0.01, 0.05, 0.1],  # 3
    "embedding_dim": [64],  # 2
    "hid_rep_dim": [32],  # 1
    "tao": [0],  # 1
    "gat_ft_dropout": [0, 0.2],  # [0.1, 0.3, 0.5],  # 3
    "gat_coef_dropout": [0, 0.2],  # [0.1, 0.3, 0.5],  # 3
    "afm_dropout_rate": [0, 0.2],  # [0.1, 0.3, 0.5],  # 3
    "ctrd_corr_weight": [0]  # 1
}


# DOM = ["trial_id", "ae_layers", "negative_sample_ratio", "loss_type",
#        "learning_rate", "regularization_weight",
#        "embedding_dim", "hid_rep_dim", "tao",
#        "gat_ft_dropout", "gat_coef_dropout", "afm_dropout_rate",
#        "ctrd_corr_weight"]

# AL = ["32", "48", "48,32"]
# NSR = [1, 3, 5]  # 3
# LT = ["ranking", "binary"]  # 2
# LR = [0.0001, 0.0003, 0.0005, 0.001]  # 4
# RW = [0.005, 0.01, 0.05, 0.1]  # 4
# ED = [64]  # 1
# HRD = [32]  # 1
# TAO = [0.1, 0.3, 0.5, 0.7]  # 4, commented due to no centroid
# TAO = [0]  # 1
# GFD = [0.1, 0.3, 0.5]  # 1
# GCD = [0.1, 0.3, 0.5]  # 1
# ADR = [0.1, 0.3, 0.5]  # 1
# CCW = [0.001, 0.005, 0.01, 0.1]  # 4, commented due to no centroid
# CCW = [0]  # 1

# PARAMS = [AL, NSR, LT, LR, RW, ED, HRD, TAO, GFD, GCD, ADR, CCW]

PARAMS = [list(x) for x in CONFIGS.values()]
TID_START = 700


def fulfill_cmd(id, l):
    assert len(l) == 12
    return CMD.format(id, *l)


if __name__ == "__main__":
    str_fn = datetime.now().isoformat()[8:19]
    # make_dir("./grid_search/")
    with open("./grid_search/{}.settings".format(str_fn), "w") as fout:
        print("tid,nsr,lt,lr,rw,ed,hrd,tao,gfd,gcd,adr,ccw", file=fout)
        for i, param_list in enumerate(product(*PARAMS)):
            tid = i + TID_START
            print(tid)
            print(param_list)
            cmd_line = fulfill_cmd(tid, param_list)
            print(str(tid)+","+",".join([str(x) for x in param_list]),
                  file=fout)
            os.system(cmd_line)
            # sys.exit()
