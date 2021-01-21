from utils import make_dir
import numpy as np
from itertools import product

CMD_DIM = """python src/train.py --experimentID {}\
    --gpu_id {} \
    --task both \
    --dataset digital_music \
    --shuffle \
    --batch_size 24 \
    --learning_rate {} \
    --num_epoch 30 \
    --regularization_weight {} \
    --eval_epoch_num 1 --eval_after_epoch_num 5 \
    --log_iter_num 20 \
    --num_aspects 296 --num_user 14138 --num_item 11707 \
    --feat_dim {} --num_last_layers {} \
    --ex_attn_temp {} --im_attn_temp {} \
    --max_review_num {} \
    --save_model --save_after_epoch_num 5 --save_epoch_num 2"""


CMD_AUT = """python src/train.py --experimentID {}\
    --gpu_id {} \
    --task both \
    --dataset automotive \
    --shuffle \
    --batch_size 24 \
    --learning_rate {} \
    --num_epoch 30 \
    --regularization_weight {} \
    --eval_epoch_num 1 --eval_after_epoch_num 5 \
    --log_iter_num 20 \
    --num_aspects 291 --num_user 2928 --num_item 1835 \
    --feat_dim {} --num_last_layers {} \
    --ex_attn_temp {} --im_attn_temp {} \
    --max_review_num {} \
    --save_model --save_after_epoch_num 5 --save_epoch_num 2"""


CMD_PES = """python src/train.py --experimentID {}\
    --gpu_id {} \
    --task both \
    --dataset pet_supplies \
    --shuffle \
    --batch_size 24 \
    --learning_rate {} \
    --num_epoch 30 \
    --regularization_weight {} \
    --eval_epoch_num 1 --eval_after_epoch_num 5 \
    --log_iter_num 20 \
    --num_aspects 529 --num_user 19854 --num_item 8510 \
    --feat_dim {} --num_last_layers {} \
    --ex_attn_temp {} --im_attn_temp {} \
    --max_review_num {} \
    --save_model --save_after_epoch_num 5 --save_epoch_num 2"""

CMD_TOG = """python src/train.py --experimentID {}\
    --gpu_id {} \
    --task both \
    --dataset toys_games \
    --shuffle \
    --batch_size 24 \
    --learning_rate {} \
    --num_epoch 30 \
    --regularization_weight {} \
    --eval_epoch_num 1 --eval_after_epoch_num 5 \
    --log_iter_num 20 \
    --num_aspects 680 --num_user 19409 --num_item 11924 \
    --feat_dim {} --num_last_layers {} \
    --ex_attn_temp {} --im_attn_temp {} \
    --max_review_num {} \
    --save_model --save_after_epoch_num 5 --save_epoch_num 2"""



# ################
# change here
# ################

# AVAILABLE_GPUS = [2, 4, 5, 6, 7]
AVAILABLE_GPUS = [2]
ds = "tog"
CMD = CMD_TOG

# End here


# 2 * 2 * 3 * 2 * 2 * 2 = 96
DEFAULT = {
    "lr": "0.00003",
    "reg_w": "0.00001",
    "feat_dim": "300",
    "nlastlyr": "2",
    "temp": "1.0",
    "max_rev_num": "30"
}
NEW_CONFIGS = [
    {"lr": ["0.00002"]},
    {"reg_w": ["0.000005"]},
]

# PARAMS = [list(x) for x in CONFIGS.values()]

def fulfill_cmd(exp_id, gpu_id, l):
    assert len(l) == 7
    return CMD.format(exp_id, gpu_id, *l)


if __name__ == "__main__":
    exp_id = 40001

    make_dir("./grid_search/")
    summary_out = open("./grid_search/summary.out", "w+")
    print("exp_id, gpu_id, ds, lr, reg_w, feat_dim, nlastlyr, ex_tmp, im_tmp, max_rev_num",
        file=summary_out)
    
    param_list = []
    for new_cf in NEW_CONFIGS:
        default_cf_copy = DEFAULT.copy()
        new_setting_key = list(new_cf.keys())[0]
        new_setting_values = new_cf[new_setting_key]
        for new_setting_val in new_setting_values:
            default_cf_copy[new_setting_key] = new_setting_val
            param_list.append(
                [default_cf_copy['lr'],
                 default_cf_copy['reg_w'],
                 default_cf_copy['feat_dim'],
                 default_cf_copy['nlastlyr'],
                 default_cf_copy['temp'],
                 default_cf_copy['temp'],  # temp twice for ex and im
                 default_cf_copy['max_rev_num']])

    each_gpu_workload = np.array_split(param_list, len(AVAILABLE_GPUS))
    for i, gpu_id in enumerate(AVAILABLE_GPUS):
        with open("./grid_search/run_bash_{}_gpu{}.sh".format(ds, gpu_id), "w") as fout:
            for param_sublists in each_gpu_workload:
                param_sublists = list(param_sublists)
                print(param_sublists)
                for param_sublist in param_sublists:
                    cmd = fulfill_cmd(exp_id, gpu_id, param_sublist)
                    print(cmd, file=fout)
                    print(type(exp_id))
                    print(type(gpu_id))
                    print(param_sublist)
                    print(",".join([str(exp_id), str(gpu_id), ds]+list(param_sublist)), file=summary_out)

                    exp_id += 1
    
    summary_out.close()
            