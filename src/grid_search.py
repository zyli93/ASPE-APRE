from utils import make_dir
from itertools import product

CMD = """python src/train.py --experimentID {}\
    --gpu_id {} \
    --task both \
    --dataset {} \
    --shuffle \
    --batch_size 24 \
    --learning_rate {} \
    --num_epoch 20 \
    --regularization_weight {} \
    --eval_epoch_num 1 --eval_after_epoch_num 5 \
    --log_iter_num 20 \
    --num_aspects 291 --num_user 2928 --num_item 1835 \
    --feat_dim {} --num_last_layers {} \
    --ex_attn_temp {} --im_attn_temp {} \
    --max_review_num {} \
    --save_model --save_after_epoch_num 5 --save_epoch_num 2"""



    .format(
            exp_id,
            gpu_id,
            ds,
            lr,
            reg_w,
            feat_dim,
            nlastlyr,
            ex_temp,
            im_temp,
            max_rev_num)

# 2 * 2 * 3 * 2 * 2 * 2 = 96
CONFIGS = {
    "ds": ["digital_music"],  # [1, 3, 5],  # 3
    "lr": [0.00003, 0.00005],  # 2, binary is better than ranking
    "reg_w": [0.0001, 0.00001],
    "feat_dim": [300],
    "nlastlyr": [1, 2, 4],
    "ex_temp": [0.8, 1.0],
    "im_temp": [0.8, 1.0],
    "max_rev_num": [20, 30]
}

PARAMS = [list(x) for x in CONFIGS.values()]

def fulfill_cmd(exp_id, gpu_id, l):
    assert len(l) == 8
    return CMD.format(exp_id, gpu_id, *l)


if __name__ == "__main__":
    start_id = 10001

    make_dir("./grid_search/")
    summary_out = open("./grid_search/summary.out", "w")
    print("exp_id, gpu_id, ds, lr, reg_w, feat_dim, nlastlyr, ex_tmp, im_tmp, max_rev_num",
        file=summary_out)
    
    for i, param_list in enumerate(product(*PARAMS)):

        for i, param_list in enumerate(product(*PARAMS)):
            tid = i + TID_START
            print(tid)
            print(param_list)
            cmd_line = fulfill_cmd(tid, param_list)
            print(str(tid)+","+",".join([str(x) for x in param_list]),
                  file=fout)
            os.system(cmd_line

    for gpu_id in [2, 4, 5, 6, 7]:
        with open("./grid_search/run_bash_{}.sh".format(gpu_id), "w") as fout:
            # output sh