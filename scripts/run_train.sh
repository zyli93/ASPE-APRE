# Tunable params:
#       learning rate: 1e-3 to 1e-4 to 3e-5
#       mid_dim: now at 300, try 64, 128, 256
#       reg_weight: now at 1e-4, try larger or smaller
# Temp params:
#       num_epoch: later change to larger
#       log_item_num: later change to larger
#       eval_epoch_num: later change to smaller (like 1).
#               Right now, only debug for training

# Run cmd:
#   $ bash run_train.sh [exp_id] [dataset_name] [gpu_id] [num_aspects] [num_user] [num_item]

#   $ bash run_train.sh [exp_id] [dataset_name] [gpu_id] [num_aspects] [num_user] [num_item]
#   $ bash run_train.sh [exp_id] [dataset_name] [gpu_id] [num_aspects] [num_user] [num_item]
#   $ bash run_train.sh [exp_id] [dataset_name] [gpu_id] [num_aspects] [num_user] [num_item]

# automotive
# num user 2928, num_item 1835
# num aspect 291

# sports_outdoors
# Oh no for user!
# num user 35590, num_item 18357
# num aspect 747

# digital_music
# Oh no for user!
# Oh no for item!
# num user 14138, num_item 11707
# num aspects 296

# toys_games
# Oh no for user!
# num user 19409, num_item 11924
# num aspects 680

# pet_supplies
# num user 19854, num_item 8510
# num aspects 529

echo "using GPU $2"

if [ $1 = automotive ]
    then
    python src/train.py --experimentID 0001 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 1 \
        --learning_rate 0.0001 \
        --num_epoch 1 \
        --log_iter_num 10 \
        --eval_epoch_num 1000 --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 300 --num_last_layers 4 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --gpu_id $2 
 else
    echo "not supported"
fi