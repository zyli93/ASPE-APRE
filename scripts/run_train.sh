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

if [ $1 = 0001 ]
then
    python src/train.py --experimentID 0001 \
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 30 \
        --learning_rate 0.00003 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 300 --num_last_layers 4 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2
    
# 0001 vs 0002 on learning rate
elif [ $1 == 0002 ]
then
    python src/train.py --experimentID 0002 \
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 30 \
        --learning_rate 0.00004 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 300 --num_last_layers 4 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2

elif [ $1 == 0003 ]
then
    python src/train.py --experimentID $1\
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 30 \
        --learning_rate 0.00003 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 200 --num_last_layers 4 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2

elif [ $1 = 0004 ]
then
    python src/train.py --experimentID $1 \
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 24 \
        --learning_rate 0.00003 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 128 --num_last_layers 4 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2
elif [ $1 = 0005 ]
then
    python src/train.py --experimentID $1 \
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 24 \
        --learning_rate 0.00003 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 300 --num_last_layers 2 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2
elif [ $1 = 0006 ]
then
    python src/train.py --experimentID $1 \
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 24 \
        --learning_rate 0.00003 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 128 --num_last_layers 2 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2

elif [ $1 = 0007 ]
then
    python src/train.py --experimentID $1 \
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 24 \
        --learning_rate 0.00002 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 128 --num_last_layers 2 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2

elif [ $1 = 0008 ]
then
    python src/train.py --experimentID $1 \
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 24 \
        --learning_rate 0.00002 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 200 --num_last_layers 2 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2

elif [ $1 = 0009 ]
then
    python src/train.py --experimentID $1 \
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 24 \
        --learning_rate 0.00003 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 128 --num_last_layers 2 \
        --regularization_weight 0.00001 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2

elif [ $1 = 0010 ]
then
    python src/train.py --experimentID $1 \
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 24 \
        --learning_rate 0.00002 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 200 --num_last_layers 2 \
        --ex_attn_temp 0.8 --im_attn_temp 0.8 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2

else
     python src/train.py --experimentID $1 \
        --gpu_id $2 \
        --task both \
        --dataset automotive \
        --shuffle \
        --batch_size 24 \
        --learning_rate 0.00003 \
        --num_epoch 20 \
        --eval_epoch_num 1 --eval_after_epoch_num 5 \
        --log_iter_num 20 \
        --num_aspects 291 --num_user 2928 --num_item 1835 \
        --feat_dim 300 --num_last_layers 2 \
        --ex_attn_temp 1.0 --im_attn_temp 1.0 \
        --max_review_num 30 \
        --save_model --save_after_epoch_num 5 --save_epoch_num 2
fi