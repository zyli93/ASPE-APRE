# This file is used for Ablation study

# Disabling Explicit Channel

# AUT 8001
# python src/train.py --experimentID 8001 --gpu_id $1 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 27 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --scheduler_stepsize 3 \
#     --disable_explicit

# # DIM 8002
# python src/train.py --experimentID 8002 --gpu_id $1 --task both --dataset digital_music --shuffle \
#     --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 27 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --scheduler_stepsize 3 \
#     --disable_explicit

# PES 8003
# python src/train.py --experimentID 8003 --gpu_id $1 --task both --dataset pet_supplies --shuffle \
#     --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 27 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --scheduler_stepsize 3 \
#     --disable_explicit

# TH 18004
python src/train.py --experimentID 18004 --gpu_id $1 --task both --dataset tools_home --shuffle \
    --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_explicit

# MI 18005
python src/train.py --experimentID 18005 --gpu_id $1 --task both --dataset musical_instruments --shuffle \
    --num_aspects 167 --num_user 1429 --num_item 900 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_explicit

# TOG 18006
python src/train.py --experimentID 18006 --gpu_id $1 --task both --dataset toys_games --shuffle \
    --num_aspects 680 --num_user 19415 --num_item 11924 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_explicit

# SPO 8007
python src/train.py --experimentID 8007 --gpu_id $1 --task both --dataset sports_outdoors --shuffle \
    --num_aspects 747 --num_user 35590 --num_item 18357 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 24 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_explicit
