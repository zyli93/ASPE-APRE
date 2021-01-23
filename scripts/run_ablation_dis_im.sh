# This file is used for Ablation study

# Disabling Explicit Channel

# AUT 9001
python src/train.py --experimentID 9001 --gpu_id $1 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_implicit

# DIM 9002
python src/train.py --experimentID 9002 --gpu_id $1 --task both --dataset digital_music --shuffle \
    --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_implicit

# PES 9003
python src/train.py --experimentID 9003 --gpu_id $1 --task both --dataset pet_supplies --shuffle \
    --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_implicit

# TH 9004
python src/train.py --experimentID 9004 --gpu_id $1 --task both --dataset tools_home --shuffle \
    --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_implicit

# MI 9005
python src/train.py --experimentID 9005 --gpu_id $1 --task both --dataset musical_instruments --shuffle \
    --num_aspects 167 --num_user 1429 --num_item 900 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_implicit

# TOG 9006
python src/train.py --experimentID 9006 --gpu_id $1 --task both --dataset toys_games --shuffle \
    --num_aspects 680 --num_user 19415 --num_item 11924 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_implicit

# SPO 9007
python src/train.py --experimentID 9007 --gpu_id $1 --task both --dataset sports_outdoors --shuffle \
    --num_aspects 747 --num_user 35590 --num_item 18357 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 24 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3 \
    --disable_implicit

# python src/train.py --experimentID 9002 --gpu_id 3 --task both --dataset digital_music --shuffle \
#     --num_aspects 296 --num_user 14235 --num_item 11745 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 27 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --scheduler_stepsize 3 \
#     --disable_implicit