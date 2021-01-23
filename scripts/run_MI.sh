# TODO: change 0

# default


# CHANGE LOG
#   1. commented 200001
#   2. changed default lr to 1e-4
#   3. change dropout default to 0.3
#   4. removed dropout 0. and 0.1
#   5. removed feat_dim 50, 100, and 150
#   6. changed batch_size to 35 as Scai2 mem is tiny

# NOTE:
#   1. test run one cmd and make sure mem doesn't explode
#   2. ID's left -2, -4, -5, -6, -7, -15, -16, -17

# 7001 learning rate 1e-4
# new default
if [ $1 = 0 ]
then
    # 7001, feat_dim 200, dropout 0.2
    CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 7001 --gpu_id $2 --task both --dataset musical_instruments --shuffle \
        --num_aspects 167 --num_user 1429 --num_item 900 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 27 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --transf_wordemb_func "else" \
        --scheduler_stepsize 3

    # 7002, feat_dim 200, dropout 0.3
    CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 7002 --gpu_id $2 --task both --dataset musical_instruments --shuffle \
        --num_aspects 167 --num_user 1429 --num_item 900 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 27 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.3 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3
    
    # 7003, feat_dim 128; dropout - 0.2
    CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 7003 --gpu_id $2 --task both --dataset musical_instruments --shuffle \
        --num_aspects 167 --num_user 1429 --num_item 900 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 27 \
        --learning_rate 1e-3 \
        --feat_dim 128 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 128 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3

    # 7004 feat_dim 128; dropout - 0.3
    CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 7004 --gpu_id $2 --task both --dataset musical_instruments --shuffle \
        --num_aspects 167 --num_user 1429 --num_item 900 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 27 \
        --learning_rate 1e-3 \
        --feat_dim 128 \
        --regularization_weight 1e-4\
        --dropout 0.3 \
        --cnn_out_channel 128 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3

   
else
    echo "Nothing"
    
fi
