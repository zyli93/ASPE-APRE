# Instruction
# bash run_train_2_pes.sh [copy ID] [gpu ID]
#   copy ID is 0 or non-0
#   gpu ID is the id of GPU

# default
# python src/train.py --experimentID 100001 --gpu_id 2 --task both --dataset pet_supplies --shuffle \
#     --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 28 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.3 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --transf_wordemb_func "else" \
#     --scheduler_stepsize 3

# CHANGE LOG
#   1. commented 100001
#   2. changed default lr to 1e-4
#   3. change dropout default to 0.3
#   4. removed dropout 0. and 0.1
#   5. removed feat_dim 50, 100, and 150
#   6. changed batch_size to 28 as Scai2 mem is tiny

# NOTE:
#   1. test run one cmd and make sure mem doesn't explode
#   2. ID's left -2, -4, -5, -6, -7, -15, -16, -17

# 100002 learning rate 1e-4
# new default
if [ $1 = 0 ]
then
    python src/train.py --experimentID 100002 --gpu_id $2 --task both --dataset pet_supplies --shuffle \
        --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 28 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.3 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3


    # 100003 feat_dim 100
    # python src/train.py --experimentID 100003 --gpu_id 2 --task both --dataset pet_supplies --shuffle \
    #     --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 28 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 100 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 100 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 100004 feat_dim 300
    python src/train.py --experimentID 100004 --gpu_id $2 --task both --dataset pet_supplies --shuffle \
        --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 28 \
        --learning_rate 1e-3 \
        --feat_dim 300 \
        --regularization_weight 1e-4\
        --dropout 0.3 \
        --cnn_out_channel 300 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3

    # 100005 dropout 0.2
    python src/train.py --experimentID 100005 --gpu_id $2 --task both --dataset pet_supplies --shuffle \
        --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 28 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.2\
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3

    # 100006 dropout 0.4
    python src/train.py --experimentID 100006 --gpu_id $2 --task both --dataset pet_supplies --shuffle \
        --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 28 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.4 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3
else

    # 100007 dropout 0.5
    python src/train.py --experimentID 100007 --gpu_id $2 --task both --dataset pet_supplies --shuffle \
        --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 28 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.5 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3


    # # 100008 dropout 0.
    # python src/train.py --experimentID 100008 --gpu_id 2 --task both --dataset pet_supplies --shuffle \
    #     --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 28 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0. \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 100009 dropout 0.1
    # python src/train.py --experimentID 100009 --gpu_id 2 --task both --dataset pet_supplies --shuffle \
    #     --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 28 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.1 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 100010 im_kernel 6
    # python src/train.py --experimentID 100010 --gpu_id 2 --task both --dataset pet_supplies --shuffle \
    #     --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 28 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 6 \
    #     --scheduler_stepsize 3


    # 100011 im_kernel 8
    # python src/train.py --experimentID 100011 --gpu_id 2 --task both --dataset pet_supplies --shuffle \
    #     --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 28 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 8 \
    #     --scheduler_stepsize 3

    # 100012 im_kernel 10
    # python src/train.py --experimentID 100012 --gpu_id 2 --task both --dataset pet_supplies --shuffle \
    #     --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 28 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 10 \
    #     --scheduler_stepsize 3


    # 100013 feat_dim 50
    # python src/train.py --experimentID 100013 --gpu_id 2 --task both --dataset pet_supplies --shuffle \
    #     --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 28 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 50 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 50 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 100014 feat_dim 150
    # python src/train.py --experimentID 100014 --gpu_id 2 --task both --dataset pet_supplies --shuffle \
    #     --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 28 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 150 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 150 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 100015 step size 2
    python src/train.py --experimentID 100015 --gpu_id $2 --task both --dataset pet_supplies --shuffle \
        --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 28 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.3 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 2

    # 100016 transf_wordemb_func relu
    python src/train.py --experimentID 100016 --gpu_id $2 --task both --dataset pet_supplies --shuffle \
        --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 28 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.3 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --transf_wordemb_func "relu" \
        --scheduler_stepsize 3

    # 100017 transf_wordemb_func leakyrelu
    python src/train.py --experimentID 100017 --gpu_id $2 --task both --dataset pet_supplies --shuffle \
        --num_aspects 529 --num_user 19854 --num_item 8510 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 28 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.3 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --transf_wordemb_func "leakyrelu" \
        --scheduler_stepsize 3
fi
