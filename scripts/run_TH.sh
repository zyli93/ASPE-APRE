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

# 4001 learning rate 1e-4
# new default
if [ $1 = 0 ]
then
    # 4001, feat_dim 200, dropout 0.2
    CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 4001 --gpu_id $2 --task both --dataset tools_home --shuffle \
        --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 27 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --transf_wordemb_func "else" \
        --scheduler_stepsize 3

    # 4002, feat_dim 200, dropout 0.3
    CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 4002 --gpu_id $2 --task both --dataset tools_home --shuffle \
        --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 27 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.3 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3
    
    # 4003, feat_dim 128; dropout - 0.2
    CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 4003 --gpu_id $2 --task both --dataset tools_home --shuffle \
        --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 27 \
        --learning_rate 1e-3 \
        --feat_dim 128 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 128 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3

    # 4004 feat_dim 128; dropout - 0.3
    CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 4004 --gpu_id $2 --task both --dataset tools_home --shuffle \
        --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 27 \
        --learning_rate 1e-3 \
        --feat_dim 128 \
        --regularization_weight 1e-4\
        --dropout 0.3 \
        --cnn_out_channel 128 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3

    # # 200004 feat_dim 300
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200004 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 300 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 300 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # # 200005 dropout 0.2
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200005 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.2\
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # # 200006 dropout 0.4
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200006 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.4 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3
else
    echo "Nothing"
    # 200007 dropout 0.5
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200007 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.5 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3


    # # 200008 dropout 0.
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200008 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0. \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 200009 dropout 0.1
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200009 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.1 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 200010 im_kernel 6
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200010 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 6 \
    #     --scheduler_stepsize 3


    # 200011 im_kernel 8
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200011 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 8 \
    #     --scheduler_stepsize 3

    # 200012 im_kernel 10
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200012 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 10 \
    #     --scheduler_stepsize 3


    # 200013 feat_dim 50
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200013 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 50 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 50 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 200014 feat_dim 150
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200014 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 150 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 150 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 200015 step size 2
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200015 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 2

    # 200016 transf_wordemb_func relu
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200016 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --transf_wordemb_func "relu" \
    #     --scheduler_stepsize 3

    # 200017 transf_wordemb_func leakyrelu
    # CUDA_LAUNCH_BLOCKING=1 python src/train.py --experimentID 200017 --gpu_id $2 --task both --dataset tools_home --shuffle \
    #     --num_aspects 659 --num_user 16633 --num_item 10217 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 27 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.3 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --transf_wordemb_func "leakyrelu" \
    #     --scheduler_stepsize 3
fi
