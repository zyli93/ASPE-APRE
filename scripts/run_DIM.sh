# TODO: change 0

# default
# python src/train.py --experimentID 90001 --gpu_id $2 --task both --dataset digital_music --shuffle \
#     --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 30 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --transf_wordemb_func "else" \
#     --scheduler_stepsize 3


# # 90002 learning rate 1e-4
# python src/train.py --experimentID 90002 --gpu_id $2 --task both --dataset digital_music --shuffle \
#     --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 30 \
#     --learning_rate 1e-4 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --scheduler_stepsize 3


if [ $1 = 0 ]
then
    # 90003 feat_dim 100
    python src/train.py --experimentID 90003 --gpu_id $2 --task both --dataset digital_music --shuffle \
        --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 30 \
        --learning_rate 1e-3 \
        --feat_dim 100 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 100 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3

    # 90004 feat_dim 300
    python src/train.py --experimentID 90004 --gpu_id $2 --task both --dataset digital_music --shuffle \
        --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 30 \
        --learning_rate 1e-3 \
        --feat_dim 300 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 300 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3

    # 90005 dropout 0.3
    python src/train.py --experimentID 90005 --gpu_id $2 --task both --dataset digital_music --shuffle \
        --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 30 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.3 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3

    # 90006 dropout 0.4
    python src/train.py --experimentID 90006 --gpu_id $2 --task both --dataset digital_music --shuffle \
        --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 30 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.4 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3

    # 90007 dropout 0.5
    python src/train.py --experimentID 90007 --gpu_id $2 --task both --dataset digital_music --shuffle \
        --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 30 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.5 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3


    # 90008 dropout 0.
    # python src/train.py --experimentID 90008 --gpu_id $2 --task both --dataset digital_music --shuffle \
    #     --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 30 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0. \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 90009 dropout 0.1
    # python src/train.py --experimentID 90009 --gpu_id $2 --task both --dataset digital_music --shuffle \
    #     --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 30 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.1 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 90010 im_kernel 6
    # python src/train.py --experimentID 90010 --gpu_id $2 --task both --dataset digital_music --shuffle \
    #     --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 30 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.2 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 6 \
    #     --scheduler_stepsize 3

else
    # 90011 im_kernel 8
    python src/train.py --experimentID 90011 --gpu_id $2 --task both --dataset digital_music --shuffle \
        --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 30 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 200 \
        --im_kernel_size 8 \
        --scheduler_stepsize 3

    # 90012 im_kernel 10
    # python src/train.py --experimentID 90012 --gpu_id $2 --task both --dataset digital_music --shuffle \
    #     --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 30 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 200 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.2 \
    #     --cnn_out_channel 200 \
    #     --im_kernel_size 10 \
    #     --scheduler_stepsize 3


    # 90013 feat_dim 50
    # python src/train.py --experimentID 90012 --gpu_id $2 --task both --dataset digital_music --shuffle \
    #     --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 30 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 50 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.2 \
    #     --cnn_out_channel 50 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 90013 feat_dim 150
    # python src/train.py --experimentID 90013 --gpu_id $2 --task both --dataset digital_music --shuffle \
    #     --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    #     --batch_size 30 \
    #     --learning_rate 1e-3 \
    #     --feat_dim 150 \
    #     --regularization_weight 1e-4\
    #     --dropout 0.2 \
    #     --cnn_out_channel 150 \
    #     --im_kernel_size 4 \
    #     --scheduler_stepsize 3

    # 90014 step size 2
    python src/train.py --experimentID 90014 --gpu_id $2 --task both --dataset digital_music --shuffle \
        --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 30 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 2

    # 90015 transf_wordemb_func relu
    python src/train.py --experimentID 90015 --gpu_id $2 --task both --dataset digital_music --shuffle \
        --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 30 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --transf_wordemb_func "relu" \
        --scheduler_stepsize 3

    # 90016 transf_wordemb_func leakyrelu
    python src/train.py --experimentID 90016 --gpu_id $2 --task both --dataset digital_music --shuffle \
        --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 30 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --transf_wordemb_func "leakyrelu" \
        --scheduler_stepsize 3
fi
