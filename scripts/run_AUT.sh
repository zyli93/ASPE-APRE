# TODO: change 0

# default
# python src/train.py --experimentID 180001 --gpu_id 0 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 40 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --transf_wordemb_func "else" \
#     --scheduler_stepsize 3


# 180002 learning rate 1e-4
python src/train.py --experimentID 180002 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-4 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3


# 180003 feat_dim 100
python src/train.py --experimentID 180003 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 100 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 100 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 180004 feat_dim 400
# python src/train.py --experimentID 180004 --gpu_id 0 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 40 \
#     --learning_rate 1e-3 \
#     --feat_dim 300 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --scheduler_stepsize 3

# 180005 dropout 0.3
python src/train.py --experimentID 180005 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.3 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 180006 dropout 0.4
python src/train.py --experimentID 180006 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.4 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 180007 dropout 0.5
python src/train.py --experimentID 180007 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.5 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3


# 180008 dropout 0.
python src/train.py --experimentID 180008 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0. \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 180009 dropout 0.1
python src/train.py --experimentID 180009 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.1 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 180010 im_kernel 6
python src/train.py --experimentID 180010 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 6 \
    --scheduler_stepsize 3


# 180011 im_kernel 8
python src/train.py --experimentID 180011 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 8 \
    --scheduler_stepsize 3

# 180012 im_kernel 10
python src/train.py --experimentID 180012 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 10 \
    --scheduler_stepsize 3


# 180013 feat_dim 50
python src/train.py --experimentID 180013 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 50 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 50 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 180014 feat_dim 150
python src/train.py --experimentID 180014 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 150 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 150 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 180015 step size 2
python src/train.py --experimentID 180015 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 2

# 180016 transf_wordemb_func relu
python src/train.py --experimentID 180016 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --transf_wordemb_func "relu" \
    --scheduler_stepsize 3

# 180017 transf_wordemb_func leakyrelu
python src/train.py --experimentID 180017 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --transf_wordemb_func "leakyrelu" \
    --scheduler_stepsize 3

