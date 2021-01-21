# TODO: change 0

# default
python src/train.py --experimentID 80001 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --trans_wordemb_func "else" \
    --scheduler_stepsize 3


# 80002 learning rate 1e-4
python src/train.py --experimentID 80002 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-4 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3


# 80003 feat_dim 100
python src/train.py --experimentID 80003 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 100 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 100 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 80004 feat_dim 400
python src/train.py --experimentID 80004 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 300 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 80005 dropout 0.3
python src/train.py --experimentID 80005 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.3 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 80006 dropout 0.4
python src/train.py --experimentID 80006 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.4 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 80007 dropout 0.5
python src/train.py --experimentID 80007 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.5 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3


# 80008 dropout 0.
python src/train.py --experimentID 80008 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0. \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 80009 dropout 0.1
python src/train.py --experimentID 80009 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.1 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 80010 im_kernel 6
python src/train.py --experimentID 80010 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 6 \
    --scheduler_stepsize 3


# 80011 im_kernel 8
python src/train.py --experimentID 80011 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 8 \
    --scheduler_stepsize 3

# 80012 im_kernel 10
python src/train.py --experimentID 80012 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 10 \
    --scheduler_stepsize 3


# 80013 feat_dim 50
python src/train.py --experimentID 80012 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 50 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 50 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 80013 feat_dim 150
python src/train.py --experimentID 80013 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 150 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 150 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 80014 step size 2
python src/train.py --experimentID 80014 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 2

# 80015 trans_wordemb_func relu
python src/train.py --experimentID 80015 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --trans_wordemb_func "relu" \
    --scheduler_stepsize 3

# 80016 trans_wordemb_func leakyrelu
python src/train.py --experimentID 80016 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --trans_wordemb_func "leakyrelu" \
    --scheduler_stepsize 3

