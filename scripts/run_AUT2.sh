# Need to rerun 180022

#  Learning rate experiments
# 180020 learning rate 5e-4
python src/train.py --experimentID 180020 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 5e-4 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3


# 180021 learning rate 1e-2
python src/train.py --experimentID 180021 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-2 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 180022 learning rate 5e-3
python src/train.py --experimentID 180022 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 5e-3 \
    --feat_dim 200 --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# Regularization weight experiments
# 180023 regW 1e-3
python src/train.py --experimentID 180023 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-3\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 180024 regW 1e-5
python src/train.py --experimentID 180024 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-5\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

# 180025 regW 1e-6
python src/train.py --experimentID 180025 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-6\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3


# Make up relu
# 180026 - relu
python src/train.py --experimentID 180026 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --transf_wordemb_func relu\
    --scheduler_stepsize 3

# 180027 tanh
python src/train.py --experimentID 180027 --gpu_id 0 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --transf_wordemb_func tanh\
    --scheduler_stepsize 3

# # 180010 im_kernel 6
# python src/train.py --experimentID 180010 --gpu_id 0 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 40 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 6 \
#     --scheduler_stepsize 3


# # 180011 im_kernel 8
# python src/train.py --experimentID 180011 --gpu_id 0 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 40 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 8 \
#     --scheduler_stepsize 3

# # 180012 im_kernel 10
# python src/train.py --experimentID 180012 --gpu_id 0 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 40 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 10 \
#     --scheduler_stepsize 3


# # 180013 feat_dim 50
# python src/train.py --experimentID 180013 --gpu_id 0 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 40 \
#     --learning_rate 1e-3 \
#     --feat_dim 50 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 50 \
#     --im_kernel_size 4 \
#     --scheduler_stepsize 3

# # 180014 feat_dim 150
# python src/train.py --experimentID 180014 --gpu_id 0 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 40 \
#     --learning_rate 1e-3 \
#     --feat_dim 150 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 150 \
#     --im_kernel_size 4 \
#     --scheduler_stepsize 3

# # 180015 step size 2
# python src/train.py --experimentID 180015 --gpu_id 0 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 40 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --scheduler_stepsize 2

# # 180016 transf_wordemb_func relu
# python src/train.py --experimentID 180016 --gpu_id 0 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 40 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --transf_wordemb_func "relu" \
#     --scheduler_stepsize 3

# # 180017 transf_wordemb_func leakyrelu
# python src/train.py --experimentID 180017 --gpu_id 0 --task both --dataset automotive --shuffle \
#     --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
#     --batch_size 40 \
#     --learning_rate 1e-3 \
#     --feat_dim 200 \
#     --regularization_weight 1e-4\
#     --dropout 0.2 \
#     --cnn_out_channel 200 \
#     --im_kernel_size 4 \
#     --transf_wordemb_func "leakyrelu" \
#     --scheduler_stepsize 3

