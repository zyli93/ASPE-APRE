# TODO: change 0

# default
python src/train.py --experimentID 70001 --gpu_id $1 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --transf_wordemb_func "else" \
    --scheduler_stepsize 3 \
    --save_model \
    --save_after_epoch_num 5 \
    --save_epoch_num 1

python src/train.py --experimentID 70002 --gpu_id $1 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.3 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --transf_wordemb_func "else" \
    --scheduler_stepsize 3 \
    --save_model \
    --save_after_epoch_num 5 \
    --save_epoch_num 1

python src/train.py --experimentID 70003 --gpu_id $1 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 128 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 128 \
    --im_kernel_size 4 \
    --transf_wordemb_func "else" \
    --scheduler_stepsize 3 \
    --save_model \
    --save_after_epoch_num 5 \
    --save_epoch_num 1


python src/train.py --experimentID 70004 --gpu_id $1 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 40 \
    --learning_rate 1e-3 \
    --feat_dim 128 \
    --regularization_weight 1e-4\
    --dropout 0.3 \
    --cnn_out_channel 128 \
    --im_kernel_size 4 \
    --transf_wordemb_func "else" \
    --scheduler_stepsize 3 \
    --save_model \
    --save_after_epoch_num 5 \
    --save_epoch_num 1