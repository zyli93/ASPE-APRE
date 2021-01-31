# [x] kernel size larger
# [x] dropout 0.5, dropout 0.1
# [x] SP disable EX

# other things like D-Attn/NARRE/MPCN/DAML/NeuMF

if [ $1 = 0 ]
then
    python src/train.py --experimentID 8007 --gpu_id $2 --task both --dataset sports_outdoors --shuffle \
        --num_aspects 747 --num_user 35592 --num_item 18357 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
        --batch_size 24 \
        --learning_rate 1e-3 \
        --feat_dim 200 \
        --regularization_weight 1e-4\
        --dropout 0.2 \
        --cnn_out_channel 200 \
        --im_kernel_size 4 \
        --scheduler_stepsize 3 \
        --disable_explicit
else
    # kernel size 12
    python src/train.py --experimentID 101  --gpu_id $2 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 12 \
    --scheduler_stepsize 3

    # kernel size 2
    python src/train.py --experimentID 102 --gpu_id $2 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 200 \
    --im_kernel_size 2 \
    --scheduler_stepsize 3

    # dropout 0.1
    python src/train.py --experimentID 103  --gpu_id $2 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.1 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

    # dropout 0.5
    python src/train.py --experimentID 104  --gpu_id $2 --task both --dataset automotive --shuffle \
    --num_aspects 291 --num_user 2928 --num_item 1835 --num_epoch 15 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 27 \
    --learning_rate 1e-3 \
    --feat_dim 200 \
    --regularization_weight 1e-4\
    --dropout 0.5 \
    --cnn_out_channel 200 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3

 
fi
