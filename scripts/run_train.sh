python src/train.py --experimentID 2021 --gpu_id 0 --task both --dataset digital_music --shuffle \
    --num_aspects 296 --num_user 14138 --num_item 11707 --num_epoch 25 --eval_epoch_num 1 --eval_after_epoch_num 1 --log_iter_num 20 --max_review_num 30 \
    --batch_size 30 \
    --learning_rate 1e-3 \
    --feat_dim 100 \
    --regularization_weight 1e-4\
    --dropout 0.2 \
    --cnn_out_channel 100 \
    --im_kernel_size 4 \
    --scheduler_stepsize 3