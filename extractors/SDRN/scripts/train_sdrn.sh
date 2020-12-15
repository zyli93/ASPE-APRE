# default settings for everything

echo "Training SDRN with {$1} for {$2} iterations ..."

python main.py --mode train \
    --data ./data/$1.pt \
    --model_dir ./model/$1 \
    --eval_dir ./model/$1 \
    --iteration $2