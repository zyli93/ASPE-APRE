echo "processing kindle_store, office_products, video_games"

python src/preprocess.py --dataset=amazon --amazon_subset=kindle_store
python src/preprocess.py --dataset=amazon --amazon_subset=office_products
python src/preprocess.py --dataset=amazon --amazon_subset=video_games

