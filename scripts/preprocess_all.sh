echo "processing office_products"
python src/preprocess.py --dataset=amazon --amazon_subset=office_products

echo "processing video_games"
python src/preprocess.py --dataset=amazon --amazon_subset=video_games

echo "processing pet_supplies"
python src/preprocess.py --dataset=amazon --amazon_subset=pet_supplies

echo "processing sports_outdoors"
python src/preprocess.py --dataset=amazon --amazon_subset=sports_outdoors

echo "processing home_kitchen"
python src/preprocess.py --dataset=amazon --amazon_subset=home_kitchen

