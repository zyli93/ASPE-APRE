# automotive 
python src/postprocess.py --data_path=./data/amazon/automotive/ --num_aspects 291 ----n_partition 1

# digital_music
python src/postprocess.py --data_path=./data/amazon/digital_music/ --num_aspects 296 --n_partition 1

# pet_supplies
python src/postprocess.py --data_path=./data/amazon/pet_supplies/ --num_aspects 529 --n_partition 1  --num_workers 16

# sports_outdoors
python src/postprocess.py --data_path=./data/amazon/sports_outdoors/ --num_aspects 747 --n_partition 1 --num_workers 16

# toys_games
python src/postprocess.py --data_path=./data/amazon/toys_games/ --num_aspects 680 --n_partition 1 --num_workers 16

