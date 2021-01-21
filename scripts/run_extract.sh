python src/extract.py --data_path=./data/amazon/automotive/ --count_threshold=50 --run_mapping
python src/extract.py --data_path=./data/amazon/digital_music/ --count_threshold=100 --run_mapping
python src/extract.py --data_path=./data/amazon/pet_supplies/ --count_threshold=150 --run_mapping
python src/extract.py --data_path=./data/amazon/toys_games/ --count_threshold=150 --run_mapping
python src/extract.py --data_path=./data/amazon/sports_outdoors/ --count_threshold=250 --run_mapping

python src/extract.py --data_path=./data/amazon/toys_games/ --count_threshold=150
python src/extract.py --data_path=./data/amazon/tools_home --count_threshold=150