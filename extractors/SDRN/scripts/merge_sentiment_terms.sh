# # home_kitchen  office_products  pet_supplies  sports_outdoors  video_games
# python src/parse_output.py 2014Lap home_kitchen
# python src/parse_output.py 2014Res home_kitchen
# python src/parse_output.py 2015Res home_kitchen

echo "merge sentiment terms..."
for ds in home_kitchen  office_products  pet_supplies  sports_outdoors  video_games digital_music
do
    python parse_output.py merge xx $ds
done