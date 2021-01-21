# # home_kitchen  office_products  pet_supplies  sports_outdoors  video_games
# python src/parse_output.py 2014Lap home_kitchen
# python src/parse_output.py 2014Res home_kitchen
# python src/parse_output.py 2015Res home_kitchen

echo "parsing output ..."
# for ds in home_kitchen  office_products  pet_supplies  sports_outdoors  video_games
# for ds in automotive
# for ds in sports_outdoors toys_games
for ds in tools_home 
do
    echo "PARSING ..."
    for trn in 2014Lap 2014Res 2015Res
    do
        python parse_output.py parse $trn $ds
    done
    echo "MERGING ..."
    python parse_output.py merge xx $ds
done