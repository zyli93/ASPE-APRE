# # home_kitchen  office_products  pet_supplies  sports_outdoors  video_games
# python src/parse_output.py 2014Lap home_kitchen
# python src/parse_output.py 2014Res home_kitchen
# python src/parse_output.py 2015Res home_kitchen

echo "parsing output ..."
for trn in 2014Lap 2014Res 2015Res
do
    for ds in home_kitchen  office_products  pet_supplies  sports_outdoors  video_games
    do
        python parse_output.py $trn $ds
    done
done