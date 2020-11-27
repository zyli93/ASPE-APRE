# python src/preprocess.py --dataset=amazon --amazon_subset=digital_music
python src/annotate.py --path=./data/amazon/digital_music \
    --sdrn_anno_path=./extractors/SDRN/infer_on_ruara_2014Lap/annotation_digital_music.txt \
    --multi_proc_dep_parsing
