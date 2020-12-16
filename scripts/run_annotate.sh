# python src/annotate.py --path=data/amazon/home_kitchen \
#     --sdrn_anno_path=extractors/SDRN/infer_on_ruara_\
#     --use_senti_word_list \
#     --glove_dimension=300 \
#     --multi_proc_dep_parsing \
#     --num_workers_mp_dep=16

python src/annotate.py --path=data/amazon/home_kitchen \
    --sdrn_anno_path=extractors/SDRN/infer_on_ruara_\
    --use_senti_word_list \
    --glove_dimension=300 \
    --multi_proc_dep_parsing \
    --num_workers_mp_dep=16