# python src/annotate.py --path=data/amazon/home_kitchen \
#     --sdrn_anno_path=extractors/SDRN/infer_on_ruara_\
#     --use_senti_word_list \
#     --glove_dimension=300 \
#     --multi_proc_dep_parsing \
#     --num_workers_mp_dep=16

export TMPDIR=./temp

python src/annotate.py \
    --path=data/amazon/$1\
    --sdrn_anno_path=extractors/SDRN/data/senti_term_$1_merged.pkl\
    --use_senti_word_list \
    --glove_dimension=300 \
    --num_senti_terms_per_pol=500
    # --do_compute_pmi \
    # --multi_proc_dep_parsing \
    # --num_workers_mp_dep=32 \