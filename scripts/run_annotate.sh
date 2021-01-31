export TMPDIR=./temp

python src/annotate.py \
    --path=data/amazon/$1\
    --sdrn_anno_path=extractors/SDRN/data/senti_term_$1_merged.pkl\
    --use_senti_word_list \
    --glove_dimension=300 \
    --num_senti_terms_per_pol=400\
    --do_compute_pmi