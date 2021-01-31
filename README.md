# ASPE + APRE
Aspect-Sentiment Pair Extractor (ASPE) and Attention-Property-aware Rating Estimator (APRE)

This document introduces how to reproduce the experiments of ASPE + APRE.
## Data and External Resources

We use the following datasets: Amazon
Two external resources are also required: GloVe pretrained word vectors and BERT pretrained parameters.

### Where to find them?

* Amazon datasets can be found [here](https://nijianmo.github.io/amazon/index.html). Many thanks to the managing team!
* GloVe: See [GloVe](###GloVe) for details.

### Unzip and rename.

#### Amazon
We used the 5-core version. The downloaded files are in the `.json.gz` extension. After decompressing, a json file will be obtained (e.g., `Office_Products_5.json.gz` to `Office_Products_5.json`). Please rename it to `office_products.json` and move it to `raw_datasets/amazon/office_products.json` since the preprocessing pipeline will locate the files to process by names. In this case, the `office_products` should be given to the `--amazon_subset` flag.

### GloVe
GloVe is a pre-trained embedding vector popularly used for a wide range of NLP tasks.
GloVe can be downloaded from [here](https://nlp.stanford.edu/projects/glove/). 
And after downloading, please place it in `./glove` and then run 
```bash
python src/reformat_glove.py
```


## Prerequisites

### (Prerequisite 1) Install dependencies

#### ALL IN ONE!
Please run this single command to download and install all Python packages, NLTK packages, and spaCy dependencies.
```bash
bash scripts/get_prereq_ready.sh
```
You will see so many lines are being printed out. If no errors, go ahead to [Data Preprocessing](#Preprocessing). If you are interested what have been installed to your Python environment, please finish this section.

#### Dependencies for Python and PyTorch
Ruara is implement by Python and PyTorch. For a one-click complete installment of all Python dependencies. Please run
```bash
pip install -r requirements.txt
```

#### Dependencies for NLP toolkits
- `nltk`:
  - install `nltk` with `pip`.
  - download `nltk` supporting corpus.
    ```python
    >>> import nltk
    >>> nltk.download('punkt')
    >>> nltk.download('averaged_perceptron_tagger')
    >>> nltk.download('words')
    ```
- `spaCy`:
  - install pos tagging package in shell
    ```bash
    python3 -m spacy download en_core_web_sm
    ```


### (Prerequisite 2) Preprocessing

To preprocess the data, run the `preprocessing.py` to do the job. Use the following command to see the help info.
```bash
python src/preprocessing.py -h
```

Detailed instruction for each flag
```text
usage: preprocess.py [-h] [--test_split_ratio TEST_SPLIT_RATIO]
                     [--k_core K_CORE] [--min_review_len MIN_REVIEW_LEN]
                     [--use_spell_check] [--amazon_subset AMAZON_SUBSET]

optional arguments:
  -h, --help            show this help message and exit
  --test_split_ratio TEST_SPLIT_RATIO
                        Ratio of test split to main dataset. Default=0.1.
  --k_core K_CORE       The number of cores of the dataset. Default=5.
  --min_review_len MIN_REVIEW_LEN
                        Minimum num of words of the reviews. Default=5.
  --use_spell_check     Whether to use spell check and correction. Turning
                        this on will SLOW DOWN the process.
  --amazon_subset AMAZON_SUBSET
                        [Amazon-only] Subset name of Amazon datasett
```

Here's an example for parsing the _Digital Music_ dataset for Amazon.
```
python src/preprocess.py --amazon_subset=digital_music
```

**Note**: We noticed that there exist words which are misspelled and can damage the PMI for aspect words. 

## ASPE

After preprocessing, we will arrive at the ASPE part. This work is done by the following steps:
1. Annotate text with the NN-based model (SDNR in our example)
2. Prepare the Sentiment lexicon
3. Use `annotate.py` to build PMI and merge the three sets to build $ST$.
4. Use `extract.py` to build AS-candidates and eventually AS-pairs.
### NN-based aspect-sentiment term co-extraction.
Please see instructions for [SDRN](#NN-based-Aspect-and-Opinion-Extractor).

### and Lexicon-based extraction
The lexicon has been included in this package. Please refer to `./configs/opinion-lexicon-English`. Well, you don't need to do anything for this step.

### Unsupervised aspect annotation

We run `annotate.py` to build PMI and merge the three sets to build $ST$. In detail, the `annotate.py` is in charge of the following tasks:
1. Compute P(w_i, w_j) and P(w_i).
2. Compute PMI for each word in the corpus.
3. Load sentiment terms extracted by the NN-based and Lexicon-based methods.
4. Use dependency parsing technique to find aspect-sentiment pair candidates. (AS-cand)
5. Save extracted AS-cand.

[HELP] Detailed instructions are below.
```text
usage: annotate.py [-h] --path PATH --sdrn_anno_path SDRN_ANNO_PATH
                   [--pmi_window_size PMI_WINDOW_SIZE]
                   [--token_min_count TOKEN_MIN_COUNT]
                   [--num_senti_terms_per_pol NUM_SENTI_TERMS_PER_POL]
                   [--use_senti_word_list] [--glove_dimension GLOVE_DIMENSION]
                   [--multi_proc_dep_parsing]
                   [--num_workers_mp_dep NUM_WORKERS_MP_DEP]
                   [--do_compute_pmi]

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path to the dataset.
  --sdrn_anno_path SDRN_ANNO_PATH
                        Path to SDRN annotation results in `.txt`
  --pmi_window_size PMI_WINDOW_SIZE
                        The window size of PMI cooccurance relations.
                        Default=5.
  --token_min_count TOKEN_MIN_COUNT
                        Minimum token occurences in corpus. Rare tokens are
                        discarded. Default=20.
  --num_senti_terms_per_pol NUM_SENTI_TERMS_PER_POL
                        Number of sentiment terms per seed. Default=300.
  --use_senti_word_list
                        If used, sentiment word table will be used as well.
  --glove_dimension GLOVE_DIMENSION
                        The dimension of glove to use in the PMI parsing.
                        Default=100.
  --multi_proc_dep_parsing
                        If used, parallel processing of dependency parsing
                        will be enabled.
  --num_workers_mp_dep NUM_WORKERS_MP_DEP
                        Number of workers to be spinned off for multiproc dep
                        parsing.
  --do_compute_pmi      Whether to redo pmi computation
```

[EXAMPLE] Here's an example for parsing the _Digital Music_ dataset for Amazon.
```
bash scripts/run_annotate.sh digital_music
```
But please pay attention to the `--do_compute_pmi` flag. When you first run this model, please enable this flag as it will execute the compute PMI for you. You will see below that it save the PMI terms after it run once so that next time you don't waste time running it again.

[OUTPUT] Most important results `annotate.py` generates:
1. `train_data_dep.pkl`: the pickle file of dataframe with a column containing `spacy.doc` objects.
2. `pmi_senti_terms.pkl`: sentiment terms extracted by PMI methods.

### Filter invalid AS-cand using $ST$ to build AS-pair
We use `extract.py` to filter useful aspects and convert aspects to index.

[HELP] Below is the help information
```text
usage: extract.py [-h] --data_path DATA_PATH --count_threshold COUNT_THRESHOLD
                  [--run_mapping]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the dataset.
  --count_threshold COUNT_THRESHOLD
                        Threshold of the count.
  --run_mapping         If off, only get aspairs but do not work on df. For
                        viewing use, cheaper.
```
Please check out the `count_threshold` of each dataset from the paper. `--run_mapping` is a flag to turn on/off the "real" heavy work. If used, the actual filtering is on. Otherwise, it only do the count-thresholding to remove the infrequent aspects.

[EXAMPLE] Please find example for building AS-pairs for `extract.py` below
```bash
bash scripts/run_extract.sh
```
Please note that this file takes care of all seven datasets. Make sure you want all of them, or the unwanted ones are commented out.

[OUTPUT] Most important results `extract.py` generates:
1. `train_data_aspairs.pkl`: all information needed for training.
2. `aspcat_counter.pkl`, `aspcat2idx,pkl`, `idx2aspcat.pkl`, and `asp2aspcat.pkl`: some useful pickles that stores the aspect to ID and ID to aspects. (Implementation related only)

### Prepare for Training
We are almost there!!! In order to speed up the training, we tokenize the text beforehand. We use `postprocess.py` to prepare the data for training. We understand some work can be done ahead of time so that it can save sometime. Especially for finding the locations of the sentiment terms.

[HELP] Please find the useful information here.
```
usage: postprocess.py [-h] --data_path DATA_PATH --num_aspects NUM_ASPECTS
                      [--max_pad_length MAX_PAD_LENGTH]
                      [--num_workers NUM_WORKERS] [--build_nbr_graph]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Path to the dataset.
  --num_aspects NUM_ASPECTS
                        Number of aspect categories in total
  --max_pad_length MAX_PAD_LENGTH
                        Max length of padding. Default=100.
  --num_workers NUM_WORKERS
                        Number of multithread workers
  --n_partition N_PARTITION
                        Number of partitions for multiprocessing.
  --build_nbr_graph     Whether to build neighborhood graph.
```
Number of aspects will be printed from `extract.py`. 

[Example] An example for digital music is 
```
python src/postprocess.py --data_path=./data/amazon/digital_music --num_aspects 296
```
Or in a bash file:
```bash
bash scripts/run_postprocess.sh
```
Again, make sure you want all of them, or the unwanted ones are commented out. Another tip for the execution: for larger datasets, the only way to make it runable is to set both `--num_workers` and `--n_partition` to be 1.

[OUTPUT] Most important results `extract.py` generates:
### Training

### NN-based Aspect and Opinion Extractor 
We managed to run `SDRN`, a Bert-based model for aspect and sentiment co-extraction. We carefully record the procedure for reproduction. __NOTE__: If trained models are already available, please jump to step 8!

1. Clone the repo from GitHub: https://github.com/NKU-IIPLab/SDRN. Many thanks for sharing the code!
2. Install PyTorch 0.4.1.
3. Install the package `pytorch_pretrained_bert`. (I know it might be outdated by the `SDRN` implementation was actually based on it.)
4. Download Bert checkpoint and config files from [here](https://github.com/ethanjperez/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py). Note that the `.bin` (checkpoint) and the `.json` (config) have to match! Add the locations of them to `main.py`.
5. The `modeling.py` didn't come with the original repo of `SDRN`. Please find it from [here](https://github.com/naver/sqlova/blob/master/bert/modeling.py).
6. Do some changes following below instructions:
   1. Make some changes in `main.py` and `opinionMining.py`, all of which is related to `from bert.modeling`.
      ```python
      # main.py
      from bert.modeling import BertConfig
      from bert.optimization import BERTAdam
      ```
      to 
      ```python
      # main.py
      from pytorch_pretrained_bert.modeling import BertConfig
      from pytorch_pretrained_bert.optimization import BertAdam as BERTAdam
      ```
      And
      ```python
      # opinionMining.py
      from bert.modeling import BertModel, BERTLayerNorm
      ```
      to 
      ```python
      # opinionMinding.py
      from modeling import BertModel, BERTLayerNorm
      ```
      assuming that `modeling.py` has been put to the right position.
   2. There's was a bug in `_load_from_state_dict` in the repo, can be fixed easily.
   3. Another problem that I encountered was the `.gamma` and `.beta` of `BertLayerNorm`. Laster fixed it by finding the original `modeling.py`.
7. Train the model with the given datasets: 2014Lap.pt, 2014Res.pt, 2015Res.pt. Using the folliwing script:
    ```bash
    # run one corpus by corpus
    [in SDRN dir]$ bash scripts/train_sdrn.sh [dataset] [No. of epochs]
    # run everything
    [in SDRN dir]$ bash scripts/train_all.sh
    # e.g.
    [in SDRN dir]$ bash scripts/train_sdrn.sh 2014Res 5
    ```
    Below are the number of epochs I used to train SDRN.

    |Name  | 2014Lap | 2014Res | 2015Res |
    |------|---------|---------|---------|
    |#. Ep |  5      |  10     | 8       |
  
8. [If trained SDRN models are **available**, start right from this step!] Massage our data into SDRN-compatible format and run inference (annotation). We wrote a Python script to do the work using the preprocessed Amazon data. Note that it takes a long time to run.
    ```bash
    [in SDRN dir]$ bash scripts/run_inference.sh
    ```
    Detailed parameters within `run_inference.sh`.
    ```bash
    [in SDRN dir]$ python ruara_evaluate.py [do_process] [training data] [annotate subset] [head] [gpu_id]
    ```
    The semantic of parameters:
    - to_process: True/False, whether to rerun formatting Amazon data to SDRN data
    - training_set: The dataset that trains the SDRN model
    - annotate subset: The Amazon subset to process
    - head: Positive number: number of top lines of Amazon data to process; 
            Negative number: process the whole dataset.
    - gpu_id: the GPU to use.
9. Parse the output annotation file. Please run the following command to merge sentiments extracted from the three SDRN versions.
    ```bash
    # Change the dataset names in this file correspondingly!
    [in SDRN dir] $ bash scripts/parse_output.sh
    ```
    Details:
    ```bash
    [in SDRN dir]$ python parse_output.py [task] [training set] [annotate subset]
    ```
    The definition of the parameters are the same as those in point 8 except `task` which can be `parse` and `merge`. 
    
    For `parse`, the output will be 
    1. `./data/anno_[train_set]_[subset]/aspect_terms.pkl`: the aspect terms list pickle.
    2. `./data/anno_[train_set]_[subset]/sentiment_terms.pkl`: the sentiment terms list pickle.
    For example, `./data/anno_2014Lap_digital_music/aspect_terms.pkl` saves all aspect terms extracted by a 2014Lap-trained SDRN model for the dataset `digital_music`.

    For `merge`, take output from the above step and merge the sentiment terms sets. Produce results to `./data/senti_term_[subset]_merged.pkl`.
10. Until here, the SDRN term extraction is done. The generated file can be picked up by `annotate.py`.

## Appendix
#### SDRN

Modifications made to run `SDRN` in the modern world (PyTorch==1.15).

8. Some useful links:
   1. [Doc](https://pypi.org/project/pytorch-pretrained-bert/) for `pytorch_pretrained_bert`.
   2. [Doc](https://huggingface.co/transformers/model_doc/bert.html) from hugging face lastest version of BERT.
   3. [Homepage](https://github.com/huggingface/transformers) of `transformers` of hugging face. Detailed [Doc](https://huggingface.co/transformers/main_classes/optimizer_schedules.html) for Bert optimizations.
   4. Later version of `modeling.py` [implementation](https://github.com/cedrickchee/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py). As you can tell, the attributes of `BertLayerNorm` have been changed to `weight` and `bias` rather than `gamma` and `beta`.
   5. Source [code](https://huggingface.co/transformers/v2.0.0/_modules/transformers/modeling_bert.html) for hugging face bert. Unfortunately, this time we didn't use the most advanced version of it :cry:. [Here](https://github.com/naver/sqlova/issues/1) is the GitHub issue about it.
   6. Of course, the GitHub [homepage](https://github.com/google-research/bert) of Google-research's Bert.

