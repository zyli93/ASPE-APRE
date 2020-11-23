# ruara
Rating prediction with Unsupervised Aspect-level Review Analysis (RUARA)

## The TODO list

- [ ] implement user review aggregation and item review aggregation for Amazon & Yelp (in annotate.py)
- [x] add dependency info for nlp toolkits such as nltk and spaCy
- [x] check if the strings need better processing?
- [x] filter out too long or too short reviews
- [ ] pmi compute error in TagGCN because `combinations()` doesn't consider order
- [x] preprocessing code cannot deal with text = `NaN`
- [x] put in note that order by speed: 
    nothing > clean_str w/o spellcheck (sc) > clean_str w/ sc
- [x] another package `autocorrect` works faster than SpellChecker
- [x] maybe the original text should also be kept for Dependency Parsing.
- [x] add discussion about window size
- [x] add example of result of PMI annotation
- [x] add notes for deleting RB and using JJ only
- [ ] add gensim with glove embedding

## Data

We use the following datasets: Amazon, Yelp, and Goodread.

### Where to find them?

* Amazon datasets can be found [here](https://nijianmo.github.io/amazon/index.html). Many thanks to the managing team!
* Yelp dataset can be found [here](https://www.yelp.com/dataset). Many thanks to Yelp for sharing the invaluable database!

### Unzip and rename.

#### Amazon
We used the 5-core version. The downloaded files are in the `.json.gz` extension. After decompressing, a json file will be obtained (e.g., `Office_Products_5.json.gz` to `Office_Products_5.json`). Please rename it to `office_products.json` and move it to `raw_datasets/amazon/office_products.json` since the preprocessing pipeline will locate the files to process by names. In this case, the `office_products` should be given to the `--amazon_subset` flag.

#### Yelp
We used the plain version of Yelp dataset and generate its 5-core


## Run Ruara

### Install dependencies

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


### Preprocessing

To preprocess the data, run the `preprocessing.py` to do the job. Use the following command to see the help info.
```bash
python src/preprocessing.py -h
```

Detailed instruction for each flag
```text
usage: preprocess.py [-h] --dataset DATASET
                     [--test_split_ratio TEST_SPLIT_RATIO] [--k_core K_CORE]
                     [--min_review_len MIN_REVIEW_LEN]
                     [--amazon_subset AMAZON_SUBSET]
                     [--yelp_min_cat_num YELP_MIN_CAT_NUM]
                     [--yelp_city YELP_CITY]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset: yelp, amazon, goodreads
  --test_split_ratio TEST_SPLIT_RATIO
                        Ratio of test split to main dataset. Default=0.05.
  --k_core K_CORE       The number of cores of the dataset. Default=5.
  --min_review_len MIN_REVIEW_LEN
                        Minimum length of the reviews. Default=20.
  --amazon_subset AMAZON_SUBSET
                        [Amazon-only] Subset name of Amazon dataset
  --yelp_min_cat_num YELP_MIN_CAT_NUM
                        [Yelp-only] Minimum number of category labels in the
                        filter set. Default=2.
  --yelp_city YELP_CITY
                        [Yelp-only] Subset city of Yelp dataset
```

Here's an example for parsing the _Digital Music_ dataset for Amazon.
```
python src/preprocess.py --dataset=amazon --amazon_subset=digital_music
```

**Note**: We noticed that there exist words which are misspelled and can damage the PMI for aspect words. 

### Unsupervised aspect annotation

The `annotate.py` is in charge of annotating the aspects and corresponding opinion. Detailed instructions are below.

```text
usage: annotate.py [-h] --path PATH [--pmi_window_size PMI_WINDOW_SIZE]
                   [--token_min_count TOKEN_MIN_COUNT]
                   [--aspect_candidate_quota_per_seed ASPECT_CANDIDATE_QUOTA_PER_SEED]

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path to the dataset goodreads
  --pmi_window_size PMI_WINDOW_SIZE
                        The window size of PMI cooccurance relations.
                        Default=3.
  --token_min_count TOKEN_MIN_COUNT
                        Minimum token occurences in corpus. Rare tokens are
                        discarded. Default=20.
  --aspect_candidate_quota_per_seed ASPECT_CANDIDATE_QUOTA_PER_SEED
                        Number of candidate aspect opinion word to extract per
                        seed. Default=3.
```

Here's an example for parsing the _Digital Music_ dataset for Amazon.
```
python src/annotate.py --path=./data/amazon/digital_music
```

Here's the annotation pipeline in `annotate.py`:

1. Load dataset and hard-coded files: POS tags as filters of aspect sentiment (`fine_grained.pos.json`) and seed words for sentiment words (`seed_words.json`).
2. Compute PMI of existing word pairs in the corpus.
3. Run POS tagging provided by NLTK and take the most popular POS as a words POS.
4. Compute modifier words' polarity using the method in SKEP. Only `JJ` are considered modifiers. Refer to `./configs/fine_grained.pos.json`.
5. Remove the tokens that aren't valid words.

Here are the top 15 sentiment words extracted by our PMI-based method. We can see most words are quick possitive. But two types of outliers exsit in this list. One, non-words (i.e., "ur"). Two, negative words (i.e., "reckless"). The existence of negative words is strongly due to the certain selection of seeds.
```
word            ,POS ,polarity                ,Mean Positive PMI      ,Mean Negative PMI
clear           ,JJ  ,0.5840987462540387      ,-0.19650273081066164   ,-0.7806014770647003
great           ,JJ  ,0.5489023712339772      ,-0.20450911693587362   ,-0.7534114881698508
believable      ,JJ  ,0.486272530389566       ,0.486272530389566      ,0.0
good            ,JJ  ,0.48500002838557305     ,0.022995079620155488   ,-0.46200494876541753
excellent       ,JJ  ,0.35263876072657313     ,-0.3218579276588113    ,-0.6744966883853845
ur              ,JJ  ,0.34734304468895105     ,0.36018580798898425    ,0.012842763300033195
unbelievable    ,JJ  ,0.3467458825426481      ,0.2899629017690204     ,-0.056782980773627714
mushy           ,JJ  ,0.3212031175587027      ,0.42669711225563556    ,0.10549399469693288
reckless        ,JJ  ,0.26206888751325863     ,0.3109020229148935     ,0.04883313540163484
amazing         ,JJ  ,0.2589900139290045      ,-0.37184520235735957   ,-0.6308352162863641
beautiful       ,JJ  ,0.25721300991552554     ,-0.5456226782054189    ,-0.8028356881209444
nice            ,JJ  ,0.2568078689974637      ,-0.20424132281965768   ,-0.4610491918171214
much            ,JJ  ,0.23952869087514955     ,-0.23410520554168252   ,-0.47363389641683207
awesome         ,JJ  ,0.22824133572268956     ,-0.2510182876984033    ,-0.47925962342109285
```


**Notes and Discussions**:
 1. If `--pmi_window_size` is increased, then `--token_min_count` should be increased as well to remove rare tokens. Usually, these rare tokens come from 
  misspelling of users of review sites.

### Aspect and Opinion Extractor with ML models

#### SDRN

We managed to run `SDRN`, a Bert-based model for aspect and sentiment co-extraction. We carefully record the procedure for reproduction.

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
    [in_SDRN_dir]$ bash train_sdrn.sh [dataset] [No. of epochs]
    # e.g.
    [in_SDRN_dir]$ bash train_sdrn.sh 2014Res 5
    ```
    Below are the number of epochs I used to train SDRN.

    |Name  | 2014Lap | 2014Res | 2015Res |
    |------|---------|---------|---------|
    |#. Ep | 
  
8. Massage our data into SDRN-compatible format and run inference (annotation). We wrote a Python script to do the work using the preprocessed Amazon data. Note that it takes a long time to run.
    ```bash
    [in_SDRN_dir]$ bash scrips/run_inference.sh
    ```
    Detailed parameters within `run_inference.sh`.
    ```bash
    python ruara_evaluate.py [do_process] [training data] [annotate subset] [head]
    ```
    The semantic of parameters:
    - to_process: True/False, whether to rerun formatting Amazon data to SDRN data
    - training_set: The dataset that trains the SDRN model
    - annotate subset: The Amazon subset to process
    - head: (positive) number of top lines of Amazon data to process; 
            (negative) process the whole dataset.
9. Parse the output annotation file. Please run the following command
    ```bash
    python parse_output.py [training set] [annotate subset]
    ```
    The definition of the parameters are the same as point 8. The output will be 
    1. `./data/anno_[subset]/aspect_terms.pkl`: the aspect terms list pickle.
    2. `./data/anno_[subset]/sentiment_terms.pkl`: the sentiment terms list pickle.


#### RINANTE

Useful links:

1. Two ways of saving models in TF
    1. [Article 1](https://www.easy-tensorflow.com/tf-tutorials/basics/save-and-restore)
    2. [Article 2](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)


Todo items:

- [x] make sure the model setting is consistent with the parameter setting as the paper.
- [ ] write code to save model
- [ ] write code to run evaluation
- [ ] find the best number of iterations


### Run with Docker

## Appendix
#### SDRN

Modifications made to run `SDRN` in the modern world (PyTorch==1.15).

8. Some useful links:
   1. [Doc](https://pypi.org/project/pytorch-pretrained-bert/) for `pytorch_pretrained_bert`.
   2. [Doc](https://huggingface.co/transformers/model_doc/bert.html) from hugging face lastest version of BERT.
   3. [Homepage](https://github.com/huggingface/transformers) of `transformers` of hugging face. Detailed [Doc](https://huggingface.co/transformers/main_classes/optimizer_schedules.html) for Bert optimizations.
   4. Later version of `modeling.py` [implementation](https://github.com/cedrickchee/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling.py). As you can tell, the attributes of `BertLayerNorm` have been changed to `weight` and `bias` rather than `gamma` and `beta`.
   5. Source [code](https://huggingface.co/transformers/v2.0.0/_modules/transformers/modeling_bert.html) for hugging face bert. Unfortunately, this time we didn't use the most advanced version of it (:cry:). [Here](https://github.com/naver/sqlova/issues/1) is the GitHub issue about it.
   6. Of course, the GitHub [homepage](https://github.com/google-research/bert) of Google-research's Bert.

