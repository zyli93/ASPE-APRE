# ruara
Rating prediction with Unsupervised Aspect-level Review Analysis (RUARA)

## The TODO list

- [ ] implement user review aggregation and item review aggregation for Amazon & Yelp (in annotate.py)
- [ ] add dependency info for nlp toolkits such as nltk and spaCy
- [ ] check if the strings need better processing?
- [x] filter out too long or too short reviews
- [ ] pmi compute error in TagGCN because `combinations()` doesn't consider order
- [x] preprocessing code cannot deal with text = `NaN`
- [ ] put in note that order by speed: 
    nothing > clean_str w/o spellcheck (sc) > clean_str w/ sc
- [ ] another package `autocorrect` works faster than SpellChecker
- [ ] maybe the original text should also be kept for Dependency Parsing.

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
    ```
- `spaCy`

python3 -m spacy download en_core_web_sm


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

optional arguments:
  -h, --help            show this help message and exit
  --path PATH           Path to the dataset goodreads
  --pmi_window_size PMI_WINDOW_SIZE
                        The window size of PMI cooccurance relations.
                        Default=3.
  --token_min_count TOKEN_MIN_COUNT
                        Minimum number of token occurences in corpus. Rare
                        tokens are discarded
```

Here's an example for parsing the _Digital Music_ dataset for Amazon.
```
python src/annotate.py --path=./data/amazon/digital_music
```


### Run with Docker

