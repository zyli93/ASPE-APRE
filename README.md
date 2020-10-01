# ruara
Rating prediction with Unsupervised Aspect-level Review Analysis (RUARA)

## The TODO list

- [ ] implement user review aggregation and item review aggregation for Amazon
- [ ] implement user review aggregation and item review aggregation for Yelp
- [ ] add dependency info for nlp toolkits such as nltk and spaCy
- [ ] check if the strings need better processing?
- [ ] filter out too long or too short reviews
- [ ] pmi compute error in TagGCN because `combinations()` doesn't consider order

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
- `NLTK`
- `spaCy`

### Preprocessing

To preprocess the data, run the `preprocessing.py` to do the job. Use the following command to see the help info.
```bash
python src/preprocessing.py -h
```

Detailed instruction for each flag
```text
usage: preprocess.py [-h] --dataset DATASET
                     [--test_split_ratio TEST_SPLIT_RATIO] [--k_core K_CORE]
                     [--amazon_subset AMAZON_SUBSET]
                     [--yelp_min_cat_num YELP_MIN_CAT_NUM]
                     [--yelp_city YELP_CITY]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset: yelp, amazon, goodreads
  --test_split_ratio TEST_SPLIT_RATIO
                        Ratio of test split to main dataset. Default=0.05.
  --k_core K_CORE       The number of cores of the dataset. Default=5.
  --amazon_subset AMAZON_SUBSET
                        [Amazon-only] Subset name of Amazon dataset
  --yelp_min_cat_num YELP_MIN_CAT_NUM
                        [Yelp-only] Minimum number of category labels in the
                        filter set. Default=2.
  --yelp_city YELP_CITY
                        [Yelp-only] Subset city of Yelp dataset
```

###

### Run with Docker

