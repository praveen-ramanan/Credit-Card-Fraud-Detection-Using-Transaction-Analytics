# Dataset

## Source

**Credit Card Transactions Fraud Detection Dataset**
Source: Kaggle, published by `kartik2112`, generated with the Sparkov Data
Generation tool (simulated transactions of ~1,000 US cardholders at 693
merchants, Jan 2019 - Dec 2020).
URL: `https://www.kaggle.com/datasets/kartik2112/fraud-detection`

This dataset was chosen over the more famous "ULB Credit Card Fraud"
dataset (`mlg-ulb/creditcardfraud`) deliberately: the ULB dataset's features
are anonymized PCA components (`V1`...`V28`), which makes it good for a pure
ML exercise but **useless for the Big Data analytics side of this
project** — there is no merchant, category, city, state, or customer
information to aggregate over in Hive/Spark/MapReduce. The Sparkov dataset
has real, named columns, which is what lets us do merchant-level,
category-level, location-level and customer-level analytics as required by
the syllabus.

## Files and where the project reads them

| File | Rows | Fraud | Fraud rate | Period | Role |
|---|---|---|---|---|---|
| `fraudTrain.csv` (351 MB) | 1,296,675 | 7,506 | 0.579% | 2019-01-01 00:00:18 - 2020-06-21 12:13:37 | training + validation |
| `fraudTest.csv` (150 MB) | 555,719 | 2,145 | 0.386% | 2020-06-21 12:14:25 - 2020-12-31 23:59:34 | final test only |

(Verified against the files: no missing values, no duplicate rows or
`trans_num`s, no `trans_num` shared between the files, identical column
lists, 14 categories in both, test states a subset of training states.)

The files are **read in place** — they are not copied into the repository.
`config/config.py` resolves them as:

```
BDA_TRAIN_CSV  (default: %USERPROFILE%\Downloads\archive (5)\fraudTrain.csv)
BDA_TEST_CSV   (default: %USERPROFILE%\Downloads\archive (5)\fraudTest.csv)
```

To use another location:

```powershell
$env:BDA_TRAIN_CSV = "E:\data\fraudTrain.csv"
$env:BDA_TEST_CSV  = "E:\data\fraudTest.csv"
```

There is **no fallback dataset**. If a file is missing or unreadable,
`clean_data.py` and the MapReduce driver stop with a message naming the
path and these variables.

### How to get it

1. Create a free Kaggle account if you don't have one.
2. Download the dataset archive from the page above (or
   `kagglehub.dataset_download("kartik2112/fraud-detection")`, as
   `notebooks/train_model_colab.ipynb` does) and extract
   `fraudTrain.csv` and `fraudTest.csv`.
3. Put them in the default folder above or set the two variables.
4. Run `.\scripts\run_pipeline.ps1`.

## Column schema

Both files have these 23 columns:

| Column | Type | Description | Model use |
|---|---|---|---|
| *(unnamed)* | int | Row index written by pandas | dropped on load |
| `trans_date_trans_time` | datetime | Transaction timestamp | derived: hour, weekend, night; chronological validation split |
| `cc_num` | int | Card number (customer identifier) | excluded (identifier); masked in outputs |
| `merchant` | string | Merchant name; the generator prefixes every name with `fraud_`, which cleaning strips (it is not a label) | excluded (693 values, memorisation) |
| `category` | string | Transaction category (e.g. `grocery_pos`, `shopping_net`) | one-hot |
| `amt` | float | Transaction amount (USD) | numeric; `is_high_value` flag (>= $500) |
| `first`, `last` | string | Customer first/last name | excluded (identifier) |
| `gender` | string | Customer gender (`M`/`F`) | one-hot |
| `street` | string | Customer street address | excluded (identifier) |
| `city`, `state`, `zip` | string/int | Customer address | `state` one-hot (51 incl. DC); `city`, `zip` excluded |
| `lat`, `long` | float | Customer location | derived: customer-merchant distance |
| `city_pop` | int | Population of customer's city | numeric |
| `job` | string | Customer occupation | excluded (494 values) |
| `dob` | date | Customer date of birth | derived: age at transaction time |
| `trans_num` | string | Unique transaction ID | excluded (identifier); kept for alerts/outputs |
| `unix_time` | int | Generator's Unix time (offset from the timestamp; not used) | excluded |
| `merch_lat`, `merch_long` | float | Merchant location | derived: customer-merchant distance |
| `is_fraud` | int (0/1) | **Target label** — 1 = fraudulent transaction | label only, never a feature |

## Processed data

`src/preprocessing/clean_data.py` cleans each file row by row. It drops
the index column, deduplicates on `trans_num`, parses dates, handles
missing values, strips the merchant prefix and derives the analytics
features. It writes both files to
`data/processed/transactions_clean.parquet` (1,852,394 rows, ~148 MB) with
a `split` column (`train` / `test`). Nothing is learned during cleaning,
so keeping the splits in one file cannot leak test information. All
learned steps use `split == "train"` rows only.

## Class imbalance

The target is **heavily imbalanced** (0.58% fraud in training, 0.39% in
test), which is realistic for card-fraud data and is exactly why this
project treats precision/recall/F1/PR-AUC as primary metrics instead of
accuracy (see `README.md` → Evaluation Metrics, and
`docs/viva_questions.md`).
