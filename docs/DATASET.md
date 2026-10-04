# Dataset

## Recommended real-world dataset

**Credit Card Transactions Fraud Detection Dataset**
Source: Kaggle, published by `kartik2112`, generated with the Sparkov Data
Generation tool.
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

### How to get it

1. Create a free Kaggle account if you don't have one.
2. Download `fraudTrain.csv` and `fraudTest.csv` from the dataset page
   above (requires login; there is no anonymous direct-download URL).
3. Combine and rename to match this project's expected file:

   ```powershell
   # from a Python shell, or adapt with pandas
   python -c "import pandas as pd; a=pd.read_csv('fraudTrain.csv'); b=pd.read_csv('fraudTest.csv'); pd.concat([a,b]).to_csv('data/raw/transactions.csv', index=False)"
   ```

4. Confirm the column names match the schema below (Kaggle's file has an
   extra unnamed index column and identical field names otherwise — no
   further changes needed).

No other code changes are required — every module reads
`data/raw/transactions.csv` via `config/config.py` and expects exactly this
schema.

## Synthetic demo dataset (used for development out of the box)

Because downloading the Kaggle dataset requires manual authentication, this
project ships `scripts/generate_synthetic_data.py`, which generates a
dataset with the **identical schema** below. It is used to build, test, and
demo the entire pipeline immediately. It is clearly labeled as synthetic
everywhere it is used (README, reports, viva notes) — it is not presented
as real-world data.

Fraud is injected in the synthetic generator using simple, documented
rules (odd transaction hour, high-risk merchant category, large
customer-merchant distance) so the classifier has genuine, learnable
signal rather than random noise — this mirrors (in simplified form) the
kind of behavioral signal real fraud-detection systems rely on.

Regenerate it any time with:

```powershell
.\venv\Scripts\python.exe scripts\generate_synthetic_data.py
```

Row count is controlled by the `BDA_SYNTH_ROWS` environment variable
(default 150,000).

## Column schema

| Column | Type | Description |
|---|---|---|
| `index` | int | Row index |
| `trans_date_trans_time` | datetime string | Transaction timestamp |
| `cc_num` | int | Card number (customer identifier proxy) |
| `merchant` | string | Merchant name |
| `category` | string | Merchant/transaction category (e.g. `grocery_pos`, `shopping_net`) |
| `amt` | float | Transaction amount |
| `first`, `last` | string | Customer first/last name |
| `gender` | string | Customer gender (`M`/`F`) |
| `city`, `state`, `zip` | string/int | Customer address |
| `lat`, `long` | float | Customer location |
| `city_pop` | int | Population of customer's city |
| `job` | string | Customer occupation |
| `dob` | date | Customer date of birth |
| `trans_num` | string | Unique transaction ID |
| `unix_time` | int | Transaction time as Unix timestamp |
| `merch_lat`, `merch_long` | float | Merchant location |
| `is_fraud` | int (0/1) | **Target label** — 1 = fraudulent transaction |

## Class imbalance

Both the real Sparkov dataset and this project's synthetic generator
produce a **heavily imbalanced** target (well under 1% fraud), which is
realistic for card-fraud data and is exactly why this project treats
precision/recall/F1/PR-AUC as primary metrics instead of accuracy (see
`docs/README.md` → Evaluation Metrics, and `docs/viva_questions.md`).
