# Banking – Credit Card Fraud Detection Using Transaction Analytics
## Semester V – Big Data Architecture
## DOC 2 / Review 2 Progress Document

*Last updated: 2026-10-28, after migrating from the synthetic demo data to the real Kaggle dataset, based on the current source files, generated outputs, and a full test run.*

---

### 1. Project Overview

An end-to-end Big Data pipeline for credit-card transaction analytics and fraud detection: Kaggle `fraudTrain.csv` / `fraudTest.csv` ingestion → local HDFS-path storage → pandas cleaning → MapReduce aggregation → Hive-equivalent SQL analytics (via Spark SQL) → PySpark DataFrame processing → supervised fraud classification → MongoDB-style alert storage → matplotlib visualization. Runs entirely locally on Windows with no paid services; real-cluster instructions are documented but not all have been executed against real infrastructure (see per-component status below).

---

### 2. Current Implementation Status

| Component | Status | Actual Files | Evidence/Output | Manual Setup |
|---|---|---|---|---|
| Dataset | COMPLETED (real Kaggle data) | `config/config.py` (`BDA_TRAIN_CSV` / `BDA_TEST_CSV`) | `fraudTrain.csv` 1,296,675 rows + `fraudTest.csv` 555,719 rows, read in place from Downloads | Kaggle download (login) |
| Data preprocessing | COMPLETED | `src/preprocessing/clean_data.py` | `data/processed/transactions_clean.parquet` (1,852,394 rows with a `split` column, verified) | None |
| EDA | PARTIALLY COMPLETED | none dedicated; overlap in `clean_data.py` (missing-value report) and `visualization/plots.py` | Console output only; no notebook/report | No dedicated EDA notebook exists despite `notebooks/` folder being present (empty except `.gitkeep`) |
| Hadoop / HDFS | PARTIALLY COMPLETED | `docs/HDFS_SETUP.md`, `scripts/hdfs_ingest.sh` | Local filesystem stand-in only (`data/raw/`, `data/processed/`) | Real Hadoop/HDFS not installed; ingest script never executed against a real cluster |
| MapReduce | COMPLETED (local mode) | `src/mapreduce/{category_aggregation,customer_aggregation}/{mapper,reducer}.py`, `local_driver.py` | `results/mapreduce_output/*.tsv` — both jobs run and verified, incl. unit tests | Real Hadoop Streaming submission documented but not executed (no cluster available) |
| Hive | PARTIALLY COMPLETED | `src/hive/hql/{create_tables,analytics_queries}.hql` | `.hql` files are real, correct HiveQL but **have never been executed by an actual Hive interpreter** (none installed) | Real Hive install required to verify the `.hql` scripts themselves |
| Pig | NOT IMPLEMENTED | none | none — confirmed via full-repo search, zero references | Not started; no Pig scripts, no Pig install |
| NoSQL (MongoDB) | COMPLETED (local `mongomock` mode) | `src/nosql/mongo_client.py`, `store_analytics.py` | Verified run: 500 `fraud_alerts` docs, 14 `category_analytics` docs, aggregation pipeline query output | Real MongoDB/Atlas never connected to (no server installed); `mongomock` data does not persist between runs |
| Spark / PySpark | COMPLETED (local mode) | `src/spark/{spark_session,data_processing,hive_style_analytics}.py` | `results/spark_output/*.csv`, `results/hive_analytics/*.csv` (13 files) — both scripts run successfully on 1.85M rows | Requires JDK 21 (`JAVA_HOME` override) since system default JDK 26 crashes Spark 4.2, and a junction for the `(BDA)` parentheses in the path — both automated in `run_pipeline.ps1` |
| Fraud detection / ML | COMPLETED | `src/fraud_detection/{features,train_model,inference,verify_model}.py`, `notebooks/train_model_colab.ipynb` | `results/metrics/model_comparison.json` (validation + final test), `results/models/best_model.joblib` (trained locally), `results/fraud_detection/scored_test_transactions.parquet`. `verify_model.py` reproduces the fraudTest.csv metrics exactly | None |
| Dashboard | COMPLETED | `app.py`, `src/dashboard/{data,views}.py` | Streamlit app with 6 sections (overview, analytics, prediction form, model performance, MongoDB alerts, CSV scoring). Server start verified, and every page is rendered by `tests/test_dashboard.py` | None |
| Visualization | COMPLETED | `src/visualization/plots.py` | 9 PNGs in `results/figures/`, verified to exist and one spot-checked visually | None |
| Testing | PARTIALLY COMPLETED | `tests/test_{preprocessing,features,mapreduce,data_pipeline,inference,mongo_client,dashboard}.py` | 49/49 tests pass (`pytest`) | No automated tests for Spark. Training is checked through its outputs (`test_data_pipeline.py`, `verify_model.py`), not re-run by `pytest` |
| Documentation | COMPLETED | `README.md`, `docs/*.md` (7 files incl. this one) | All present and cross-referenced | None |

**Note on declared-but-unused dependencies**: `requirements.txt` lists `seaborn` and `imbalanced-learn`, but a repo-wide search confirms neither is actually imported anywhere in `src/`. `plotly` is used only by the Streamlit dashboard (`src/dashboard/views.py`); the static charts use matplotlib; class imbalance is handled via `class_weight`/`scale_pos_weight`, not `imblearn`. Not a functional problem, but worth knowing before claiming those libraries are "used."

**Note on project state**: the project is version-controlled with Git.

---

### 3. Dataset

- **Dataset used**: the real Kaggle "Credit Card Transactions Fraud Detection Dataset" (Sparkov-simulated, `kartik2112`). Both files are read in place from `config/config.py` paths (`BDA_TRAIN_CSV` / `BDA_TEST_CSV`). The former synthetic generator has been removed. `docs/DATASET.md` has details.
- **Dataset size**: `fraudTrain.csv` 1,296,675 rows (2019-01-01 to 2020-06-21) and `fraudTest.csv` 555,719 rows (2020-06-21 to 2020-12-31). Each has 23 columns: an unnamed index column (dropped), the 21 schema fields, and `street`.
- **Features**: `trans_date_trans_time`, `cc_num`, `merchant`, `category`, `amt`, `first`, `last`, `gender`, `city`, `state`, `zip`, `lat`, `long`, `city_pop`, `job`, `dob`, `trans_num`, `unix_time`, `merch_lat`, `merch_long`, plus derived columns from preprocessing (`txn_hour`, `txn_day_of_week`, `txn_month`, `is_weekend`, `is_night_txn`, `customer_age`, `customer_merchant_distance_km`, `is_high_value`) , plus `street` and `split` — 31 columns after cleaning (verified against the Parquet schema).
- **Target/fraud column**: `is_fraud` (binary, 0/1). Verified class balance: 7,506 / 1,296,675 = 0.579% (train), 2,145 / 555,719 = 0.386% (test).
- **Current preprocessing status**: COMPLETED. `src/preprocessing/clean_data.py` validates both files exist and are readable, drops the index column, dedupes on `trans_num` (0 duplicates found), handles missing values (none found), parses timestamps, strips the generator's `fraud_` merchant prefix, derives 8 analytics features, and writes one Parquet file (1,852,394 rows) in ~8 seconds.

---

### 4. Big Data Architecture

What is **actually implemented and executable today**:

```
Kaggle CSVs (read in place) → local files standing in for HDFS
      → pandas cleaning
      → [MapReduce: real Hadoop-Streaming-compatible mapper/reducer, run via a
         pure-Python local shuffle/sort driver — no Hadoop daemon involved]
      → [Spark: real PySpark 4.2 local[*] session — DataFrame ops + Spark SQL]
      → [Hive: .hql files exist as real HiveQL but are unexecuted; the
         reported "Hive" results were actually produced by Spark SQL]
      → scikit-learn/XGBoost fraud classifier
      → [NoSQL: mongomock in-memory store — no real MongoDB server involved]
      → matplotlib charts
```

No Hadoop daemon, YARN, real Hive metastore, or MongoDB server has been started in this environment at any point. Every "Big Data" component that claims to run currently runs in a **local, single-process substitute** that is code-compatible with the real tool (same mapper/reducer contract, same SparkSession API, same pymongo API), not against real distributed infrastructure. This distinction matters for Review 2 and should be stated plainly if asked.

---

### 5. Hadoop / HDFS

- **Current status**: PARTIALLY COMPLETED / REQUIRES MANUAL SETUP.
- **Files implemented**: `docs/HDFS_SETUP.md` (WSL2 + pseudo-distributed Hadoop instructions), `scripts/hdfs_ingest.sh` (real `hdfs dfs -put/-ls/-du/-cat` commands).
- **Commands/configuration**: `config/config.py` centralizes all storage paths so the same code works whether those paths are local or (with env/config changes) HDFS-backed; `create_tables.hql` already points at an HDFS location (`/user/bda/data/processed/transactions_clean`).
- **Evidence available**: only local-filesystem evidence — `data/raw/`, `data/processed/` populated and verified. `hdfs_ingest.sh` has never been run (no `hdfs` client on this machine — confirmed via `which hdfs`... not installed).
- **Remaining work**: install WSL2 + Hadoop (or another real cluster), run `scripts/hdfs_ingest.sh`, and capture real `hdfs dfs -ls`/`-du` output as evidence for the review if a real-cluster demo is required.

---

### 6. MapReduce

- **Current status**: COMPLETED (local execution mode, verified).
- **Mapper/reducer files**:
  - `src/mapreduce/category_aggregation/mapper.py` + `reducer.py`
  - `src/mapreduce/customer_aggregation/mapper.py` + `reducer.py`
  - `src/mapreduce/local_driver.py` (simulates Hadoop's shuffle/sort in pure Python; same mapper/reducer scripts are Hadoop-Streaming-submittable as-is)
- **What they calculate**: per category (or per anonymized customer): transaction count, total amount, fraud count, fraud amount, fraud rate.
- **Output/evidence**: `results/mapreduce_output/category_aggregation.tsv` (14 categories) and `customer_aggregation.tsv` (999 anonymized cards), over all 1,852,394 rows of both raw Kaggle files (one map task per file) — both verified present; `tests/test_mapreduce.py` runs the actual mapper/reducer subprocesses against controlled input and asserts correct aggregation and card-number anonymization (2/2 tests passing).
- **Remaining work**: execution against a real Hadoop Streaming job (`hadoop jar hadoop-streaming-*.jar ...`, documented in `docs/MAPREDUCE_SETUP.md`) has not been performed — no Hadoop cluster available in this environment.

---

### 7. Hive

- **Current status**: PARTIALLY COMPLETED. The `.hql` scripts are real, syntactically-intended-correct HiveQL, but **no Hive interpreter has ever executed them** — there is no Hive install on this machine, so this has not been verified to actually run.
- **Tables**: `create_tables.hql` — one external table `fraud_analytics.transactions` over the cleaned Parquet (`STORED AS PARQUET`).
- **Queries**: `analytics_queries.hql` — 11 queries (totals, fraud vs legit, fraud amount share, category fraud rate, highest-value transactions, high-value fraud rate, hourly pattern, state distribution, top customers, top merchants by fraud amount, train vs test file summary).
- **Results**: what actually produced results is a **parallel Spark SQL implementation** (`src/spark/hive_style_analytics.py`) that mirrors each `.hql` query's logic and was run successfully — outputs in `results/hive_analytics/*.csv` (11 files, verified present). These are labeled throughout the docs/README as Spark-SQL-executed, Hive-equivalent results, not as Hive itself having run.
- **Remaining work**: install Hive (see `docs/HIVE_SETUP.md`, WSL2-based) and run `hive -f src/hive/hql/create_tables.hql` / `analytics_queries.hql` for real, to actually validate the `.hql` files execute without syntax/semantic errors on a real Hive engine.

---

### 8. NoSQL

- **MongoDB/Cassandra used**: MongoDB (via `pymongo`), with automatic fallback to `mongomock` (in-memory, same API) when no real server is reachable. Cassandra was not implemented (a deliberate choice, documented and justified in `docs/NOSQL_SETUP.md`, not an oversight).
- **Current implementation**: `src/nosql/mongo_client.py` (connection + fallback logic), `src/nosql/store_analytics.py` (loads documents, runs example queries).
- **Schema**: `fraud_alerts` collection — one document per high-risk transaction with a nested `customer` and `merchant` sub-document, model probability, and flag reason. `category_analytics` collection — flat category-level aggregate documents.
- **Queries**: `find().sort().limit()` for top alerts; a MongoDB `$group`/`$sort` aggregation pipeline for alerts-by-category.
- **Evidence**: verified run: backend reported as `mongomock`, 500 `fraud_alerts` + 14 `category_analytics` documents loaded, example queries returned expected results.
- **Remaining work**: has never connected to a real MongoDB server or Atlas cluster — `mongomock` data is in-memory only and does not persist between runs, so there is currently no persistent NoSQL store backing this project. Installing MongoDB Community Server locally, or creating a free MongoDB Atlas cluster, would let this be demonstrated against real infrastructure.

---

### 9. Pig

- **Current status**: NOT IMPLEMENTED.
- **Pig scripts**: none exist. Confirmed via a full-repository search — zero files, zero references to Pig anywhere in `src/`, `docs/`, `config/`, or `scripts/`.
- **Operations performed**: none.
- **Evidence**: none.
- **Remaining work**: Pig is listed in the syllabus but was not part of the original implementation scope agreed at project start. If Review 2 requires a Pig component, this needs to be built from scratch: a Pig Latin script (`.pig`) performing at least one meaningful transformation/aggregation over the transaction data, plus local/cluster execution instructions consistent with the rest of this project's "local demo + real cluster docs" pattern. Not started.

---

### 10. Spark

- **Current status**: COMPLETED (local mode, verified working).
- **PySpark files**: `src/spark/spark_session.py` (shared `SparkSession` factory), `src/spark/data_processing.py` (DataFrame pipeline), `src/spark/hive_style_analytics.py` (SQL analytics).
- **DataFrame operations**: schema read/print, `.filter()`, `.groupBy().agg()` with `pyspark.sql.functions` (count/sum/avg), `.orderBy()`.
- **Spark SQL**: `createOrReplaceTempView()` + `spark.sql(...)` for both a state-level summary and the full 11-query Hive-equivalent set.
- **Results**: `results/spark_output/{category_summary,state_summary}.csv` and `results/hive_analytics/*.csv` (11 files) — all verified present and consistent with the MapReduce output (e.g. category fraud rates match between the MapReduce TSV and Spark SQL CSV).
- **Remaining work**: only ever run in `local[*]` mode; **never submitted to a real cluster** (`yarn`/`spark://...`) — `BDA_SPARK_MASTER` env var support exists in code but is unverified against real infrastructure. Also worth re-stating: this machine's default JDK (26) cannot run Spark 4.2 at all; every successful Spark run in this project depended on manually pointing `JAVA_HOME` at JDK 21.

---

### 11. Fraud Detection

- **Preprocessing**: handled upstream in `src/preprocessing/clean_data.py` (Section 3/5).
- **Feature engineering**: `src/fraud_detection/features.py` — numeric (amount, hour, customer age, customer-merchant distance, city population), boolean (weekend/night/high-value), one-hot encoded categoricals (category, state, gender). Identifiers (`cc_num`, `trans_num`, `unix_time`, names, street, zip), high-cardinality fields (merchant, city, job), raw coordinates, `txn_month` (time leakage across the train/test periods) and the label are excluded, enforced by an assertion. 75 total features (verified: `Feature matrix: 75 features` from the last training run).
- **Validation protocol**: chronological holdout inside `fraudTrain.csv`. Models are fit on the earliest 80% (1,037,340 rows) and validated on the latest 20% (259,335 rows, 1,538 fraud). The model is chosen by validation PR-AUC, and the threshold is the validation F1 optimum. `fraudTest.csv` is scored once, at the end.
- **Models**: Logistic Regression (scaled, `class_weight="balanced"`), Random Forest (`class_weight="balanced_subsample"`), XGBoost (`scale_pos_weight`) — all three trained by `src/fraud_detection/train_model.py` in ~65 s total.
- **Best model**: XGBoost, selected by highest validation PR-AUC (0.899), decision threshold 0.949.
- **Validation metrics** (latest 20% of fraudTrain.csv):

  | Model | ROC-AUC | PR-AUC | Precision@thr | Recall@thr | F1@thr |
  |---|---|---|---|---|---|
  | Logistic Regression | 0.945 | 0.198 | 0.304 | 0.560 | 0.394 |
  | Random Forest | 0.994 | 0.835 | 0.791 | 0.743 | 0.766 |
  | XGBoost | 0.998 | **0.899** | 0.878 | 0.803 | 0.839 |

- **Final test metrics** (fraudTest.csv, 555,719 rows / 2,145 fraud, XGBoost @ 0.949): ROC-AUC 0.9970, PR-AUC 0.8554, precision 0.813, recall 0.777, F1 0.794; TP=1,666, FP=384, FN=479, TN=553,190.
- **Current results**: real and reproducible (`RANDOM_STATE=42`) — taken directly from `results/metrics/model_comparison.json`, and re-verified by `verify_model.py`.

---

### 12. Visualization

All 9 charts confirmed present in `results/figures/`:

1. `01_fraud_vs_legit_count.png` — fraud vs legitimate counts (log scale)
2. `02_fraud_amount_distribution.png` — amount histogram, fraud vs legit
3. `03_transactions_over_time.png` — daily transaction/fraud volume
4. `04_fraud_rate_by_hour.png` — fraud rate by hour of day
5. `05_top_suspicious_categories.png` — fraud rate by category
6. `06_top_suspicious_merchants.png` — fraud rate by merchant (min. 20 txns)
7. `07_high_value_fraud_rate.png` — high-value vs normal fraud rate
8. `08_confusion_matrix.png` — selected model's confusion matrix on fraudTest.csv
9. `09_model_comparison.png` — validation precision/recall/F1/ROC-AUC/PR-AUC across all 3 models

All generated by `src/visualization/plots.py`, matplotlib only (no seaborn/plotly despite both being listed in `requirements.txt` — see Section 2 note).

---

### 13. Testing

| Test file | Tests | Status |
|---|---|---|
| `tests/test_preprocessing.py` | 8 (dedup/invalid amounts, missing-value fill, merchant prefix, unnamed index column, missing file, derived columns, haversine x2) | PASS |
| `tests/test_features.py` | 2 (feature matrix shape/types + label excluded, binary label) | PASS |
| `tests/test_mapreduce.py` | 2 (category aggregation, customer aggregation + anonymization) | PASS |
| `tests/test_data_pipeline.py` | 6 (Kaggle CSV header/labels x2, shared schema, split separation and time order, chronological validation split, validation-vs-test metric integrity) | PASS |
| `tests/test_inference.py` | 14 (scoring, encoding alignment, invalid inputs, mixed batch, schema/threshold of the bundle, stale-model rejection, missing columns) | PASS |
| `tests/test_mongo_client.py` | 5 (mongomock fallback policy) | PASS |
| `tests/test_dashboard.py` | 12 (every page renders, prediction form, invalid dates, alert query, train/test selector x2, no synthetic-data text) | PASS |

**49/49 tests passing**, confirmed via `pytest -q` → `49 passed`.

**Not covered by automated tests**: Spark DataFrame/SQL correctness (verified by running the scripts), and re-running training itself (its outputs are checked instead).

---

### 14. Syllabus Mapping

| Syllabus Topic | Project Implementation | Status |
|---|---|---|
| Big Data fundamentals | README §3–6, this document §4 | COMPLETED (conceptual + architectural) |
| Hadoop architecture | `docs/HDFS_SETUP.md`, `docs/MAPREDUCE_SETUP.md` | PARTIALLY COMPLETED (documented, not run on real cluster) |
| HDFS | `docs/HDFS_SETUP.md`, `scripts/hdfs_ingest.sh`, local path stand-in | PARTIALLY COMPLETED / REQUIRES MANUAL SETUP |
| MapReduce | `src/mapreduce/*` | COMPLETED (local mode) |
| NoSQL (MongoDB) | `src/nosql/*` | COMPLETED (local `mongomock` mode) |
| Hive | `src/hive/hql/*.hql` | PARTIALLY COMPLETED (written, unexecuted; substitute Spark SQL results only) |
| Pig | — | NOT IMPLEMENTED |
| Apache Spark | `src/spark/*` | COMPLETED (local mode) |
| Spark DataFrames & SQL | `src/spark/data_processing.py`, `hive_style_analytics.py` | COMPLETED |
| Data analytics & visualization | `src/visualization/plots.py` | COMPLETED |
| Machine learning | `src/fraud_detection/*` | COMPLETED |

---

### 15. Review-2 Readiness

**What is completed** (verified working, with real generated output):
- Real Kaggle dataset (1.85M transactions) with separate train/test files
- Data cleaning/preprocessing pipeline
- MapReduce (2 jobs, local execution, unit-tested)
- Spark DataFrame processing + Spark SQL analytics (local execution)
- Fraud detection ML pipeline (3 models, chronological validation, final metrics on the separate test file)
- NoSQL document storage/query pattern (local `mongomock` mode)
- 9 visualizations
- 49 passing tests
- Full documentation set (README + 7 docs files)

**What is partially completed**:
- Hive: real `.hql` exists but has never run under an actual Hive engine — only a Spark SQL stand-in has produced results
- HDFS: local path substitute only, real HDFS never provisioned
- EDA: informal/overlapping with preprocessing and visualization, no dedicated deliverable
- Testing: core logic covered, but Spark/NoSQL/training pipelines aren't under automated tests

**What is missing entirely**:
- Pig — not started at all
- Any execution against real Hadoop/HDFS/Hive/MongoDB infrastructure — everything currently runs in local/substitute mode

**What to demonstrate during the review**:
- The full local pipeline running live (`scripts/run_pipeline.ps1`) as the primary demo
- The MapReduce mapper→shuffle→reducer mechanics explained against real output (`results/mapreduce_output/`)
- The model comparison table and an honest explanation of why PR-AUC, not ROC-AUC or accuracy, was used to pick the best model
- Be prepared to state clearly, if asked, that Hive/HDFS/MongoDB are demonstrated via documented real-cluster procedures plus a locally-running functional substitute (Spark SQL / local paths / mongomock), not against live cluster infrastructure — and that Pig is not yet implemented.

---

### 16. Next Implementation Steps

In order of priority for closing the biggest gaps before Review 2:

1. **Decide whether Pig is required for Review 2.** If yes, implement a minimal but real Pig Latin script (e.g. category-level aggregation, mirroring the MapReduce job) plus local/cluster run instructions, following this project's established pattern.
2. **Stand up one real piece of infrastructure** to convert at least one "PARTIALLY COMPLETED / REQUIRES MANUAL SETUP" row into fully verified — most impactful candidate: install MongoDB Community Server locally (single binary, fastest path) and re-run `src/nosql/store_analytics.py` against it, capturing real (non-mongomock) evidence.
3. **If time allows, stand up WSL2 + Hadoop + Hive** to actually execute `create_tables.hql` / `analytics_queries.hql` for real, replacing the "written but unexecuted" status with verified output.
4. **Add a short, explicit EDA script or notebook** (even a simple one) so that syllabus item is a clearly separate, demonstrable deliverable rather than folded into preprocessing/visualization.
5. **Clean up `requirements.txt`** by removing `seaborn`/`plotly`/`imbalanced-learn` (unused) or actually using them, so declared dependencies match reality.
