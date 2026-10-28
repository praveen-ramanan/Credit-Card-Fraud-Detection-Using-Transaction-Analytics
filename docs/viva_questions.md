# Viva Questions & Answers

## Big Data Fundamentals

**Q: Why is Big Data relevant to banking transaction processing?**
Banks process millions of card transactions daily, each with dozens of
attributes (amount, merchant, location, time, customer). This satisfies
the classic "3 Vs": **Volume** (millions of records), **Velocity**
(continuous transaction streams), and **Variety** (structured transaction
fields alongside semi-structured fraud-alert/investigation data). A
single relational database on one machine struggles to store, query, and
analyze this at scale reliably - which is why a distributed storage and
processing stack (HDFS, MapReduce, Hive, Spark) is the standard
architecture, and why this project builds one instead of just using a
single pandas DataFrame throughout.

**Q: Why not just use a single powerful server with a normal SQL database?**
Vertical scaling (a bigger single machine) hits a cost and physical
ceiling and offers no fault tolerance - one disk failure loses everything.
Horizontal scaling (more machines, each ordinary/commodity hardware) is
cheaper per unit of capacity and, in HDFS's case, keeps working through
individual node failures because data is replicated across nodes.

## Hadoop / HDFS

**Q: Why Hadoop/HDFS specifically?**
HDFS splits large files into blocks and replicates them (default 3x)
across a cluster, giving fault tolerance and high aggregate read
throughput for batch analytics - exactly the access pattern MapReduce,
Hive, and Spark all use in this project (read large files sequentially,
in parallel, across many nodes).

**Q: What is a NameNode vs a DataNode?**
The **NameNode** holds the filesystem metadata (which blocks make up
which file, and which DataNodes hold which blocks) - it is the single
"directory" for the cluster. **DataNodes** actually store the data
blocks and serve read/write requests. Losing the NameNode without a
backup loses the ability to locate data even though the DataNodes' blocks
are intact, which is why production HDFS setups run NameNode HA.

**Q: Does this project use a real HDFS cluster?**
By default, no - it runs in local demo mode, with `data/raw/` and
`data/processed/` standing in for HDFS paths (see `docs/HDFS_SETUP.md`
for why, and the real setup/ingestion commands via WSL2 for the cluster
demonstration).

## MapReduce

**Q: What is MapReduce?**
A programming model for processing large datasets in parallel across a
cluster in three conceptual phases: **Map** (transform each input record
independently into key-value pairs), **Shuffle & Sort** (the framework
groups all values sharing the same key together), and **Reduce**
(aggregate each key's group of values into a result). It scales
horizontally because the map phase is embarrassingly parallel and the
reduce phase only needs one key's group in memory at a time.

**Q: Walk through this project's MapReduce job.**
`src/mapreduce/category_aggregation/mapper.py` reads transaction CSV
rows and emits `<category>\t<amount>,<is_fraud>` per row. The framework
(or, locally, `local_driver.py`'s Python `sort()`) groups and sorts these
by category. `reducer.py` then streams through each category's group,
keeping a running `count`, `total_amount`, `fraud_count`, `fraud_amount`,
and emits one summary line per category when the key changes - it never
holds more than one category's running total in memory at once.

**Q: Why mask the card number in the customer_aggregation job?**
Real Primary Account Numbers (PANs) should never appear in plaintext in
analytics output - this mirrors PCI-DSS handling practice. The mapper
hashes `cc_num` (SHA-256, truncated) into an anonymized `cust_...` key
instead of emitting the raw number.

**Q: Why does Hadoop guarantee all values for a key arrive together at the reducer?**
That grouping - the shuffle/sort step - is the core service the
MapReduce framework provides between the map and reduce phases; it's
what lets the reducer be written as a simple single-pass streaming
aggregator instead of needing to search for matching keys itself.

## Hive

**Q: Why Hive if Spark can also run SQL?**
Hive was built first and established SQL-on-Hadoop as a pattern:
translating HiveQL into MapReduce (or Tez/Spark) jobs so analysts don't
have to hand-write MapReduce for routine aggregation. It's still commonly
used for its Hive Metastore (a shared catalog of table schemas other
tools, including Spark, can read) and for organizations already
standardized on it. This project implements Hive queries as the
authoritative version (`src/hive/hql/`) and runs the same logic through
Spark SQL locally as the demo substitute.

**Q: What does `EXTERNAL TABLE` mean in `create_tables.hql`, and why use it here?**
An external table tells Hive the data lives at a specified HDFS location
and is managed outside Hive - `DROP TABLE` removes only Hive's metadata,
never the underlying files. That matters here because the same HDFS
files are also read directly by the MapReduce and Spark stages; those
files should not disappear just because someone drops the Hive table.

## Spark

**Q: Why Spark instead of just MapReduce?**
MapReduce writes intermediate results to disk between every map and
reduce stage and forces every computation into that rigid two-phase
shape. Spark keeps data in memory across a chain of transformations and
optimizes the whole query plan (via Catalyst/Tungsten), so a chained
`filter → groupBy → agg` runs as one optimized job instead of several
disk-synchronized MapReduce passes. Spark's DataFrame/SQL API is also
much higher-level than hand-writing mappers and reducers.

**Q: What is the difference between a Spark transformation and an action?**
Transformations (`filter`, `groupBy`, `select`) are lazy - they build up
a query plan without executing. Actions (`count`, `show`, `collect`,
writing output) trigger actual execution of that plan. This project's
`data_processing.py` builds several transformations before calling
`.count()`/`.show()`, which is where Spark actually runs the job.

**Q: What's `spark.sql()` doing differently from the DataFrame API calls in this project?**
Nothing fundamentally - `createOrReplaceTempView()` registers a
DataFrame as a named view, and `spark.sql("SELECT ...")` compiles to the
same underlying query plan as the equivalent `.groupBy().agg()` calls.
Both paths are shown in `src/spark/data_processing.py` specifically to
make that equivalence visible.

## NoSQL

**Q: What is NoSQL, and why not just use another Hive/SQL table for the fraud alerts too?**
NoSQL databases relax the fixed-schema, tabular model relational
databases use, trading strict schema/joins for flexibility and horizontal
write scalability. This project's fraud-alert records are naturally
semi-structured and can vary in shape (a rule-based alert vs. a
model-score alert may carry different fields, and alerts nest a customer
sub-document) - forcing that into a fixed relational schema would mean
either a very wide, sparse table or frequent `ALTER TABLE` migrations as
fraud rules evolve. A document store (MongoDB here) stores each alert
document as-is.

**Q: MongoDB vs Cassandra - why MongoDB?**
MongoDB uses a document data model (nested JSON-like records), suited to
the flexible, nested fraud-alert documents this project stores. Cassandra
uses a wide-column model tuned for extremely high write throughput spread
across many nodes/datacenters via a partition key - a different strength
this project's demo scale doesn't specifically need. MongoDB is also far
simpler to install and demo on a single laptop. (Full comparison table in
`docs/NOSQL_SETUP.md`.)

**Q: Did this project use a real MongoDB server?**
It supports one (`pymongo` against a real server or MongoDB Atlas), but
falls back automatically to `mongomock` - an in-memory library
implementing the same API - if no server is reachable, so the exact same
code path runs either way. Development/testing here used the
`mongomock` fallback.

## Machine Learning / Fraud Detection

**Q: Why is fraud detection a classification problem, and specifically an imbalanced one?**
Each transaction has a binary outcome to predict - fraud or legitimate -
which is classification by definition. It's an *imbalanced* classification
problem because the positive (fraud) class is rare: 0.58% of
`fraudTrain.csv` and 0.39% of `fraudTest.csv` transactions in this project's data, and well under 1% in real-world card data too.
That imbalance is the central challenge - most of the standard ML
machinery (accuracy, default 0.5 thresholds, unweighted loss functions)
implicitly assumes roughly balanced classes and misleads badly if used
unchanged here.

**Q: Why not just report accuracy?**
Because a model that predicts "legitimate" for every single transaction
would score >99% accuracy on this data while catching zero fraud - a
useless model with a misleadingly great-looking number. This project
reports precision, recall, F1, ROC-AUC, and PR-AUC instead, and never
quotes accuracy as a headline metric.

**Q: Precision vs recall - what do they mean here, and why does the tradeoff matter?**
**Precision** = of the transactions the model flagged as fraud, what
fraction actually were fraud. Low precision means many legitimate
customers get incorrectly blocked/flagged (false positives) - a customer
experience and manual-review cost problem. **Recall** = of the actual
fraud, what fraction did the model catch. Low recall means real fraud
slips through (false negatives) - a direct financial loss. The two
trade off against each other as the decision threshold moves, which is
why this project reports both, plus F1 (their harmonic mean) and the
metrics at a tuned threshold, not just accuracy or one metric alone.

**Q: False positives vs false negatives in this domain, concretely?**
A **false positive** is a legitimate transaction the model flags as
fraud - the customer's card may get declined or held for review even
though nothing was wrong. A **false negative** is actual fraud the model
misses - the bank/customer absorbs that financial loss directly. Banks
generally tolerate some false positives to catch more fraud, but too many
erodes customer trust and generates review workload - this is a genuine
business tradeoff, not a purely technical one, which is exactly why
threshold tuning (Section 15 of the README) is a deliberate step here
rather than always using 0.5.

**Q: Why is ROC-AUC alone not enough for this dataset?**
ROC-AUC evaluates the true-positive rate against the false-positive rate,
and the false-positive rate is computed against the (very large) true
negative population - so ROC-AUC can look good even when precision on
the rare positive class is poor. This project observed this directly:
on the validation set Logistic Regression has a respectable-looking
ROC-AUC of 0.945 but a PR-AUC of only 0.198 (at its best threshold only
30% of its flags are fraud), while XGBoost reaches ROC-AUC 0.998 and
PR-AUC 0.899. PR-AUC (precision-recall AUC) is
reported alongside it because it's computed only from precision and
recall on the positive class, making it far more informative under this
level of imbalance.

**Q: How was class imbalance handled?**
Via `class_weight="balanced"` (Logistic Regression, Random Forest) and
`scale_pos_weight` (XGBoost, set from the train-set class ratio) - these
tell the model to penalize mistakes on the rare fraud class more heavily
during training, rather than duplicating minority-class rows
(oversampling) or discarding majority-class rows (undersampling).

**Q: Which features were used, and why were card number/name excluded?**
Amount, transaction hour, customer age, customer-merchant distance,
city population, weekend/night/high-value flags, and one-hot encoded
category/state/gender - 75 columns (`src/fraud_detection/features.py`).
Card number, transaction ID, names, street and zip were excluded because
they identify a specific individual rather than generalizing: the dataset
has only ~1,000 cards, so the model would memorise cards instead of
learning fraud patterns. Merchant, city and job were excluded for the
same memorisation reason, and month because the training and test files
cover different months/years. The label itself is guarded by an
assertion in `build_features()`. Raw timestamps and DOB are used only
through derived features (hour, weekend, age).

**Q: Why train three different models instead of just one?**
To compare how different model families handle the same imbalanced
problem - Logistic Regression (a simple linear baseline), Random Forest
(a tree ensemble that can capture non-linear interactions), and XGBoost
(gradient boosting, generally strong on tabular data). Comparing them,
rather than assuming one is best, is itself part of demonstrating the
evaluation methodology, and the results (Section 15 of the README) show
that the linear baseline is far behind the tree ensembles on PR-AUC
(0.198 vs 0.835 / 0.899), even though its ROC-AUC (0.945) looks fine.

**Q: How did you avoid data leakage between training and testing?**
The two Kaggle files are kept separate. `fraudTrain.csv` is split
chronologically: models are fit on the earliest 80% and validated on the
latest 20%. Model choice (PR-AUC) and the decision threshold (F1-optimal)
are made on that validation set. `fraudTest.csv` is never merged,
re-split, fitted on or tuned on - it is scored once at the end, giving
ROC-AUC 0.997, PR-AUC 0.855, precision 0.813, recall 0.777. The one-hot
vocabulary and scaler are learned from training rows only. Cleaning is
row-by-row, so it learns nothing.

## Architecture / End-to-End

**Q: Describe how the complete pipeline works, end to end.**
Transaction data (the Kaggle `fraudTrain.csv` / `fraudTest.csv` files,
1.85M transactions) is ingested and placed under storage paths that
stand in for HDFS locally (with real HDFS commands documented separately).
`clean_data.py` deduplicates, handles missing values, and derives
analytics-friendly features using pandas. From there, three parallel
analytical views are produced: (1) real Hadoop-Streaming-compatible
MapReduce jobs aggregate by category and customer; (2) Hive `.hql`
queries (mirrored locally via Spark SQL) answer eleven transaction-analytics
questions; (3) Spark DataFrames demonstrate distributed-style
filter/groupBy/SQL processing directly. The cleaned data also feeds a
supervised fraud classifier (Logistic Regression / Random Forest /
XGBoost, imbalance-corrected, evaluated on precision/recall/F1/ROC-AUC/
PR-AUC), whose highest-risk predictions are stored as semi-structured
alert documents in MongoDB (or its local `mongomock` fallback) alongside
the aggregated analytics. Finally, matplotlib visualizations turn the
aggregated and model outputs into the fraud/transaction analytics charts
a bank fraud-review team would actually look at.

**Q: What would need to change to run this on a real cluster instead of locally?**
Almost nothing in the Python code - `config/config.py` centralizes all
paths and settings, and `BDA_SPARK_MASTER`/`BDA_MONGO_URI` env vars
already parameterize the Spark master and MongoDB connection. The changes
are: point those env vars at the real cluster/server, upload the data to
HDFS (`scripts/hdfs_ingest.sh`), and run the real `.hql` scripts via
`hive -f` and the MapReduce jobs via `hadoop jar hadoop-streaming-*.jar`
instead of the local driver - all documented per-component in `docs/`.
