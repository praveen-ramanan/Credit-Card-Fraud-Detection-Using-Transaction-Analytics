# Hive Setup & Usage

## What's implemented

Real HiveQL scripts under `src/hive/hql/`:

- `create_tables.hql` - creates an external Hive table over the cleaned
  transaction data on HDFS.
- `analytics_queries.hql` - ten analytics queries: totals, fraud vs
  legitimate breakdown, fraud amount share, category-level fraud rates,
  highest-value transactions, high-value fraud-rate comparison, hourly
  pattern, state distribution, top customers, top merchants by fraud
  amount.

## Local demo mode

Since a Hive metastore/cluster is not assumed to be available on a
student's laptop, the **identical analytical questions** are answered
locally through Spark SQL in `src/spark/hive_style_analytics.py` - Spark
SQL is HiveQL-compatible for the SELECT/GROUP BY/window-function syntax
used here, so the queries and their logic are the same; only the engine
executing them differs. Each query in that file is commented with which
`.hql` query it mirrors. Run it with:

```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21.0.10"   # see docs/SPARK_SETUP.md for why
.\venv\Scripts\python.exe src\spark\hive_style_analytics.py
```

Results are saved as CSVs under `results/hive_analytics/`.

## Running the real `.hql` scripts on an actual Hive installation

Requires HDFS + Hive installed (Hive typically runs on top of an existing
Hadoop/HDFS setup - see `docs/HDFS_SETUP.md` first). The simplest path on
a laptop is installing Hive inside the same WSL2 Ubuntu environment used
for Hadoop:

```bash
wget https://downloads.apache.org/hive/hive-3.1.3/apache-hive-3.1.3-bin.tar.gz
tar -xzf apache-hive-3.1.3-bin.tar.gz
mv apache-hive-3.1.3-bin ~/hive
export HIVE_HOME=~/hive
export PATH=$PATH:$HIVE_HOME/bin

# initialize the (default, embedded Derby) metastore
schematool -dbType derby -initSchema
```

Then, with HDFS running and the cleaned CSV already uploaded (see
`scripts/hdfs_ingest.sh`):

```bash
hive -f src/hive/hql/create_tables.hql
hive -f src/hive/hql/analytics_queries.hql
```

Or interactively:

```bash
hive
hive> USE fraud_analytics;
hive> SELECT category, COUNT(*) FROM transactions GROUP BY category;
```

## Why Hive here

Hive gives a familiar SQL interface (HiveQL) over data stored as plain
files in HDFS, translating each query into one or more MapReduce (or, in
modern Hive, Tez/Spark) jobs. For this project it demonstrates the
"SQL-on-Hadoop" layer of the Big Data stack: analysts can query
transaction data declaratively without writing MapReduce jobs by hand,
while the underlying execution still runs distributed batch jobs over the
same HDFS files the MapReduce and Spark stages also read.
