# MapReduce: Local Demo & Real Hadoop Cluster

## What's implemented

Two classic Hadoop Streaming-style MapReduce jobs, under `src/mapreduce/`:

- `category_aggregation/` - transaction count, total amount, fraud count,
  fraud amount, and fraud rate **per merchant category**.
- `customer_aggregation/` - the same aggregates **per (anonymized)
  customer**, used to surface top-spending and most-flagged customers.

Each job is a standalone `mapper.py` / `reducer.py` pair that reads from
`stdin` and writes to `stdout` using tab-separated key/value pairs - the
exact interface Hadoop Streaming uses. This means the identical scripts
run either locally (no Hadoop needed) or submitted to a real cluster.

## How the algorithm works (for the viva)

1. **Map phase**: each mapper reads transaction rows and emits
   `<key>\t<amount>,<is_fraud>` - one line per input transaction. The key
   is the category (or anonymized customer ID) being aggregated by.
2. **Shuffle & sort**: the MapReduce framework groups all values sharing
   the same key together, and sorts keys - this is provided by Hadoop's
   own framework on a cluster, and simulated with a plain Python
   `list.sort()` in `src/mapreduce/local_driver.py` for the local demo.
3. **Reduce phase**: for each key, the reducer streams through its grouped
   values exactly once, maintaining a running total (count, sum, fraud
   count, fraud amount) and emits one summary line when the key changes.
   This is the classic "combine while streaming" reducer pattern - it
   never needs to hold the whole dataset in memory.

## Running locally (no Hadoop needed)

```powershell
.\venv\Scripts\python.exe src\mapreduce\local_driver.py category_aggregation
.\venv\Scripts\python.exe src\mapreduce\local_driver.py customer_aggregation
```

Output: `results/mapreduce_output/<job_name>.tsv`

You can also run the classic Unix pipeline manually to see the same
mapper/reducer contract Hadoop Streaming uses:

```bash
# from a bash shell (e.g. Git Bash); the mapper reads the raw Kaggle CSV
cat "$HOME/Downloads/archive (5)/fraudTrain.csv" \
  | python src/mapreduce/category_aggregation/mapper.py \
  | sort \
  | python src/mapreduce/category_aggregation/reducer.py
```

(the mapper uses `csv.DictReader` on stdin, which consumes the first line
as the header itself - pass the full CSV including its header, do not
strip it. Quoted merchant names containing commas are parsed correctly.
`local_driver.py` runs one map task per input file - `fraudTrain.csv` and
`fraudTest.csv` - and reduces their combined output, 1,852,394 records.)

## Running on a real Hadoop cluster (Hadoop Streaming)

Requires HDFS to be running with the input file uploaded (see
`docs/HDFS_SETUP.md` / `scripts/hdfs_ingest.sh`).

```bash
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
    -input  /user/bda/data/raw/fraudTrain.csv \
    -input  /user/bda/data/raw/fraudTest.csv \
    -output /user/bda/output/category_aggregation \
    -mapper "python3 mapper.py" \
    -reducer "python3 reducer.py" \
    -file src/mapreduce/category_aggregation/mapper.py \
    -file src/mapreduce/category_aggregation/reducer.py

hdfs dfs -cat /user/bda/output/category_aggregation/part-00000
```

Repeat with the `customer_aggregation/` mapper and reducer for the
customer-level job. `-output` must not already exist on HDFS (Hadoop
Streaming refuses to overwrite it) - delete it first with
`hdfs dfs -rm -r /user/bda/output/category_aggregation` if re-running.
