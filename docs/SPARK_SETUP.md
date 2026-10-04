# Spark Setup & Usage

## Java version requirement (important on this machine)

PySpark 4.x is built on Spark's JVM engine and only supports specific Java
versions (17 or 21 LTS at the time this project was built). This
machine's default `java` is JDK 26, which is too new for Spark 4.2 and
fails at SparkContext startup with `NoClassDefFoundError:
org.apache.spark.unsafe.array.ByteArrayMethods`. **This was verified while
building this project** - it is not a hypothetical caveat.

A compatible JDK 21 installation was found on this machine at
`C:\Program Files\Java\jdk-21.0.10`. Every Spark script in this project
must be run with `JAVA_HOME` pointed at that JDK instead of the system
default:

```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21.0.10"
$env:PATH = "$env:JAVA_HOME\bin;" + $env:PATH
.\venv\Scripts\python.exe src\spark\data_processing.py
```

`scripts\run_pipeline.ps1` already does this automatically for the Spark
steps. If you don't have a JDK 17/21 installed, download Eclipse Temurin
21 (LTS): `https://adoptium.net/temurin/releases/?version=21` - update
`$SparkJavaHome` in `scripts\run_pipeline.ps1` and the path above to match
where you install it.

## What's implemented

- `src/spark/spark_session.py` - shared SparkSession factory
  (`local[*]` master by default, configurable via `BDA_SPARK_MASTER`).
- `src/spark/data_processing.py` - reads the cleaned Parquet dataset into
  a Spark DataFrame, prints its schema, filters high-value transactions,
  runs a `groupBy().agg()` aggregation via the DataFrame API, and a second
  aggregation via Spark SQL against a registered temp view.
- `src/spark/hive_style_analytics.py` - runs the ten Hive-equivalent
  analytics queries (see `docs/HIVE_SETUP.md`) via `spark.sql()`.

## Why Spark (vs. plain MapReduce) here

MapReduce forces every computation into an explicit map/shuffle/reduce
pass and writes intermediate results to disk between stages, which is
correct but slow for anything beyond a single aggregation. Spark keeps
data in memory across transformations (resilient distributed datasets
under the hood) and offers a much higher-level DataFrame/SQL API, so the
same category/state aggregations that took a hand-written mapper and
reducer in `src/mapreduce/` are a single `.groupBy().agg()` call here -
and Spark can optimize/parallelize the whole query plan (Catalyst
optimizer) rather than executing each stage exactly as literally written.
This project keeps the MapReduce implementation specifically to make that
contrast visible for the viva.

## Running on a real cluster

Submit the same scripts with `spark-submit` instead of running through
the Python interpreter directly, and point `BDA_SPARK_MASTER` at the
cluster's master URL (e.g. `yarn`, or `spark://host:7077`):

```bash
export BDA_SPARK_MASTER=yarn
spark-submit --deploy-mode client src/spark/data_processing.py
```

No code changes are required - `get_spark()` in
`src/spark/spark_session.py` already reads the master from config.
