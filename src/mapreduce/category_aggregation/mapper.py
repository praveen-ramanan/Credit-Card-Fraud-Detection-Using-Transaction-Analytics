#!/usr/bin/env python
"""
MapReduce Mapper - Category-level transaction aggregation.

Reads transaction CSV lines from stdin (Hadoop Streaming passes each HDFS
input split's lines to stdin, one line per call in local testing) and emits
one key/value pair per transaction:

    <category>\t<amount>,<is_fraud>

This is a classic Hadoop Streaming mapper: it can be run standalone via
    hadoop jar hadoop-streaming-*.jar \
        -input /user/bda/data/raw/fraudTrain.csv -output /data/out \
        -mapper mapper.py -reducer reducer.py \
        -file mapper.py -file reducer.py

or locally (no Hadoop needed) via the pipeline in
src/mapreduce/local_driver.py, which pipes:
    cat fraudTrain.csv | mapper.py | sort | reducer.py
"""

import csv
import sys


def main():
    reader = csv.DictReader(sys.stdin)
    for row in reader:
        try:
            category = row["category"] or "unknown"
            amount = float(row["amt"])
            is_fraud = int(row["is_fraud"])
        except (KeyError, ValueError):
            # Skip malformed rows rather than crashing the whole job -
            # a single bad line should not fail a distributed job.
            continue
        print(f"{category}\t{amount},{is_fraud}")


if __name__ == "__main__":
    main()
