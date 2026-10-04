#!/usr/bin/env bash
# Real HDFS ingestion commands for this project.
#
# Run this from a shell with a working `hdfs` client (e.g. inside a WSL2
# Ubuntu session with Hadoop installed and HDFS started - see
# docs/HDFS_SETUP.md). It copies this project's local data/ files into
# HDFS under /user/bda/, matching the paths expected by
# src/hive/hql/create_tables.hql and the MapReduce Hadoop Streaming
# commands in docs/MAPREDUCE_SETUP.md.
#
# Usage (from WSL2, with the project mounted under /mnt/d/PROJECTS/BDA):
#   bash scripts/hdfs_ingest.sh /mnt/d/PROJECTS/BDA

set -euo pipefail

PROJECT_DIR="${1:-/mnt/d/PROJECTS/BDA}"

echo "Creating HDFS directories..."
hdfs dfs -mkdir -p /user/bda/data/raw
hdfs dfs -mkdir -p /user/bda/data/processed

echo "Uploading raw transaction data..."
hdfs dfs -put -f "${PROJECT_DIR}/data/raw/transactions.csv" /user/bda/data/raw/

echo "Uploading cleaned transaction data..."
hdfs dfs -put -f "${PROJECT_DIR}/data/processed/transactions_clean.csv" /user/bda/data/processed/

echo "Verifying upload..."
hdfs dfs -ls -h /user/bda/data/raw
hdfs dfs -ls -h /user/bda/data/processed
hdfs dfs -du -h /user/bda/data

echo "Done. Sample of ingested raw data:"
hdfs dfs -cat /user/bda/data/raw/transactions.csv | head -5
