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
# Usage (from WSL2, with the project mounted under /mnt/d/PROJECTS/BDA and
# the Kaggle files in the Windows Downloads folder):
#   bash scripts/hdfs_ingest.sh /mnt/d/PROJECTS/BDA "/mnt/c/Users/<you>/Downloads/archive (5)"

set -euo pipefail

PROJECT_DIR="${1:-/mnt/d/PROJECTS/BDA}"
KAGGLE_DIR="${2:-/mnt/c/Users/${USER}/Downloads/archive (5)}"

echo "Creating HDFS directories..."
hdfs dfs -mkdir -p /user/bda/data/raw
hdfs dfs -mkdir -p /user/bda/data/processed/transactions_clean

echo "Uploading raw Kaggle transaction files..."
hdfs dfs -put -f "${KAGGLE_DIR}/fraudTrain.csv" /user/bda/data/raw/
hdfs dfs -put -f "${KAGGLE_DIR}/fraudTest.csv" /user/bda/data/raw/

echo "Uploading cleaned transaction data (Parquet)..."
hdfs dfs -put -f "${PROJECT_DIR}/data/processed/transactions_clean.parquet" /user/bda/data/processed/transactions_clean/

echo "Verifying upload..."
hdfs dfs -ls -h /user/bda/data/raw
hdfs dfs -ls -h /user/bda/data/processed/transactions_clean
hdfs dfs -du -h /user/bda/data

echo "Done. Sample of ingested raw data:"
hdfs dfs -cat /user/bda/data/raw/fraudTrain.csv | head -5
