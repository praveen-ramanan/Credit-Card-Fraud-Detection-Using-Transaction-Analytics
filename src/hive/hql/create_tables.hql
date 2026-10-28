-- ============================================================================
-- Hive DDL: create an external table over the cleaned transactions data.
--
-- Run this against a real Hadoop + Hive installation (see docs/HIVE_SETUP.md)
-- after copying the cleaned Parquet file into HDFS, e.g.:
--   hdfs dfs -mkdir -p /user/bda/data/processed/transactions_clean
--   hdfs dfs -put data/processed/transactions_clean.parquet /user/bda/data/processed/transactions_clean/
--
-- The file is produced by src/preprocessing/clean_data.py from the Kaggle
-- fraudTrain.csv + fraudTest.csv (1,852,394 rows); the `split` column says
-- which file each row came from. Parquet (columnar, compressed, typed) is
-- used instead of a ~600 MB text CSV.
--
-- EXTERNAL is used (not a managed table) so dropping the Hive table never
-- deletes the underlying HDFS data - Hive is only providing a SQL layer on
-- top of files that other tools (Spark, MapReduce) also read directly.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS fraud_analytics;
USE fraud_analytics;

DROP TABLE IF EXISTS transactions;

CREATE EXTERNAL TABLE transactions (
    trans_date_trans_time TIMESTAMP,
    cc_num              BIGINT,
    merchant            STRING,
    category            STRING,
    amt                 DOUBLE,
    `first`             STRING,
    `last`              STRING,
    gender              STRING,
    street              STRING,
    city                STRING,
    state               STRING,
    zip                 BIGINT,
    lat                 DOUBLE,
    long                DOUBLE,
    city_pop            BIGINT,
    job                 STRING,
    dob                 TIMESTAMP,
    trans_num           STRING,
    unix_time           BIGINT,
    merch_lat           DOUBLE,
    merch_long          DOUBLE,
    is_fraud            TINYINT,
    txn_hour            INT,
    txn_day_of_week     STRING,
    txn_month           INT,
    is_weekend          BOOLEAN,
    is_night_txn        BOOLEAN,
    customer_age        BIGINT,
    customer_merchant_distance_km DOUBLE,
    is_high_value       BOOLEAN,
    split               STRING
)
STORED AS PARQUET
LOCATION '/user/bda/data/processed/transactions_clean';
