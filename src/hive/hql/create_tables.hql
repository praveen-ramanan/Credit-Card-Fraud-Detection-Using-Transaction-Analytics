-- ============================================================================
-- Hive DDL: create an external table over the cleaned transactions data.
--
-- Run this against a real Hadoop + Hive installation (see docs/HIVE_SETUP.md)
-- after copying the cleaned CSV into HDFS, e.g.:
--   hdfs dfs -mkdir -p /user/bda/data/processed
--   hdfs dfs -put data/processed/transactions_clean.csv /user/bda/data/processed/
--
-- EXTERNAL is used (not a managed table) so dropping the Hive table never
-- deletes the underlying HDFS data - Hive is only providing a SQL layer on
-- top of files that other tools (Spark, MapReduce) also read directly.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS fraud_analytics;
USE fraud_analytics;

DROP TABLE IF EXISTS transactions;

CREATE EXTERNAL TABLE transactions (
    idx                 INT,
    trans_date_trans_time STRING,
    cc_num              BIGINT,
    merchant            STRING,
    category            STRING,
    amt                 DOUBLE,
    first_name          STRING,
    last_name           STRING,
    gender              STRING,
    city                STRING,
    state               STRING,
    zip                 STRING,
    lat                 DOUBLE,
    long                DOUBLE,
    city_pop            INT,
    job                 STRING,
    dob                 STRING,
    trans_num           STRING,
    unix_time           BIGINT,
    merch_lat           DOUBLE,
    merch_long          DOUBLE,
    is_fraud            INT,
    txn_hour            INT,
    txn_day_of_week     STRING,
    txn_month           INT,
    is_weekend          BOOLEAN,
    is_night_txn        BOOLEAN,
    customer_age        INT,
    customer_merchant_distance_km DOUBLE,
    is_high_value       BOOLEAN
)
ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '/user/bda/data/processed/transactions_clean'
TBLPROPERTIES ("skip.header.line.count"="1");
