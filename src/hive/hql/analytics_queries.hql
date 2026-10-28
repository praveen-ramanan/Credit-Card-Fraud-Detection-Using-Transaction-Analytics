-- ============================================================================
-- Hive analytics queries for the credit-card transaction dataset.
-- Run after create_tables.hql on a real Hive installation (see
-- docs/HIVE_SETUP.md). The identical analytics are also executed locally via
-- Spark SQL in src/spark/hive_style_analytics.py, since Spark SQL is
-- HiveQL-compatible for these queries - that is the "local demo mode" for
-- this stage when a Hive cluster isn't available.
-- ============================================================================

USE fraud_analytics;

-- 1. Total number of transactions and total transaction value
SELECT
    COUNT(*)          AS total_transactions,
    ROUND(SUM(amt),2) AS total_transaction_value,
    ROUND(AVG(amt),2) AS avg_transaction_value
FROM transactions;

-- 2. Fraud vs legitimate transaction count and fraud percentage
SELECT
    is_fraud,
    COUNT(*)                                              AS txn_count,
    ROUND(SUM(amt), 2)                                    AS total_amount,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 3)    AS pct_of_transactions
FROM transactions
GROUP BY is_fraud;

-- 3. Fraud amount vs legitimate amount
SELECT
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2) AS total_fraud_amount,
    ROUND(SUM(CASE WHEN is_fraud = 0 THEN amt ELSE 0 END), 2) AS total_legit_amount,
    ROUND(100.0 * SUM(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END) / SUM(amt), 3) AS fraud_amount_pct
FROM transactions;

-- 4. Top transaction categories by volume and by fraud rate
SELECT
    category,
    COUNT(*)                                         AS txn_count,
    ROUND(SUM(amt), 2)                                AS total_amount,
    SUM(is_fraud)                                     AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)        AS fraud_rate_pct
FROM transactions
GROUP BY category
ORDER BY fraud_rate_pct DESC;

-- 5. Highest-value transactions (top 20)
SELECT trans_num, cc_num, merchant, category, amt, is_fraud, trans_date_trans_time
FROM transactions
ORDER BY amt DESC
LIMIT 20;

-- 6. High-value transaction analysis: fraud rate among high-value transactions
--    vs overall fraud rate (are big transactions riskier?)
SELECT
    is_high_value,
    COUNT(*)                                    AS txn_count,
    SUM(is_fraud)                               AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)  AS fraud_rate_pct
FROM transactions
GROUP BY is_high_value;

-- 7. Transactions by hour of day (time-based pattern analysis)
SELECT
    txn_hour,
    COUNT(*)                                    AS txn_count,
    SUM(is_fraud)                               AS fraud_count,
    ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3)  AS fraud_rate_pct
FROM transactions
GROUP BY txn_hour
ORDER BY txn_hour;

-- 8. Transactions by state/location
SELECT
    state,
    COUNT(*)                                    AS txn_count,
    ROUND(SUM(amt), 2)                          AS total_amount,
    SUM(is_fraud)                               AS fraud_count
FROM transactions
GROUP BY state
ORDER BY txn_count DESC;

-- 9. Customer-level aggregation: top 20 customers by total spend
SELECT
    cc_num,
    COUNT(*)             AS txn_count,
    ROUND(SUM(amt), 2)   AS total_spend,
    SUM(is_fraud)         AS fraud_count
FROM transactions
GROUP BY cc_num
ORDER BY total_spend DESC
LIMIT 20;

-- 10. Merchant-level aggregation: top 20 merchants by fraud amount
SELECT
    merchant,
    COUNT(*)                                       AS txn_count,
    SUM(is_fraud)                                  AS fraud_count,
    ROUND(SUM(CASE WHEN is_fraud=1 THEN amt ELSE 0 END), 2) AS fraud_amount
FROM transactions
GROUP BY merchant
ORDER BY fraud_amount DESC
LIMIT 20;

-- 11. Train vs test file summary (fraudTrain.csv vs fraudTest.csv rows)
SELECT split,
       COUNT(*) AS txn_count,
       SUM(is_fraud) AS fraud_count,
       ROUND(100.0 * SUM(is_fraud) / COUNT(*), 3) AS fraud_rate_pct,
       MIN(trans_date_trans_time) AS first_txn,
       MAX(trans_date_trans_time) AS last_txn
FROM transactions
GROUP BY split
ORDER BY split DESC;
