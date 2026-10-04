#!/usr/bin/env python
"""
MapReduce Reducer - Category-level transaction aggregation.

Hadoop Streaming guarantees that, between the map and reduce phase, all
values for the same key are grouped together and keys arrive in sorted
order (the "shuffle & sort" step). This reducer relies on exactly that
guarantee: it keeps a running aggregate for the "current key" and flushes
it to stdout whenever the key changes, so it never needs to hold the whole
dataset in memory - it processes one category's group at a time.

Output line format:
    <category>\t<txn_count>\t<total_amount>\t<fraud_count>\t<fraud_amount>\t<fraud_pct>
"""

import sys


def emit(key, count, total_amt, fraud_count, fraud_amt):
    fraud_pct = (fraud_count / count * 100) if count else 0.0
    print(f"{key}\t{count}\t{total_amt:.2f}\t{fraud_count}\t{fraud_amt:.2f}\t{fraud_pct:.3f}")


def main():
    current_key = None
    count = 0
    total_amt = 0.0
    fraud_count = 0
    fraud_amt = 0.0

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        key, value = line.split("\t", 1)
        amt_str, fraud_str = value.split(",")
        amt = float(amt_str)
        is_fraud = int(fraud_str)

        if current_key is not None and key != current_key:
            emit(current_key, count, total_amt, fraud_count, fraud_amt)
            count, total_amt, fraud_count, fraud_amt = 0, 0.0, 0, 0.0

        current_key = key
        count += 1
        total_amt += amt
        if is_fraud:
            fraud_count += 1
            fraud_amt += amt

    if current_key is not None:
        emit(current_key, count, total_amt, fraud_count, fraud_amt)


if __name__ == "__main__":
    main()
