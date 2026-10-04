#!/usr/bin/env python
"""
MapReduce Mapper - Customer-level transaction aggregation.

Same Hadoop Streaming pattern as category_aggregation/mapper.py, keyed by
customer instead of category. The card number is hashed to a short
anonymized customer ID rather than emitted in full - real PAN (Primary
Account Number) data should never appear in plaintext analytics output,
consistent with PCI-DSS handling practice for card data. This is a
worthwhile point to raise in the viva.

Emits: <customer_id>\t<amount>,<is_fraud>
"""

import csv
import hashlib
import sys


def anonymize(cc_num: str) -> str:
    return "cust_" + hashlib.sha256(cc_num.encode()).hexdigest()[:10]


def main():
    reader = csv.DictReader(sys.stdin)
    for row in reader:
        try:
            customer_id = anonymize(row["cc_num"])
            amount = float(row["amt"])
            is_fraud = int(row["is_fraud"])
        except (KeyError, ValueError):
            continue
        print(f"{customer_id}\t{amount},{is_fraud}")


if __name__ == "__main__":
    main()
