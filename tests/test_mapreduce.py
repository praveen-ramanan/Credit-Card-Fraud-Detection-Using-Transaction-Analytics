"""
Unit tests for the MapReduce mapper/reducer logic, run as real subprocess
pipelines (mapper -> sort -> reducer) exactly as src/mapreduce/local_driver
does, to test the actual Hadoop-Streaming-compatible scripts rather than
re-implementing their logic in the test.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAPREDUCE_DIR = PROJECT_ROOT / "src" / "mapreduce"

CSV_INPUT = (
    "index,category,amt,is_fraud,cc_num\n"
    "0,grocery,100.0,0,111\n"
    "1,grocery,50.0,1,111\n"
    "2,travel,900.0,1,222\n"
)


def _run_job(job_name: str, csv_text: str) -> dict:
    job_dir = MAPREDUCE_DIR / job_name
    map_proc = subprocess.run(
        [sys.executable, str(job_dir / "mapper.py")],
        input=csv_text.encode(), capture_output=True,
    )
    assert map_proc.returncode == 0, map_proc.stderr.decode()

    lines = sorted(map_proc.stdout.decode().splitlines(), key=lambda l: l.split("\t", 1)[0])

    reduce_proc = subprocess.run(
        [sys.executable, str(job_dir / "reducer.py")],
        input=("\n".join(lines) + "\n").encode(), capture_output=True,
    )
    assert reduce_proc.returncode == 0, reduce_proc.stderr.decode()

    result = {}
    for line in reduce_proc.stdout.decode().splitlines():
        key, count, total_amt, fraud_count, fraud_amt, fraud_pct = line.split("\t")
        result[key] = {
            "count": int(count), "total_amt": float(total_amt),
            "fraud_count": int(fraud_count), "fraud_amt": float(fraud_amt),
        }
    return result


def test_category_aggregation():
    result = _run_job("category_aggregation", CSV_INPUT)
    assert result["grocery"]["count"] == 2
    assert result["grocery"]["total_amt"] == 150.0
    assert result["grocery"]["fraud_count"] == 1
    assert result["grocery"]["fraud_amt"] == 50.0

    assert result["travel"]["count"] == 1
    assert result["travel"]["fraud_count"] == 1
    assert result["travel"]["fraud_amt"] == 900.0


def test_customer_aggregation_anonymizes_and_aggregates():
    result = _run_job("customer_aggregation", CSV_INPUT)
    # two distinct customers (cc_num 111, 222) -> two anonymized keys
    assert len(result) == 2
    total_txns = sum(v["count"] for v in result.values())
    assert total_txns == 3
    # keys must be anonymized, not the raw card numbers
    assert "111" not in result and "222" not in result
    assert all(k.startswith("cust_") for k in result)
