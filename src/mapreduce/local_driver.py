"""
Local MapReduce driver - simulates the Hadoop Streaming shuffle/sort phase
in pure Python so the mapper.py / reducer.py scripts can be demonstrated
without a Hadoop installation.

This runs the EXACT SAME mapper.py and reducer.py scripts that would be
submitted to a real cluster via:

    hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
        -input /user/bda/transactions.csv \
        -output /user/bda/output/category_aggregation \
        -mapper "python mapper.py" -reducer "python reducer.py" \
        -file src/mapreduce/category_aggregation/mapper.py \
        -file src/mapreduce/category_aggregation/reducer.py

Locally, it just replaces HDFS + YARN's job scheduling and the shuffle/sort
step with: subprocess pipe -> Python sort() -> subprocess pipe. The
map/reduce logic itself is untouched, which is the point being
demonstrated - the same code runs standalone or on a cluster.

Usage:
    python src/mapreduce/local_driver.py category_aggregation
    python src/mapreduce/local_driver.py customer_aggregation
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import config as cfg

MAPREDUCE_DIR = Path(__file__).resolve().parent


def run_job(job_name: str, input_csv: Path = cfg.CLEANED_TRANSACTIONS_CSV) -> Path:
    job_dir = MAPREDUCE_DIR / job_name
    mapper_path = job_dir / "mapper.py"
    reducer_path = job_dir / "reducer.py"
    if not mapper_path.exists() or not reducer_path.exists():
        raise FileNotFoundError(f"MapReduce job '{job_name}' not found under {job_dir}")
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input file {input_csv} not found. Run "
            f"'python src/preprocessing/clean_data.py' first."
        )

    python_exe = sys.executable

    # --- Map phase ----------------------------------------------------
    with open(input_csv, "rb") as f_in:
        map_proc = subprocess.run(
            [python_exe, str(mapper_path)],
            stdin=f_in, capture_output=True, text=False,
        )
    if map_proc.returncode != 0:
        raise RuntimeError(f"Mapper failed:\n{map_proc.stderr.decode(errors='replace')}")

    mapped_lines = map_proc.stdout.decode("utf-8").splitlines()
    print(f"[{job_name}] Map phase emitted {len(mapped_lines):,} key-value pairs")

    # --- Shuffle & sort (this is what Hadoop's framework does natively) ---
    mapped_lines.sort(key=lambda line: line.split("\t", 1)[0])

    # --- Reduce phase ---------------------------------------------------
    reduce_proc = subprocess.run(
        [python_exe, str(reducer_path)],
        input=("\n".join(mapped_lines) + "\n").encode("utf-8"),
        capture_output=True,
    )
    if reduce_proc.returncode != 0:
        raise RuntimeError(f"Reducer failed:\n{reduce_proc.stderr.decode(errors='replace')}")

    output_lines = reduce_proc.stdout.decode("utf-8").splitlines()
    print(f"[{job_name}] Reduce phase produced {len(output_lines):,} aggregated groups")

    out_dir = cfg.RESULTS_DIR / "mapreduce_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{job_name}.tsv"
    header = "key\ttxn_count\ttotal_amount\tfraud_count\tfraud_amount\tfraud_pct\n"
    out_path.write_text(header + "\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"[{job_name}] Output written -> {out_path}")
    return out_path


if __name__ == "__main__":
    job = sys.argv[1] if len(sys.argv) > 1 else "category_aggregation"
    run_job(job)
