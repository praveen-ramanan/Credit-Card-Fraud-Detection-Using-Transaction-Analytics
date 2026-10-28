# ============================================================================
# End-to-end pipeline runner (Windows PowerShell, local/demo mode).
#
# Runs every stage of the project in the correct order:
#   1. Check the Kaggle fraudTrain.csv / fraudTest.csv files are readable
#      (BDA_TRAIN_CSV / BDA_TEST_CSV, default ~/Downloads/archive (5)/)
#   2. Data cleaning & preprocessing (pandas -> one Parquet file)
#   3. MapReduce aggregations (local, no Hadoop needed)
#   4. Spark DataFrame processing + Spark SQL (Hive-style) analytics
#   5. Fraud detection: train + validate on fraudTrain, final test on fraudTest
#   6. NoSQL (MongoDB / mongomock) load & example queries
#   7. Visualization (chart generation)
#
# Usage (from the project root):
#   .\scripts\run_pipeline.ps1
#
# Spark requires a JDK Spark actually supports (17 or 21 LTS); this
# machine's default JDK may be newer than Spark supports, so this script
# temporarily points JAVA_HOME at a compatible JDK for the Spark steps only.
# Adjust $SparkJavaHome below if your compatible JDK lives elsewhere.
#
# Spark's Windows launch scripts (*.cmd) break on paths containing
# parentheses, such as "Credit Card(BDA)". When the project path has one, the
# Spark steps run through a directory junction to the venv at
# %TEMP%da_venv (created automatically; it is only a link, no copy).
# ============================================================================

# NOTE: deliberately NOT setting $ErrorActionPreference = "Stop" here.
# Spark/Hadoop write normal INFO/WARN logging to stderr; under PowerShell
# 5.1, redirecting a native command's stderr wraps each line as a
# NativeCommandError and would abort the whole script on "Stop" even
# though the command actually succeeds. Real failures are instead caught
# by checking $LASTEXITCODE after each step (see Run-Step below).
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$Python = ".\venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Error "venv not found. Run: py -3.11 -m venv venv ; .\venv\Scripts\python.exe -m pip install -r requirements.txt"
}

# Known-good JDK for Spark on this machine (LTS release). Change this path
# if your system's compatible JDK is installed elsewhere.
$SparkJavaHome = "C:\Program Files\Java\jdk-21.0.10"

function Run-Step($Description, $Command) {
    Write-Host "`n=== $Description ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$Description FAILED (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# 1. Dataset check (no fallback dataset - the real Kaggle files are required)
Run-Step "1/7 Checking Kaggle dataset files" { & $Python -c "from config import config as c; from src.preprocessing.clean_data import check_raw_file; [check_raw_file(p) or print('found', p) for p in c.RAW_SPLITS.values()]" }

# 2. Preprocessing
Run-Step "2/7 Cleaning & preprocessing data" { & $Python src\preprocessing\clean_data.py }

# 3. MapReduce
Run-Step "3/7 MapReduce: category aggregation" { & $Python src\mapreduce\local_driver.py category_aggregation }
Run-Step "3/7 MapReduce: customer aggregation" { & $Python src\mapreduce\local_driver.py customer_aggregation }

# 4. Spark (needs a Spark-compatible JDK)
if (Test-Path $SparkJavaHome) {
    $env:JAVA_HOME = $SparkJavaHome
    $env:PATH = "$SparkJavaHome\bin;" + $env:PATH
} else {
    Write-Warning "Spark-compatible JDK not found at $SparkJavaHome - Spark steps may fail. See docs/SPARK_SETUP.md."
}
$SparkPython = $Python
if ($ProjectRoot -match "[()]") {
    $VenvLink = Join-Path $env:TEMP "bda_venv"
    if (-not (Test-Path $VenvLink)) {
        New-Item -ItemType Junction -Path $VenvLink -Target (Resolve-Path ".\venv").Path | Out-Null
    }
    $SparkPython = "$VenvLink\Scripts\python.exe"
    $env:SPARK_HOME = "$VenvLink\Lib\site-packages\pyspark"
    $env:PYSPARK_PYTHON = $SparkPython
}
Run-Step "4/7 Spark DataFrame processing" { & $SparkPython src\spark\data_processing.py }
Run-Step "4/7 Spark SQL (Hive-style) analytics" { & $SparkPython src\spark\hive_style_analytics.py }

# 5. Fraud detection model
Run-Step "5/7 Training & evaluating fraud detection models" { & $Python src\fraud_detection\train_model.py }
Run-Step "5/7 Verifying the saved model" { & $Python src\fraud_detection\verify_model.py }

# 6. NoSQL
Run-Step "6/7 Loading analytics into MongoDB/mongomock" { & $Python src\nosql\store_analytics.py }

# 7. Visualization
Run-Step "7/7 Generating charts" { & $Python src\visualization\plots.py }

Write-Host "`nPipeline complete. See results/ for metrics, figures, and outputs." -ForegroundColor Green
