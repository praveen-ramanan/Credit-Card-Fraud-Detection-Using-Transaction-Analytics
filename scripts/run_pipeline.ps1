# ============================================================================
# End-to-end pipeline runner (Windows PowerShell, local/demo mode).
#
# Runs every stage of the project in the correct order:
#   1. Synthetic data generation (skipped if data/raw/transactions.csv exists)
#   2. Data cleaning & preprocessing (pandas)
#   3. MapReduce aggregations (local, no Hadoop needed)
#   4. Spark DataFrame processing + Spark SQL (Hive-style) analytics
#   5. Fraud detection model training & evaluation
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

# 1. Data generation (only if no raw dataset present)
if (-not (Test-Path "data\raw\transactions.csv")) {
    Run-Step "1/7 Generating synthetic dataset" { & $Python scripts\generate_synthetic_data.py }
} else {
    Write-Host "`n=== 1/7 Raw dataset already present - skipping generation ===" -ForegroundColor Cyan
}

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
Run-Step "4/7 Spark DataFrame processing" { & $Python src\spark\data_processing.py }
Run-Step "4/7 Spark SQL (Hive-style) analytics" { & $Python src\spark\hive_style_analytics.py }

# 5. Fraud detection model
Run-Step "5/7 Training & evaluating fraud detection models" { & $Python src\fraud_detection\train_model.py }

# 6. NoSQL
Run-Step "6/7 Loading analytics into MongoDB/mongomock" { & $Python src\nosql\store_analytics.py }

# 7. Visualization
Run-Step "7/7 Generating charts" { & $Python src\visualization\plots.py }

Write-Host "`nPipeline complete. See results/ for metrics, figures, and outputs." -ForegroundColor Green
