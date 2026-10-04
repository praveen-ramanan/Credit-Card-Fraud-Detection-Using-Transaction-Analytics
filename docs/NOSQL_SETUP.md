# NoSQL (MongoDB) Setup & Usage

## Local demo mode (default, zero install)

`src/nosql/mongo_client.py` tries to connect to a real MongoDB server at
`BDA_MONGO_URI` (default `mongodb://localhost:27017`) with a short
timeout; if none is reachable, it transparently falls back to
`mongomock`, an in-memory library implementing the same `pymongo` API.
This means `src/nosql/store_analytics.py` runs identically whether or not
MongoDB is installed - the only difference is data does not persist
between runs under `mongomock`.

```powershell
.\venv\Scripts\python.exe src\nosql\store_analytics.py
```

## Running against a real MongoDB instance

**Option A - local install:**
Download MongoDB Community Server for Windows:
`https://www.mongodb.com/try/download/community`, install it as a
service (default), then just re-run the same script - it will detect the
running server on `localhost:27017` automatically and use it instead of
`mongomock`.

**Option B - free-tier MongoDB Atlas (no local install, still free):**
Create a free M0 cluster at `https://www.mongodb.com/cloud/atlas/register`,
get its connection string, then:

```powershell
$env:BDA_MONGO_URI = "mongodb+srv://<user>:<password>@<cluster-url>/"
.\venv\Scripts\python.exe src\nosql\store_analytics.py
```

Verify which backend was used from the script's first printed line:
`[nosql] Using backend: mongodb` (or `mongomock` for the local fallback).

## What's stored, and why NoSQL fits this data

- **`fraud_alerts`** - one flexible document per high-risk transaction
  (from `results/fraud_detection/scored_transactions.csv`), with a nested
  customer sub-document and model score. Alert-style records like this
  commonly vary in shape (a rule-based alert vs. a model-based alert may
  carry different fields), which fits a document store more naturally
  than a fixed relational schema.
- **`category_analytics`** - the category-level aggregates already
  computed by the Spark/Hive stage, re-stored as documents for fast
  dashboard-style lookups without re-running a Spark job.

`src/nosql/store_analytics.py` also demonstrates MongoDB's own
aggregation pipeline (`$group`, `$sort`) as a second, independent
analytics engine distinct from the Spark/Hive SQL stage - a legitimate
alternative way to compute the same kind of summary directly over stored
documents.

## MongoDB vs Cassandra (why MongoDB was chosen here)

Both are NoSQL databases, but with different data models and strengths:

| | MongoDB | Cassandra |
|---|---|---|
| Data model | Document (JSON-like, nested) | Wide-column (row-per-partition-key) |
| Best fit | Flexible, nested, evolving records | Very high write throughput, time-series-like data spread across many nodes |
| Local setup | Single-binary install, trivial | Needs a multi-node-aware setup (or Docker) even for a single node |
| Query style | Rich query language + aggregation pipeline | CQL (SQL-like), but partition-key-driven access patterns |

MongoDB was chosen for this project because the "why NoSQL" story here is
about flexible, nested fraud-alert documents (not about extreme write
throughput across a multi-datacenter cluster, which is Cassandra's core
strength), and because it is far simpler to install and demo locally on a
single Windows laptop.
