# HDFS Setup & Usage

## Why HDFS for this project

HDFS (Hadoop Distributed File System) is designed to store very large
files reliably across many commodity machines by splitting them into
blocks (default 128 MB) and replicating each block (default 3x) across
different nodes. For a banking transaction system, the relevant properties
are:

- **Fault tolerance** - a disk/node failure does not lose data, because
  blocks are replicated elsewhere in the cluster.
- **Horizontal scalability** - transaction volume grows by adding nodes,
  not by buying a single bigger machine.
- **High-throughput batch access** - well suited to the batch analytics
  (MapReduce, Hive, Spark) this project runs, as opposed to low-latency
  single-record lookups (which is what the NoSQL/MongoDB stage is for
  instead - see docs/NOSQL_SETUP.md).

This project treats HDFS as the storage layer that sits between raw data
ingestion and the Hive/MapReduce/Spark analytics stages. Locally, without
a cluster, the same directory layout is simulated on the local filesystem
(`data/raw/`, `data/processed/`) so every downstream script works
unchanged in both modes - config/config.py resolves those paths, and only
the ingestion step differs (`hdfs dfs -put` on a cluster vs. plain file
placement locally).

## Local demo mode (what actually runs on this machine)

No Hadoop installation is required to build, run, or demo this project.
`data/raw/` and `data/processed/` on the local filesystem stand in for
HDFS paths. Every script in `src/` reads/writes these local paths via
`config/config.py`.

## Real HDFS setup (for the cluster demonstration / viva)

The most reliable way to get a real single-node HDFS instance on a Windows
11 laptop is **WSL2 + a pseudo-distributed Hadoop install** (native
Windows Hadoop requires `winutils.exe` and is far more fragile).

### 1. Install WSL2 + Ubuntu (if not already installed)

```powershell
wsl --install -d Ubuntu
```

### 2. Inside the WSL2 Ubuntu shell: install Java + Hadoop

```bash
sudo apt update && sudo apt install -y openjdk-11-jdk ssh
wget https://downloads.apache.org/hadoop/common/hadoop-3.3.6/hadoop-3.3.6.tar.gz
tar -xzf hadoop-3.3.6.tar.gz
mv hadoop-3.3.6 ~/hadoop
```

Set environment variables (add to `~/.bashrc`):

```bash
export HADOOP_HOME=~/hadoop
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

Configure `core-site.xml`, `hdfs-site.xml` for pseudo-distributed mode
(single node, replication factor 1) - follow the official Apache Hadoop
"Single Node Setup" guide:
`https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SingleCluster.html`

### 3. Format the namenode and start HDFS

```bash
hdfs namenode -format
start-dfs.sh
jps   # should show NameNode, DataNode, SecondaryNameNode
```

### 4. Ingest the project's transaction data into HDFS

From the WSL2 shell, with this project accessible under `/mnt/d/PROJECTS/BDA`:

```bash
hdfs dfs -mkdir -p /user/bda/data/raw
hdfs dfs -mkdir -p /user/bda/data/processed

hdfs dfs -put /mnt/d/PROJECTS/BDA/data/raw/transactions.csv /user/bda/data/raw/
hdfs dfs -put /mnt/d/PROJECTS/BDA/data/processed/transactions_clean.csv /user/bda/data/processed/

# Verify
hdfs dfs -ls /user/bda/data/raw
hdfs dfs -ls /user/bda/data/processed
hdfs dfs -du -h /user/bda/data
hdfs dfs -cat /user/bda/data/raw/transactions.csv | head -5
```

(Equivalent commands are collected in `scripts/hdfs_ingest.sh`.)

### 5. Point Hive/MapReduce Streaming at HDFS

- `src/hive/hql/create_tables.hql` already expects the cleaned CSV at
  `/user/bda/data/processed/transactions_clean` on HDFS.
- For MapReduce on a real cluster, submit via Hadoop Streaming pointing
  `-input` at the HDFS path instead of a local file (see
  `docs/MAPREDUCE_SETUP.md`).

### Stopping the cluster

```bash
stop-dfs.sh
```
