# Deploying the dashboard on Streamlit Community Cloud

The deployed app is the **serving layer** only: `app.py` reads the committed
files in `deploy/` (cleaned transactions, saved model, test-set scores,
precomputed fraud alerts) and
the committed `results/` outputs. No Spark, Hadoop, Hive or model training
runs in the cloud. Those are the offline batch layer, run locally with
`scripts/run_pipeline.ps1`.

Streamlit Community Cloud installs `requirements.txt` (dashboard runtime
only, pinned) from the repository root. Measured locally with
`scripts/measure_memory.py --deploy`, the app peaks at about 605 MB RSS
across all six pages, under Community Cloud's ~690 MB guaranteed memory.

The four `deploy/` files are written by `scripts/build_deploy_artifacts.py`:
`transactions_dashboard.parquet` (45.5 MB, no `trans_num` column),
`scored_test_transactions.parquet` (19.2 MB), `fraud_alerts.parquet`
(0.3 MB, the 5,321 test rows scoring at least 0.5) and `best_model.joblib`
(1.4 MB). `BDA_DASHBOARD_DATA`, `BDA_DASHBOARD_MODEL`, `BDA_DASHBOARD_SCORED`
and `BDA_DASHBOARD_ALERTS` force a specific file.

## 1. Push the branch and merge it into `main`

From the project root:

```powershell
git push -u origin deploy-streamlit
```

Every file in `deploy/` is under GitHub's recommended 50 MB (the largest,
`transactions_dashboard.parquet`, is 45.5 MB), so the push prints no
large-file warning. The build script fails if any file reaches 95 MB.

Then merge, either through a pull request on GitHub
(`deploy-streamlit` -> `main`) or locally:

```powershell
git checkout main
git merge --ff-only deploy-streamlit
git push origin main
```

## 2. Create the app

1. Go to <https://share.streamlit.io> and sign in with the GitHub account
   that owns the repository (authorise Streamlit's GitHub access if asked).
2. **Create app** -> deploy from a GitHub repository.
3. Fill in:
   - **Repository**: `praveen-ramanan/Credit-Card-Fraud-Detection-Using-Transaction-Analytics`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: choose a subdomain (optional)
4. Open **Advanced settings**:
   - **Python version**: `3.11` (the version the model was trained and verified with).
   - **Secrets**: leave empty for the default demo (see section 3).
5. **Deploy**. The first build installs the requirements (the XGBoost wheel
   for Linux is large, so allow a few minutes) and then starts the app.

The app is ready when the sidebar shows the six sections. The first page view
after a start loads the 1.85M-row dataset, which takes a few seconds.

No `.streamlit/config.toml` is needed: Community Cloud runs headless by
default and the app uses Streamlit's default theme.

## 3. Secrets

Secrets go in the app's **Settings -> Secrets** box (TOML), or under
**Advanced settings** while creating the app. Never commit them:
`.streamlit/secrets.toml` is in `.gitignore`.

| Setup | Secrets to set |
|---|---|
| Default demo: Fraud alerts page uses **mongomock** (in-memory) | none |
| Real MongoDB (Atlas) for the Fraud alerts page | `BDA_MONGO_URI` (and optionally `BDA_MONGO_DB`) |

Example for MongoDB Atlas:

```toml
BDA_MONGO_URI = "mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
BDA_MONGO_DB = "fraud_analytics"
```

- Root-level secrets become environment variables, which is how
  `config/config.py` reads them. `src/dashboard/settings_from_secrets.py` also copies
  TOML booleans (for example `BDA_MONGO_FALLBACK = false`), which Streamlit
  itself skips.
- In Atlas, allow connections from anywhere (**Network Access** ->
  `0.0.0.0/0`), because Community Cloud has no fixed outbound IP. Use a
  database user with access to this database only.
- If the `fraud_alerts` collection is empty, or was filled by the old top-500
  loader, the app fills it with every test transaction scoring at least
  `ALERT_MIN_PROBABILITY` (0.5) on first use (the same loader as
  `src/nosql/store_analytics.py`), from `deploy/fraud_alerts.parquet`, or from
  the full scored file if that precomputed file is missing or was built with
  a different floor.
- With `BDA_MONGO_URI` set, an unreachable server is shown as an error on the
  Fraud alerts page. The app never falls back to mongomock silently. The page
  shows which backend is in use.
- Settings are read once when the app process starts. **Reboot the app after
  changing secrets** (section 4).

`BDA_DATA_URL` is **not** used: all data files fit in the repository, so
there is no download at startup.

## 4. Logs, reboot and sleeping

- **Logs**: open the app, then use **Manage app** (bottom-right corner, only
  visible to you when signed in). The panel shows the build and runtime log,
  which covers pip install errors, Python tracebacks and memory warnings.
- **Reboot**: in **Manage app**, open the menu (**...**) and choose
  **Reboot app**. You can also do this from your app list on
  share.streamlit.io. Reboot after changing secrets, or if the app hits its
  resource limits.
- **Redeploys**: every push to `main` redeploys the app automatically.
- **Sleeping**: an app with no traffic for 12 hours goes to sleep. The next
  visitor sees a "wake up" button and waits for it to restart. **Open the app
  a few minutes before a demo.**

## 5. If something goes wrong

| Symptom in the app or log | Likely cause | Fix |
|---|---|---|
| "Required pipeline output is missing ... `deploy/...`" | `deploy/` files not pushed | Check `git ls-files deploy` lists all four files and that `main` contains them |
| "This app has gone over its resource limits" | Memory | Reboot. If it repeats, check the log and compare with `scripts/measure_memory.py --deploy` locally |
| pip error or very slow build on `xgboost` | Large Linux wheel (installs NVIDIA NCCL as a dependency) | Replace `xgboost==3.2.0` with `xgboost-cpu==3.2.0` in `requirements.txt` |
| `StaleModelError` / model load error | Package versions differ from the pins | Check that Python 3.11 was selected and `requirements.txt` was not changed |
| Fraud alerts: "Could not reach the MongoDB server configured in BDA_MONGO_URI" | Atlas network access or credentials | Allow `0.0.0.0/0`, check user/password, reboot |

## 6. After deploying

Check the live app by hand:

1. The public URL opens and the sidebar lists the six sections.
2. Each page loads without an error: Overview, Transaction analytics, Fraud
   detection, Model performance, Fraud alerts and CSV analysis.
3. Overview shows the data source `deploy/transactions_dashboard.parquet`, and
   Fraud alerts shows **Backend: mongomock (in-memory demo)**.
4. On Fraud detection, **Load a fraud example** then **Assess transaction**
   gives risk level **High**, and **Load a legitimate example** then **Assess
   transaction** gives **Low**.
5. Model performance shows the stored test metrics (ROC-AUC 0.9970, PR-AUC
   0.8554).
