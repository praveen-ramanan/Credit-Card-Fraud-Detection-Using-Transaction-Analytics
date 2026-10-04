"""
Generate a synthetic credit-card transaction dataset for local development
and demonstration.

WHY THIS EXISTS
----------------
The recommended real-world dataset for this project is the Kaggle
"Credit Card Transactions Fraud Detection Dataset" (Sparkov-generated,
published by kartik2112). It requires a Kaggle account to download
(see docs/DATASET.md). To allow the project to be built, run, and
demonstrated immediately without that manual step, this script generates
a synthetic dataset with the EXACT SAME COLUMN SCHEMA. Every module in
src/ is written against that schema, so swapping this file for the real
Kaggle CSV (fraudTrain.csv / fraudTest.csv, renamed/merged to
data/raw/transactions.csv) requires no code changes.

The synthetic data is NOT a substitute for real results in the final
report -- it exists purely so the pipeline can be developed and demoed
locally. Fraud is injected using simple, documented rules (odd hours,
high amounts, high-risk categories, far customer-merchant distance) so
the classifier has real, learnable signal rather than pure noise.

Run:
    python scripts/generate_synthetic_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import config as cfg

RNG = np.random.default_rng(cfg.SYNTHETIC_RANDOM_SEED)

CATEGORIES = [
    "grocery_pos", "grocery_net", "gas_transport", "home", "shopping_pos",
    "shopping_net", "kids_pets", "personal_care", "health_fitness",
    "entertainment", "food_dining", "misc_pos", "misc_net", "travel",
]
# categories that are, in this synthetic world, more associated with fraud
HIGH_RISK_CATEGORIES = {"shopping_net", "misc_net", "travel", "grocery_net"}

STATES = ["MH", "KA", "TN", "DL", "GJ", "UP", "WB", "RJ", "MP", "KL"]
CITIES_BY_STATE = {
    "MH": ["Mumbai", "Pune", "Nagpur"],
    "KA": ["Bengaluru", "Mysuru"],
    "TN": ["Chennai", "Coimbatore"],
    "DL": ["New Delhi"],
    "GJ": ["Ahmedabad", "Surat"],
    "UP": ["Lucknow", "Kanpur"],
    "WB": ["Kolkata", "Howrah"],
    "RJ": ["Jaipur", "Udaipur"],
    "MP": ["Bhopal", "Indore"],
    "KL": ["Kochi", "Thiruvananthapuram"],
}
JOBS = [
    "Software Engineer", "Teacher", "Doctor", "Accountant", "Nurse",
    "Sales Executive", "Civil Engineer", "Graphic Designer", "Farmer",
    "Bank Clerk", "Marketing Manager", "Electrician", "Chef", "Lawyer",
]
FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Ananya", "Diya", "Ishaan",
               "Kavya", "Meera", "Rohan", "Sara", "Vihaan", "Zara",
               "Arjun", "Priya", "Karan", "Neha"]
LAST_NAMES = ["Sharma", "Verma", "Iyer", "Nair", "Patel", "Gupta",
              "Reddy", "Singh", "Das", "Menon", "Rao", "Khan"]
MERCHANT_PREFIXES = ["Fresh", "Prime", "Metro", "City", "Star", "Global",
                      "National", "Quick", "Value", "Elite", "Urban", "Royal"]
MERCHANT_SUFFIXES = ["Mart", "Store", "Traders", "Services", "Retail",
                      "Wholesale", "Shop", "Outlet", "Depot", "Hub"]


def _gen_customers(n):
    ids = np.arange(1000, 1000 + n)
    cc_nums = RNG.integers(4_000_000_000_000_000, 4_999_999_999_999_999, size=n, endpoint=False)
    first = RNG.choice(FIRST_NAMES, size=n)
    last = RNG.choice(LAST_NAMES, size=n)
    gender = RNG.choice(["M", "F"], size=n)
    state = RNG.choice(STATES, size=n)
    city = [RNG.choice(CITIES_BY_STATE[s]) for s in state]
    lat = RNG.uniform(8.0, 34.0, size=n).round(6)   # rough India lat range
    long = RNG.uniform(70.0, 89.0, size=n).round(6)  # rough India long range
    city_pop = RNG.integers(20_000, 12_000_000, size=n)
    job = RNG.choice(JOBS, size=n)
    dob = pd.to_datetime(
        RNG.integers(pd.Timestamp("1955-01-01").value // 10**9,
                     pd.Timestamp("2003-01-01").value // 10**9, size=n),
        unit="s",
    ).date
    zip_codes = RNG.integers(100000, 899999, size=n)

    return pd.DataFrame({
        "customer_id": ids,
        "cc_num": cc_nums,
        "first": first,
        "last": last,
        "gender": gender,
        "city": city,
        "state": state,
        "zip": zip_codes,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "job": job,
        "dob": dob,
    })


def _gen_merchants(n):
    names = [f"{RNG.choice(MERCHANT_PREFIXES)} {RNG.choice(MERCHANT_SUFFIXES)} #{i}"
              for i in range(n)]
    categories = RNG.choice(CATEGORIES, size=n)
    lat = RNG.uniform(8.0, 34.0, size=n).round(6)
    long = RNG.uniform(70.0, 89.0, size=n).round(6)
    return pd.DataFrame({
        "merchant": [f"fraud_{n_}" if False else n_ for n_ in names],  # keep names readable
        "category": categories,
        "merch_lat": lat,
        "merch_long": long,
    })


def generate(num_transactions: int, num_customers: int, num_merchants: int,
             fraud_rate: float) -> pd.DataFrame:
    customers = _gen_customers(num_customers)
    merchants = _gen_merchants(num_merchants)

    cust_idx = RNG.integers(0, num_customers, size=num_transactions)
    merch_idx = RNG.integers(0, num_merchants, size=num_transactions)

    txn_customers = customers.iloc[cust_idx].reset_index(drop=True)
    txn_merchants = merchants.iloc[merch_idx].reset_index(drop=True)

    # Timestamps spread over 1 year, with realistic hour-of-day weighting
    # (more transactions during daytime/evening than late night).
    start = pd.Timestamp("2024-01-01")
    day_offsets = RNG.integers(0, 365, size=num_transactions)
    hour_weights = np.array([
        0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.6, 1.0, 1.3, 1.5, 1.6, 1.7,
        1.8, 1.7, 1.6, 1.5, 1.6, 1.8, 2.0, 1.9, 1.6, 1.2, 0.8, 0.4,
    ])
    hour_weights = hour_weights / hour_weights.sum()
    hours = RNG.choice(24, size=num_transactions, p=hour_weights)
    minutes = RNG.integers(0, 60, size=num_transactions)
    seconds = RNG.integers(0, 60, size=num_transactions)
    timestamps = (start
                  + pd.to_timedelta(day_offsets, unit="D")
                  + pd.to_timedelta(hours, unit="h")
                  + pd.to_timedelta(minutes, unit="m")
                  + pd.to_timedelta(seconds, unit="s"))

    # Base transaction amounts: log-normal (many small, few large purchases)
    amounts = RNG.lognormal(mean=3.2, sigma=1.0, size=num_transactions).round(2)
    amounts = np.clip(amounts, 1.0, 25000.0)

    is_high_risk_cat = txn_merchants["category"].isin(HIGH_RISK_CATEGORIES).to_numpy()
    is_odd_hour = ((hours >= 0) & (hours <= 4))

    # crude customer-merchant distance signal (deg -> rough km factor)
    dist = np.sqrt((txn_customers["lat"].to_numpy() - txn_merchants["merch_lat"].to_numpy()) ** 2
                    + (txn_customers["long"].to_numpy() - txn_merchants["merch_long"].to_numpy()) ** 2)
    is_far = dist > np.percentile(dist, 90)

    # Fraud probability rises with risk signals, then we sample to hit the
    # target overall fraud_rate roughly.
    risk_score = (0.5 * is_high_risk_cat + 0.3 * is_odd_hour + 0.2 * is_far)
    base_p = fraud_rate * 0.4
    fraud_p = np.clip(base_p + risk_score * fraud_rate * 6, 0, 0.35)
    is_fraud = (RNG.random(num_transactions) < fraud_p).astype(int)

    # Fraudulent transactions tend to be higher amounts (card testing /
    # cash-out patterns) -- inflate amount for fraud rows.
    fraud_multiplier = np.where(is_fraud == 1, RNG.uniform(2.0, 8.0, size=num_transactions), 1.0)
    amounts = (amounts * fraud_multiplier).round(2)

    trans_num = [f"txn_{i:09d}" for i in range(num_transactions)]
    unix_time = (timestamps.view("int64") // 10**9)

    df = pd.DataFrame({
        "trans_date_trans_time": timestamps.astype(str),
        "cc_num": txn_customers["cc_num"].to_numpy(),
        "merchant": txn_merchants["merchant"].to_numpy(),
        "category": txn_merchants["category"].to_numpy(),
        "amt": amounts,
        "first": txn_customers["first"].to_numpy(),
        "last": txn_customers["last"].to_numpy(),
        "gender": txn_customers["gender"].to_numpy(),
        "city": txn_customers["city"].to_numpy(),
        "state": txn_customers["state"].to_numpy(),
        "zip": txn_customers["zip"].to_numpy(),
        "lat": txn_customers["lat"].to_numpy(),
        "long": txn_customers["long"].to_numpy(),
        "city_pop": txn_customers["city_pop"].to_numpy(),
        "job": txn_customers["job"].to_numpy(),
        "dob": txn_customers["dob"].to_numpy(),
        "trans_num": trans_num,
        "unix_time": unix_time,
        "merch_lat": txn_merchants["merch_lat"].to_numpy(),
        "merch_long": txn_merchants["merch_long"].to_numpy(),
        "is_fraud": is_fraud,
    })

    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    df.insert(0, "index", np.arange(len(df)))
    return df


def main():
    df = generate(
        num_transactions=cfg.SYNTHETIC_NUM_TRANSACTIONS,
        num_customers=cfg.SYNTHETIC_NUM_CUSTOMERS,
        num_merchants=cfg.SYNTHETIC_NUM_MERCHANTS,
        fraud_rate=cfg.SYNTHETIC_FRAUD_RATE,
    )
    cfg.RAW_TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.RAW_TRANSACTIONS_FILE, index=False)

    fraud_count = int(df["is_fraud"].sum())
    print(f"Generated {len(df):,} synthetic transactions -> {cfg.RAW_TRANSACTIONS_FILE}")
    print(f"Fraud transactions: {fraud_count:,} ({fraud_count / len(df) * 100:.3f}%)")
    print(f"Customers: {df['cc_num'].nunique():,} | Merchants: {df['merchant'].nunique():,} "
          f"| Categories: {df['category'].nunique()}")


if __name__ == "__main__":
    main()
