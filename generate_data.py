import pandas as pd
import numpy as np
from pathlib import Path

# Required assignment seed
np.random.seed(44)

# Output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# 1. AGENT REFERENCE DATA
# -----------------------------

agents = pd.DataFrame({
    "agent_id": [f"AG{i:03d}" for i in range(1, 21)],
    "county": [
        "Nairobi",
        "Mombasa",
        "Kisumu",
        "Nakuru",
        "Laikipia",
        "Samburu",
        "Meru",
        "Kiambu",
        "Machakos",
        "Kajiado",
        "Nyeri",
        "Bungoma",
        "Kakamega",
        "Garissa",
        "Turkana",
        "Uasin Gishu",
        "Homa Bay",
        "Kilifi",
        "Narok",
        "Bomet"
    ]
})

# Deliberately add one duplicate agent reference
agents = pd.concat(
    [
        agents,
        pd.DataFrame({
            "agent_id": ["AG005"],
            "county": ["Laikipia"]
        })
    ],
    ignore_index=True
)

# -----------------------------
# 2. TRANSACTION DATA
# -----------------------------

n = 100

transaction_ids = [f"TXN{i:04d}" for i in range(1, n + 1)]

agent_ids = np.random.choice(
    [f"AG{i:03d}" for i in range(1, 21)],
    size=n
)

txn_types = np.random.choice(
    ["Deposit", "Withdrawal"],
    size=n,
    p=[0.55, 0.45]
)

amounts = np.random.randint(
    100,
    15000,
    size=n
).astype(object)

dates = pd.date_range(
    start="2026-01-01",
    periods=100,
    freq="D"
).strftime("%Y-%m-%d").tolist()

txns = pd.DataFrame({
    "txn_id": transaction_ids,
    "agent_id": agent_ids,
    "txn_type": txn_types,
    "amount": amounts,
    "txn_date": dates
})

# -----------------------------
# 3. INTRODUCE DATA QUALITY ISSUES
# -----------------------------

# Duplicate transaction
txns = pd.concat(
    [
        txns,
        txns.iloc[[9]]
    ],
    ignore_index=True
)

# Invalid transaction types
txns.loc[20, "txn_type"] = "Transfer"
txns.loc[21, "txn_type"] = "deposit "
txns.loc[22, "txn_type"] = " WITHDRAWAL"

# Missing and invalid amounts
txns.loc[30, "amount"] = np.nan
txns.loc[31, "amount"] = "abc"
txns.loc[32, "amount"] = "4,500"

# Suspicious extreme amount
txns.loc[40, "amount"] = 45000

# Invalid dates
txns.loc[50, "txn_date"] = "not-a-date"

# Alternative valid date formats
txns.loc[60, "txn_date"] = "2026/03/02"
txns.loc[61, "txn_date"] = "03-03-2026"

# Unknown agent
txns.loc[70, "agent_id"] = "AG999"

# Whitespace in agent ID
txns.loc[71, "agent_id"] = " AG010 "

# Missing date
txns.loc[80, "txn_date"] = np.nan

# More missing amount
txns.loc[90, "amount"] = np.nan

# Save raw datasets
txns.to_csv(
    OUTPUT_DIR / "raw_transactions.csv",
    index=False
)

agents.to_csv(
    OUTPUT_DIR / "agents.csv",
    index=False
)

print("=" * 60)
print("DATA GENERATION COMPLETE")
print("=" * 60)

print(f"Transactions generated: {len(txns)}")
print(f"Agents generated: {len(agents)}")

print("\nFiles created:")
print(OUTPUT_DIR / "raw_transactions.csv")
print(OUTPUT_DIR / "agents.csv")

print("\nTransaction preview:")
print(txns.head())

print("\nAgent preview:")
print(agents.head())

print("\nSeed: 44")