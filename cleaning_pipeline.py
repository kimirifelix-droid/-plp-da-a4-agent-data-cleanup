import pandas as pd
import numpy as np


def clean_agent_data(txns, agents):
    """
    Clean mobile money transaction and agent reference data.

    Parameters
    ----------
    txns : pandas.DataFrame
        Raw transaction data.

    agents : pandas.DataFrame
        Agent reference data containing agent_id and county.

    Returns
    -------
    tuple
        A tuple containing:
        clean_df : pandas.DataFrame
            Cleaned and enriched transaction data.

        log : pandas.DataFrame
            Data-quality audit log with columns:
            issue, dimension, column, rows affected.
    """

    txns = txns.copy()
    agents = agents.copy()

    audit = []

    def add_log(issue, dimension, column, rows_affected):
        audit.append(
            {
                "issue": issue,
                "dimension": dimension,
                "column": column,
                "rows affected": int(rows_affected),
            }
        )

    # =========================================================
    # 1. REQUIRED COLUMN VALIDATION
    # =========================================================

    required_transaction_columns = {
        "txn_id",
        "agent_id",
        "txn_type",
        "amount",
        "txn_date",
    }

    missing_transaction_columns = (
        required_transaction_columns - set(txns.columns)
    )

    if missing_transaction_columns:
        raise ValueError(
            "Missing transaction columns: "
            + ", ".join(sorted(missing_transaction_columns))
        )

    required_agent_columns = {"agent_id", "county"}

    missing_agent_columns = (
        required_agent_columns - set(agents.columns)
    )

    if missing_agent_columns:
        raise ValueError(
            "Missing agent columns: "
            + ", ".join(sorted(missing_agent_columns))
        )

    # =========================================================
    # 2. UNIQUENESS — DUPLICATE TRANSACTIONS
    # =========================================================

    duplicate_mask = txns.duplicated(
        subset=["txn_id"],
        keep="first"
    )

    duplicate_count = int(duplicate_mask.sum())

    add_log(
        "Duplicate transaction IDs removed",
        "Uniqueness",
        "txn_id",
        duplicate_count
    )

    txns = txns.loc[~duplicate_mask].copy()

    # =========================================================
    # 3. CONSISTENCY — TRANSACTION TYPE
    # =========================================================

    txns["txn_type"] = (
        txns["txn_type"]
        .astype("string")
        .str.strip()
        .str.title()
    )

    valid_transaction_types = {
        "Deposit",
        "Withdrawal"
    }

    invalid_type_mask = ~txns["txn_type"].isin(
        valid_transaction_types
    )

    invalid_type_count = int(invalid_type_mask.sum())

    add_log(
        "Invalid transaction types removed",
        "Validity",
        "txn_type",
        invalid_type_count
    )

    txns = txns.loc[~invalid_type_mask].copy()

    # =========================================================
    # 4. VALIDITY — AMOUNT
    # =========================================================

    amount_before = txns["amount"].copy()

    txns["amount"] = (
        txns["amount"]
        .astype("string")
        .str.replace(",", "", regex=False)
    )

    txns["amount"] = pd.to_numeric(
        txns["amount"],
        errors="coerce"
    )

    invalid_amount_mask = (
        txns["amount"].isna()
        & amount_before.notna()
    )

    invalid_amount_count = int(
        invalid_amount_mask.sum()
    )

    add_log(
        "Non-numeric amounts converted to missing",
        "Validity",
        "amount",
        invalid_amount_count
    )

    # =========================================================
    # 5. COMPLETENESS — MISSING AMOUNTS
    # =========================================================

    txns["amount_imputed"] = txns["amount"].isna()

    missing_amount_count = int(
        txns["amount"].isna().sum()
    )

    if missing_amount_count > 0:

        median_amount = txns["amount"].median()

        if pd.isna(median_amount):
            raise ValueError(
                "Cannot impute amount because no valid "
                "numeric amount exists."
            )

        txns["amount"] = txns["amount"].fillna(
            median_amount
        )

    add_log(
        "Missing amounts imputed using the median",
        "Completeness",
        "amount",
        missing_amount_count
    )

    # =========================================================
    # 6. VALIDITY / TIMELINESS — DATES
    # =========================================================

    date_before = txns["txn_date"].copy()

    txns["txn_date"] = pd.to_datetime(
        txns["txn_date"],
        errors="coerce"
    )

    invalid_date_mask = (
        txns["txn_date"].isna()
        & date_before.notna()
    )

    invalid_date_count = int(
        invalid_date_mask.sum()
    )

    add_log(
        "Invalid dates converted to missing",
        "Validity",
        "txn_date",
        invalid_date_count
    )

    missing_date_mask = txns["txn_date"].isna()

    missing_date_count = int(
        missing_date_mask.sum()
    )

    add_log(
        "Transactions with missing dates removed",
        "Completeness",
        "txn_date",
        missing_date_count
    )

    txns = txns.loc[~missing_date_mask].copy()

    # =========================================================
    # 7. ACCURACY — OUTLIER FLAGGING
    # =========================================================

    txns["is_outlier"] = False

    if not txns.empty:

        q1 = txns["amount"].quantile(0.25)
        q3 = txns["amount"].quantile(0.75)

        iqr = q3 - q1

        lower_fence = q1 - (1.5 * iqr)
        upper_fence = q3 + (1.5 * iqr)

        outlier_mask = (
            (txns["amount"] < lower_fence)
            |
            (txns["amount"] > upper_fence)
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        txns.loc[
            outlier_mask,
            "is_outlier"
        ] = True

    else:
        outlier_count = 0

    add_log(
        "Amounts outside the IQR fences flagged as outliers",
        "Accuracy",
        "amount",
        outlier_count
    )

    # =========================================================
    # 8. AGENT REFERENCE CLEANING
    # =========================================================

    txns["agent_id"] = (
        txns["agent_id"]
        .astype("string")
        .str.strip()
    )

    agents["agent_id"] = (
        agents["agent_id"]
        .astype("string")
        .str.strip()
    )

    duplicate_agent_mask = agents.duplicated(
        subset=["agent_id"],
        keep="first"
    )

    duplicate_agent_count = int(
        duplicate_agent_mask.sum()
    )

    add_log(
        "Duplicate agent reference IDs removed",
        "Uniqueness",
        "agent_id",
        duplicate_agent_count
    )

    agents = agents.loc[
        ~duplicate_agent_mask
    ].copy()

    # =========================================================
    # 9. MERGE TRANSACTIONS WITH AGENT COUNTY
    # =========================================================

    clean_df = txns.merge(
        agents[["agent_id", "county"]],
        on="agent_id",
        how="left",
        validate="many_to_one"
    )

    unknown_agent_mask = clean_df["county"].isna()

    unknown_agent_count = int(
        unknown_agent_mask.sum()
    )

    add_log(
        "Transactions without a matching agent",
        "Consistency",
        "agent_id",
        unknown_agent_count
    )

    # =========================================================
    # 10. FINAL DATA TYPES
    # =========================================================

    clean_df["amount"] = pd.to_numeric(
        clean_df["amount"],
        errors="raise"
    )

    clean_df["txn_date"] = pd.to_datetime(
        clean_df["txn_date"],
        errors="raise"
    )

    clean_df["amount_imputed"] = (
        clean_df["amount_imputed"].astype(bool)
    )

    clean_df["is_outlier"] = (
        clean_df["is_outlier"].astype(bool)
    )

    clean_df = clean_df.sort_values(
        ["txn_date", "txn_id"]
    ).reset_index(drop=True)

    # =========================================================
    # 11. CREATE AUDIT LOG
    # =========================================================

    log = pd.DataFrame(
        audit,
        columns=[
            "issue",
            "dimension",
            "column",
            "rows affected"
        ]
    )

    return clean_df, log