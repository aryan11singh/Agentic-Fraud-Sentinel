import numpy as np
import pandas as pd
from pandas import DataFrame


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function to apply all engineering steps
    Call this on full joined dataframe before splitting"""

    df = df.copy()
    df = add_time_features(df)
    df = add_missingness_flags(df)
    df = add_frequency_encoding(df)
    df = add_behavioral_aggregates(df)
    df = impute_missing(df)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """TransactionDT is seconds since an arbitrary reference point
    We extract cyclic signals from it- fraud has strong time patterns"""

    df["tx_hour"] = (df["TransactionDT"] // 3600) % 24
    df["tx_day"] = (df["TransactionDT"] // 86400) % 7

    df["tx_hour_sin"] = np.sin(2 * np.pi * df["tx_hour"] / 24)
    df["tx_hour_cos"] = np.cos(2 * np.pi * df["tx_hour"] / 24)

    return df


## Missingness Flags

COLS_TO_FLAG = [
    "id_01",
    "id_02",
    "id_03",
    "id_04",
    "id_05",
    "id_06",
    "id_07",
    "id_08",
    "id_09",
    "id_10",
    "id_11",
    "DeviceType",
    "DeviceInfo",
]


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    """For key columns, add a binary flag Before imputing.
    'No device info attached is also a fraud signal'"""

    for col in COLS_TO_FLAG:
        if col in df.columns:
            df[f"{col}_was_missing"] = df[col].isnull().astype(int)

    return df


## Frequency encoding

FREQ_COLS = [
    "card1",
    "card2",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceInfo",
    "id_30",
    "id_31",
]

freq_maps: dict = {}


def add_frequency_encoding(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """Replace each categorical value with how often it appears.
    Rare values(unusual device, unknown email domain) get low scores

    fti= true -> learn frequencies from this df (use on train)
    fit = false -> apply stored frequencies ( use on test)
    """

    global freq_maps
    for col in FREQ_COLS:
        if col not in df.columns:
            continue
        if fit:
            freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
        df[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0)
    return df


## Behavioral aggregates


def add_behavioral_aggregates(df: pd.DataFrame) -> DataFrame:
    """How does THIS transaction compare to this card's recent history?
    This is where real fraud signal lives — not the transaction in isolation.

    We sort by time first so rolling windows only look backward."""

    df = df.sort_values("TransactionDT").reset_index(drop=True)

    for window_secs, label in [(3600, "1h"), (86400, "24h")]:

        df[f"card1_count_{label}"] = df.groupby("card1")["TransactionDT"].transform(
            lambda x: x.expanding().count()
        )

        df[f"card1_mean_amt_{label}"] = df.groupby("card1")["TransactionAmt"].transform(
            lambda x: x.expanding().mean().shift(1)
        )

    df["amt_deviation"] = df["TransactionAmt"] - df["card1_mean_amt_24h"]

    return df


## Imputation

from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")


def impute_missing(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Fill remaining NaNa
    fit=True  → fit imputers on this data (train only)
    fit=False → transform using already-fitted imputers (test)
    """

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if fit:
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    else:
        df[num_cols] = num_imputer.transform(df[num_cols])
        df[cat_cols] = cat_imputer.transform(df[cat_cols])

    return df
