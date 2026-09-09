import gc
import os

import joblib
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

DROP_COLS = ["TransactionID", "TransactionDT", "isFraud"]
PROCESSED_DIR = "data/processed/"
label_encoders = {}


def get_dtype_dict(csv_path, usecols):
    sample = pd.read_csv(csv_path, nrows=5000, usecols=usecols)
    dtype_dict = {}
    skip = {"TransactionID", "isFraud", "TransactionDT"}
    for col in sample.columns:
        if col in skip:
            continue
        if sample[col].dtype == object:
            continue
        try:
            pd.to_numeric(sample[col], errors="raise")
            dtype_dict[col] = np.float32
        except (ValueError, TypeError):
            continue
    return dtype_dict


def time_based_split(df, split_quantile=0.8):
    split_point = df["TransactionDT"].quantile(split_quantile)
    print(f"Split point: {split_point:,.0f} seconds")
    train = df[df["TransactionDT"] <= split_point].copy()
    test = df[df["TransactionDT"] > split_point].copy()
    print(f"Train: {len(train):,} rows | Fraud rate: {train['isFraud'].mean():.3%}")
    print(f"Test:  {len(test):,} rows  | Fraud rate: {test['isFraud'].mean():.3%}")
    return train, test


def encode_categoricals(train, test):
    global label_encoders
    protected = {"isFraud", "TransactionID", "TransactionDT"}
    cat_cols = [
        c for c in train.select_dtypes(include=["object"]).columns if c not in protected
    ]
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col].astype(str))
        le_classes = set(le.classes_)
        test[col] = (
            test[col]
            .astype(str)
            .apply(lambda x: x if x in le_classes else "__unknown__")
        )
        if "__unknown__" not in le_classes:
            le.classes_ = np.append(le.classes_, "__unknown__")
        test[col] = le.transform(test[col])
        label_encoders[col] = le
    print(f"Encoded {len(cat_cols)} categorical columns")
    return train, test


def apply_smote(X_train, Y_train, sampling_strategy=0.1, random_state=42):
    if isinstance(Y_train, pd.DataFrame):
        Y_train = Y_train.iloc[:, 0]
    Y_train = Y_train.astype(int)
    X_train = X_train.reset_index(drop=True)
    Y_train = Y_train.reset_index(drop=True)
    print(f"X_train rows: {len(X_train):,} | Y_train rows: {len(Y_train):,}")
    assert len(X_train) == len(Y_train), f"Misaligned: {len(X_train)} vs {len(Y_train)}"
    print(
        f"Before SMOTE � fraud: {int(Y_train.sum()):,} / {len(Y_train):,} ({Y_train.mean():.3%})"
    )
    sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_res, Y_res = sm.fit_resample(X_train, Y_train)
    print(
        f"After SMOTE  � fraud: {int(Y_res.sum()):,} / {len(Y_res):,} ({pd.Series(Y_res).mean():.3%})"
    )
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(
        Y_res, name="isFraud"
    )


def save_processed(X_train, X_test, Y_train, Y_test):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    X_train.to_parquet(f"{PROCESSED_DIR}X_train.parquet", index=False)
    X_test.to_parquet(f"{PROCESSED_DIR}X_test.parquet", index=False)
    Y_train.to_frame().to_parquet(f"{PROCESSED_DIR}Y_train.parquet", index=False)
    Y_test.to_frame().to_parquet(f"{PROCESSED_DIR}Y_test.parquet", index=False)
    joblib.dump(label_encoders, f"{PROCESSED_DIR}label_encoders.pkl")
    print(f"Saved to {PROCESSED_DIR}")


def run_preprocessing_pipeline(raw_transaction_path, raw_identity_path):
    from src.data.features import add_frequency_encoding, engineer_features

    # 1. Load with memory optimization
    print("Loading data...")
    trans_sample = pd.read_csv(raw_transaction_path, nrows=10000)
    ident_sample = pd.read_csv(raw_identity_path, nrows=10000)

    trans_missing = trans_sample.isnull().mean()
    ident_missing = ident_sample.isnull().mean()

    trans_drop = set(trans_missing[trans_missing > 0.9].index) - {
        "isFraud",
        "TransactionID",
        "TransactionDT",
    }
    ident_drop = set(ident_missing[ident_missing > 0.9].index) - {"TransactionID"}

    print(f"Dropping {len(trans_drop)} high-missing transaction cols")
    print(f"Dropping {len(ident_drop)} high-missing identity cols")

    trans_cols = [c for c in trans_sample.columns if c not in trans_drop]
    ident_cols = [c for c in ident_sample.columns if c not in ident_drop]
    del trans_sample, ident_sample

    trans_dtypes = get_dtype_dict(raw_transaction_path, trans_cols)
    ident_dtypes = get_dtype_dict(raw_identity_path, ident_cols)

    print("Loading transaction file...")
    trans = pd.read_csv(raw_transaction_path, usecols=trans_cols, dtype=trans_dtypes)
    print(f"Transaction: {trans.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print("Loading identity file...")
    ident = pd.read_csv(raw_identity_path, usecols=ident_cols, dtype=ident_dtypes)
    print(f"Identity: {ident.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    df = trans.merge(ident, on="TransactionID", how="left")
    del trans, ident
    gc.collect()
    print(f"Joined shape: {df.shape}")

    # 2. Feature engineering
    print("\nEngineering features...")
    df = engineer_features(df)

    # 3. Split
    print("\nSplitting...")
    train, test = time_based_split(df)
    del df
    gc.collect()

    # 4. Frequency encoding � fit on train only
    print("\nFrequency encoding...")
    from src.data.features import add_frequency_encoding

    train = add_frequency_encoding(train, fit=True)
    test = add_frequency_encoding(test, fit=False)

    # 5. Encode categoricals � fit on train only
    print("\nEncoding categoricals...")
    train, test = encode_categoricals(train, test)

    # 6. Separate X and y � inline, no function call
    print("\nSeparating features and target...")
    drop = [c for c in DROP_COLS if c in train.columns]

    X_train = train.drop(columns=drop).reset_index(drop=True)
    Y_train = train["isFraud"].reset_index(drop=True)
    X_test = test.drop(columns=drop).reset_index(drop=True)
    Y_test = test["isFraud"].reset_index(drop=True)

    del train, test
    gc.collect()

    print(f"X_train: {len(X_train):,} rows | Y_train: {len(Y_train):,} rows")
    print(f"X_test:  {len(X_test):,} rows  | Y_test:  {len(Y_test):,} rows")

    # 7. SMOTE on training only
    print("\nApplying SMOTE...")
    X_train, Y_train = apply_smote(X_train, Y_train)

    # 8. Save
    print("\nSaving...")
    save_processed(X_train, X_test, Y_train, Y_test)

    return X_train, X_test, Y_train, Y_test
