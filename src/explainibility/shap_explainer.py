# pyrefly: ignore [missing-import]
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap

SHAP_SAMPLE_SIZE = 5000


def build_explainer(model, X_train_sample: pd.DataFrame):
    """
    Build a TreeExplainer — SHAP's fast path for tree-based models.
    XGBoost/LightGBM have exact SHAP values computable in O(TLD) time
    where T=trees, L=leaves, D=depth.
    Much faster than KernelExplainer which approximates via sampling.

    X_train_sample is used to set the background distribution —
    what "baseline" the SHAP values are measured against.
    """
    print("Building TreeExplainer...")

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    print("explainer built.")
    return explainer


def compute_shap_values(explainer, X: pd.DataFrame):
    """
    Compute SHAP values for a set of rows.
    Returns array of shape (n_samples, n_features).
    Each value = that feature's contribution to pushing
    the prediction away from the baseline.
    Positive = pushes toward fraud.
    Negative = pushes toward legitimate.
    """

    print(f"Computing shap values for {len(X):,} rows")
    shap_values = explainer.shap_values(X)
    print(f"SHAP values shape: {shap_values.shape}")
    return shap_values


def get_global_importance(shap_values, feature_names, top_n=20) -> pd.DataFrame:
    """
    Global feature importance = mean absolute SHAP value per feature.
    Mean absolute because we care about magnitude of impact,
    not direction — a feature that sometimes helps and sometimes
    hurts is still important.
    """

    importance = (
        pd.DataFrame(
            {"feature": feature_names, "importance": np.abs(shap_values).mean(axis=0)}
        )
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    return importance


def explain_single_prediction(explainer, X_row: pd.DataFrame, top_n=10) -> dict:
    """
    Per-prediction explanation for one transaction.
    This is what the Explainer Agent calls for each scored transaction.

    Returns a structured dict the agent can convert to natural language:
    {
        "base_value": float,        # average model output
        "prediction": float,        # this transaction's score
        "top_features": [
            {"feature": str, "shap_value": float, "actual_value": float},
            ...
        ]
    }
    """

    shap_vals = explainer.shap_values(X_row)

    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    feature_names = X_row.columns.tolist()
    actual_values = X_row.iloc[0].tolist()

    feature_impacts = [
        {
            "feature": feature_names[i],
            "shap_value": round(float(shap_vals[i]), 6),
            "actual_value": (
                round(float(actual_values[i]), 4)
                if isinstance(actual_values[i], (int, float))
                else actual_values[i]
            ),
        }
        for i in range(len(feature_names))
    ]

    feature_impacts.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "base_value": round(float(explainer.expected_value), 6),
        "prediction": round(float(shap_vals.sum() + explainer.expected_value), 6),
        "top_features": feature_impacts[:top_n],
    }


def save_explainer(explainer, path="data/models/shap_explainer.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(explainer, path)
    print(f"Explainer saved to {path}")


def load_explainer(path="data/models/shap_explainer.pkl"):
    return joblib.load(path)
