import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

RESULTS_DIR = "data/model_results"


def evaluate_model(model, X_test, Y_test, threshold=0.5, model_name="model"):

    Y_prob = model.predict_proba(X_test)[:, 1]

    Y_pred = (Y_prob >= threshold).astype(int)

    auc_roc = roc_auc_score(Y_test, Y_prob)

    auc_pr = average_precision_score(Y_test, Y_prob)

    precision = precision_score(Y_test, Y_pred, zero_division=0)

    recall = recall_score(Y_test, Y_pred, zero_division=0)

    f1 = f1_score(Y_test, Y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "threshold": threshold,
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(false_positive_rate, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    print(f"\n{'='*45}")
    print(f"  {model_name}")
    print(f"{'='*45}")
    print(f"  AUC-ROC:            {auc_roc:.4f}")
    print(f"  AUC-PR:             {auc_pr:.4f}")
    print(f"  Precision:          {precision:.4f}")
    print(f"  Recall:             {recall:.4f}")
    print(f"  F1:                 {f1:.4f}")
    print(f"  False positive rate:{false_positive_rate:.4f}")
    print(f"  Confusion matrix:")
    print(f"    TP={tp:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TN={tn:,}")
    print(f"{'='*45}\n")

    return results


def log_results(results: dict):

    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_path = os.path.join(RESULTS_DIR, "model_log.json")

    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)
    else:
        log = []

    log.append(results)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"Results logged to {log_path}")


def compare_runs():
    log_path = os.path.join(RESULTS_DIR, "model_log.json")

    if not os.path.exists(log_path):
        print("No runs logged yet")
        return

    with open(log_path, "r") as f:
        log = json.load(f)

    df = pd.DataFrame(log)[
        [
            "model_name",
            "auc_roc",
            "auc_pr",
            "precision",
            "recall",
            "f1",
            "false_positive_rate",
            "timestamp",
        ]
    ]

    print(df.to_string(index=False))

    return df


# pyrefly: ignore [missing-import]
