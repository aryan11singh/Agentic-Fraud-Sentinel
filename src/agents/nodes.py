import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.agents.state import FraudDetectionState
from src.explainibility.shap_explainer import explain_single_prediction

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.path.exists(os.path.join(BASE_DIR, "data", "models", "xgboost_production.pkl")):
    DATA_DIR = os.path.join(BASE_DIR, "data")
else:
    DATA_DIR = os.path.join(BASE_DIR, "notebooks", "data")


MODEL_PATH = os.path.join(DATA_DIR, "models", "xgboost_production.pkl")
EXPLAINER_PATH = os.path.join(DATA_DIR, "models", "shap_explainer.pkl")


print("Loading model bundle...")
bundle = joblib.load(MODEL_PATH)
MODEL = bundle["model"]
THRESHOLD = bundle["threshold"]

print("Loading SHAP explainer...")
EXPLAINER = joblib.load(EXPLAINER_PATH)


FEATURE_COLS = MODEL.get_booster().feature_names


def _encode_features(tx_data: dict) -> dict:

    encoded = {}
    for col in FEATURE_COLS:
        val = tx_data.get(col, 0)
        if isinstance(val, str):
            val = 0
        encoded[col] = val

    return encoded


# NODE 1: Risk Scorer


def risk_scorer_node(state: FraudDetectionState) -> dict:
    """
    Loads transaction features, run Xgboost model, returns fraud probability and a risk tier.

    Risk tiers:
    low < threshold* 0.5 -> almost cerrtainly legitimate
    medium threshold * 0.5 to threshold -> borderline
    high >= threshold -> model says fraud
    """

    try:
        tx_data = _encode_features(state["transaction_data"])
        X = pd.DataFrame([tx_data], columns=FEATURE_COLS)
        X = X.reindex(columns=FEATURE_COLS, fill_value=0)

        fraud_prob = float(MODEL.predict_proba(X)[:, 1][0])

        low_cutoff = THRESHOLD * 0.5
        if fraud_prob >= THRESHOLD:
            risk_level = "high"
        elif fraud_prob >= low_cutoff:
            risk_level = "medium"
        else:
            risk_level = "low"

        print(
            f"[RiskScorer] tx={state['transaction_id']} "
            f"prob={fraud_prob:.4f} risk={risk_level}"
        )

        return {
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
        }

    except Exception as e:
        return {"processing_errors": [f"RiskScorer error: {str(e)}"]}


# Node 2: Explainer


def explainer_node(state: FraudDetectionState) -> dict:
    """Computes SHAP values for this transaction and converts them into a human readable
    explanation string. Only runs if risk level is med or high.

    """
    try:

        tx_data = _encode_features(state["transaction_data"])
        X = pd.DataFrame([tx_data], columns=FEATURE_COLS)
        X = X.reindex(columns=FEATURE_COLS, fill_value=0)

        explanation = explain_single_prediction(EXPLAINER, X, top_n=5)

        print(f"[Explainer DEBUG] explanation keys: {explanation.keys()}")
        print(f"[Explainer DEBUG] first feature item: {explanation['top_features'][0]}")

        top = explanation["top_features"]
        lines = []
        for feat in top:
            direction = "increased" if feat["shap_value"] > 0 else "decreased"
            lines.append(
                f"- {feat['feature']} (value: {feat['actual_value']})"
                f"{direction} fraud score by {abs(feat['shap_value']): .3f}"
            )

        explanation_text = (
            f"Fraud probability: {state['fraud_probability']:.4f}\n"
            f"Key factors:\n" + "\n".join(lines)
        )

        print(
            f"[Explainer] tx={state['transaction_id']} "
            f"top_feature={top[0]['feature']}"
        )

        return {
            "explanation_text": explanation_text,
            "shap_probability": explanation,
        }

    except Exception as e:
        print(f"[Explainer] ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        return {
            "shap_explanation": None,
            "explanation_text": f"Explanation unavailable: {str(e)}",
            "processing_errors": [f"Explainer error: {str(e)}"],
        }


# Node3: Policy


def policy_node(state: FraudDetectionState) -> dict:
    """Applies business rules on top of the model score.
    This is deliberately rule based , not ML policy decisions
    need to be auditable and adjustable without retraining.

    Rules:
    high risk -> deny+ flag for human review
    medium risk -> flag for human review , dont auto deny
    low risk -> approve automatically
    """

    risk_level = state["risk_level"]
    fraud_prob = state["fraud_probability"]

    if risk_level == "high":
        decision = "deny"
        requires_human = True
        reasoning = (
            f"Transaction denied. Fraud probability {fraud_prob:.4f}"
            f"exceeds threshold  {THRESHOLD:.4f}. Flagged for human review."
        )

    elif risk_level == "medium":
        decision = "flag"
        requires_human = True
        reasoning = (
            f"Transaction flagged for review. Fraud probability "
            f"{fraud_prob:.4f} is in borderline range. "
            f"Human review required before processing."
        )

    else:
        decision = "approve"
        requires_human = False
        reasoning = (
            f"Transaction approved. Fraud probability {fraud_prob:.4f} "
            f"is below threshold. No action required."
        )

    print(
        f"[Policy] tx={state['transaction_id']} "
        f"decision={decision} human={requires_human}"
    )

    return {
        "decision": decision,
        "requires_human": requires_human,
        "policy_reasoning": reasoning,
    }


# Node 4a : Human review


def human_review_node(state: FraudDetectionState) -> dict:
    """
    In a real system this would pause the graph and wait for a human
    analyst to log a decision. Here we log the case for review and add a note
    to the report. This node existing is what makes the graph genuinely agentic--
    a conditional branch.
    """

    print(f"[HumanReview] tx = {state['transaction_id']}" f"queued for analyst review")

    return {
        "policy_reasoning": state["policy_reasoning"] + "\n[QUEUED FOR HUMAN REVIEW]"
    }


# Node 4b: Auto apporve


def auto_approve_node(state: FraudDetectionState) -> dict:
    """
    Low risk transactions skip human review entirely.
    Logs the auto approval decision.
    """

    print(f"[Auto Approve] tx = {state['transaction_id']}" f"auto-approved")

    return {"policy_reasoning": state["policy_reasoning"] + "\n[AUTO_APPROVED]"}


# Node 5: Report


def report_node(state: FraudDetectionState) -> dict:
    """
    Assembles the complete audit trail for this transaction. Every field from every agent ends
    up here. This is what gets returned to the API caller and logged to the monitoring dashboard.
    """

    print(
        f"[REPORT] tx = {state['transaction_id']}"
        f"decision= {state.get('decision')} complete"
    )

    shap_features = []

    if state.get("shap_explanation"):
        shap_features = state["shap_explanation"].get("top_features", [])

    report = {
        "transaction_id": state["transaction_id"],
        "timestamp": datetime.now().isoformat(),
        "fraud_probability": state["fraud_probability"],
        "risk_level": state["risk_level"],
        "decision": state["decision"],
        "requires_human": state["requires_human"],
        "policy_reasoning": state["policy_reasoning"],
        "explanation": state["explanation_text"],
        "shap_top_features": shap_features,
        "errors": state.get("processing_errors", []),
    }

    print(
        f"[REPORT] tx={state['transaction_id']}"
        f"decision={state['decision']} complete"
    )

    return {"final_report": report}
