# pyrefly: ignore [missing-import]
import os
import sys
from typing import Any, List, Optional

import joblib
import pandas as pd
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
from numpy.random import logistic
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if os.path.exists(os.path.join(BASE_DIR, "data", "models", "xgboost_production.pkl")):
    DATA_DIR = os.path.join(BASE_DIR, "data")
else:
    DATA_DIR = os.path.join(BASE_DIR, "notebooks", "data")





from src.agents.graph import build_fraud_graph

# App setup

app = FastAPI(
    title="Agentic-Fraud-Sentinel API",
    description="Multi agent fraud detection - XGBoost and LangGraph",
    version="1.0.0",
)

# ── Load graph once at startup ───────────────────────────────────
# We build the graph when the server starts, not on every request
# Building it on every request would reload model files from disk
# each time — extremely slow under load

print("Building fraud detection graph..")
fraud_graph = build_fraud_graph()
print("Graph ready...")

# Load feature columns so we know what the model expects
bundle = joblib.load(os.path.join(DATA_DIR, "models", "xgboost_production.pkl"))
FEATURE_COLS = bundle["model"].get_booster().feature_names


# REQUEST MODEL


class TransactionRequest(BaseModel):
    transaction_id: str
    features: dict


# Response Model


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    decision: str
    requires_human: bool
    explanation: str
    policy_reasoning: str
    shap_top_features: List[Any]  # simplified — no nested Dict typing
    errors: List[Any]

    model_config = {"arbitrary_types_allowed": True}


# Endpoints


@app.get("/health")
def health_check():
    """
    Simplest possible endpoint
    Load balancers and monitoring tools ping this to check
    whether the server is alive. Returns 200 if ok
    """

    return {"status": "ok", "model": "xgboost_tuned", "version": "1.0.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    """
    Main endpoints - runs the full graph pipeline.

    Flow :-
    1. Receives transaction features as JSON
    2. Build initial state for the graph
    3. Run graph: RiskScorer-> explainer-> Policy -> [HumanReview|AutoApprove] -> Report
    4. Return structured decision with explanation
    """

    try:
        # Align incoming features to models expected column order
        # Missing features get filled with 0.

        feature_row = {col: request.features.get(col, 0) for col in FEATURE_COLS}

        initial_state = {
            "transaction_id": request.transaction_id,
            "transaction_data": feature_row,
            "fraud_probability": None,
            "risk_level": None,
            "shap_explanation": None,
            "explanation_text": None,
            "decision": None,
            "policy_reasoning": None,
            "requires_human": None,
            "final_report": None,
            "processing_errors": [],
        }

        # Run the graph - this is the single line that runs
        # all 5 agents in sequence with the conditional routing

        result = fraud_graph.invoke(initial_state)
        report = result["final_report"]

        if report is None:
            raise HTTPException(status_code=500, detail="graph produced no report")

        return PredictionResponse(
            transaction_id=report["transaction_id"],
            fraud_probability=report["fraud_probability"],
            risk_level=report["risk_level"],
            decision=report["decision"],
            requires_human=report["requires_human"],
            explanation=report["explanation"] or "",
            policy_reasoning=report["policy_reasoning"] or "",
            shap_top_features=report["shap_top_features"],
            errors=report["errors"],
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(status_code=500, detail=f"Predcition failed: {str(e)}")


@app.get("/")
def read_root():
    return {"message": "Welcome to Agentic Fraud Sentinel API. Visit /docs for documentation."}


@app.get("/metrics")
def get_metrics():
    """
    Basic model metadata- feeds the streamlit dashboard.

    """

    return {
        "model": "xgboost_tuned",
        "threshold": 0.2161,
        "auc_roc": 0.9070,
        "auc_pr": 0.5758,
        "precision": 0.8335,
        "recall": 0.3942,
    }
