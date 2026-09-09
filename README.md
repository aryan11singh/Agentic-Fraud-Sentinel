# Agentic-Fraud-Sentinel

A production-grade fraud detection system combining classical Machine Learning with a multi-agent orchestration pipeline, SHAP explainability, and a real-time monitoring dashboard. 

This repository demonstrates an end-to-end FinTech machine learning lifecycle—from handling massively imbalanced tabular data to deploying an inference API backed by LLM-powered explanatory agents.

---

## Architecture Overview

The platform processes the standard **IEEE-CIS Dataset** (590k transactions) through a multi-stage pipeline:

### 1. Data & Feature Engineering Layer
- **Temporal Splitting:** Time-based train/test split (80/20 by `TransactionDT`) to simulate real-world production deployment.
- **Feature Engineering:** Implemented rolling aggregates, frequency encoding, missingness flags, and time-based features.
- **Class Balancing:** Utilized SMOTE (Synthetic Minority Over-sampling Technique) to address extreme class imbalance (3.5% → 10% fraud).

### 2. Classical ML Core
- **Classifier:** XGBoost optimized for tabular data inference.
- **Hyperparameter Tuning:** Optuna framework integrated for automated optimization (50 trials, 9 parameters).
- **Threshold Optimization:** Precision-recall curve analysis for optimal operational decision boundaries.

### 3. Explainability Layer
- **SHAP Integration:** TreeExplainer deployed for per-prediction feature attribution to satisfy regulatory explainability requirements.

### 4. Agentic AI Pipeline (LangGraph)
- **Risk Scorer Agent:** Computes fraud probability + risk tier.
- **Explainer Agent:** Translates SHAP values into human-readable rationale.
- **Policy Agent:** Executes business logic (Approve / Flag / Deny).
- **Report Agent:** Generates a complete audit trail for compliance.

### 5. Deployment Services
- **Backend:** FastAPI REST service (`POST /predict`, `GET /health`, `GET /metrics`).
- **Frontend:** Streamlit monitoring dashboard.
- **Infrastructure:** Fully containerized with Docker.

---

## Model Performance Metrics

| Optimization Stage | AUC-ROC | AUC-PR | Precision | Recall | F1 Score |
|:---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost Baseline** | 0.9012 | 0.5246 | 0.8007 | 0.3253 | 0.4626 |
| **Optuna Tuned** | 0.9070 | 0.5758 | 0.8335 | 0.3942 | 0.5352 |
| **Optimal Threshold** (0.216) | 0.9070 | 0.5758 | 0.6990 | 0.4845 | 0.5723 |

**Key Exploratory Data Analysis (EDA) Insights:**
- Fraud rate is strictly non-stationary, ranging from 2%–4.8% over a 185-day window.
- Handled 214 high-cardinality features with >50% missing values via strict missingness flags.
- `TransactionAmt` alone is an extremely weak signal; 4 out of 6 engineered behavioral features dominate the SHAP top 20 impact list.

---

## Technology Stack

| Component | Technologies |
|:---|:---|
| **Data Processing** | `pandas`, `numpy`, `scikit-learn` |
| **Imbalanced Learning** | `imbalanced-learn` (SMOTE) |
| **Machine Learning** | `XGBoost`, `LightGBM` |
| **Hyperparameter Tuning** | `Optuna` |
| **Explainability** | `SHAP` |
| **Agent Orchestration** | `LangGraph` |
| **API & Serving** | `FastAPI`, `uvicorn` |
| **Frontend & Visualization** | `Streamlit`, `Plotly` |
| **Containerization** | `Docker` |

---

## Project Structure

```text
fraud-detection-platform/
├── data/
│   ├── raw/                  # IEEE-CIS source CSVs
│   └── processed/            # Engineered, split, balanced data
├── notebooks/
│   ├── eda.ipynb             # Exploratory Data Analysis
│   ├── preprocessing.ipynb   # Feature engineering pipelines
│   ├── model.ipynb           # XGBoost & Optuna training
│   ├── shap.ipynb            # SHAP value extraction
│   └── agents.ipynb          # LangGraph agent testing
├── src/
│   ├── data/                 # Data pipelines
│   ├── models/               # Training & evaluation scripts
│   ├── explainability/       # SHAP integration
│   └── agents/               # LangGraph node configurations
├── api/
│   └── main.py               # FastAPI application
├── dashboard/
│   └── app.py                # Streamlit monitoring UI
├── docker/
│   └── Dockerfile            # Container configuration
└── requirements.txt          # Python dependencies
```

---

## Setup & Execution

### 1. Environment Initialization
```bash
git clone https://github.com/Shashank17singh/Agentic-Fraud-Sentinel.git
cd Agentic-Fraud-Sentinel
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### 2. Dataset Ingestion
Download the IEEE-CIS Fraud Detection dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection). Place the raw files in the `data/raw/` directory:
- `data/raw/train_transaction.csv`
- `data/raw/train_identity.csv`

### 3. Deployment
*(Update this section with your deployment instructions or Cloud Provider details once live).*
- **API URL:** https://agentic-fraud-sentinel.onrender.com/docs
- **Dashboard URL:** [Your Streamlit Cloud URL]