from typing import List, Optional, TypedDict


class FraudDetectionState(TypedDict):
    """
    Shared state passed between every node in the graph. Every agent reads from
    this and writes back to it. TypedDict gives us type safety without a full dataclass.
    """

    transaction_id: str
    transaction_data: dict

    fraud_probability: Optional[float]
    risk_level: Optional[str]

    shap_probability: Optional[dict]
    explanation_text: Optional[str]

    decision: Optional[str]
    policy_reasoning: Optional[str]

    requires_human: Optional[bool]

    final_report: Optional[dict]

    processing_errors: Optional[List[str]]
