from typing import List, Dict
from dataclasses import dataclass
from statistics import mean


@dataclass
class NuggetMetrics:
    qid: str
    strict_vital_score: float
    strict_all_score: float
    vital_score: float
    all_score: float


def calculate_nugget_scores(qid: str, nuggets: List[Dict]) -> NuggetMetrics:
    """Calculate various nugget scores for a single response."""
    vital_nuggets = [n for n in nuggets if n["importance"] == "vital"]
    all_nuggets = nuggets

    # Strict scores (only count full support)
    strict_vital_supported = sum(
        1 for n in vital_nuggets if n["assignment"] == "support"
    )
    strict_all_supported = sum(1 for n in all_nuggets if n["assignment"] == "support")

    # Scores with partial support (0.5 for partial_support)
    vital_supported = strict_vital_supported + sum(
        0.5 for n in vital_nuggets if n["assignment"] == "partial_support"
    )
    all_supported = strict_all_supported + sum(
        0.5 for n in all_nuggets if n["assignment"] == "partial_support"
    )

    # Calculate final scores
    strict_vital_score = (
        strict_vital_supported / len(vital_nuggets) if vital_nuggets else 0.0
    )
    strict_all_score = strict_all_supported / len(all_nuggets) if all_nuggets else 0.0
    vital_score = vital_supported / len(vital_nuggets) if vital_nuggets else 0.0
    all_score = all_supported / len(all_nuggets) if all_nuggets else 0.0

    return NuggetMetrics(
        qid=qid,
        strict_vital_score=strict_vital_score,
        strict_all_score=strict_all_score,
        vital_score=vital_score,
        all_score=all_score,
    )


def calculate_global_metrics(records: List[Dict]) -> Dict[str, float | str]:
    """Calculate global mean metrics across all responses."""
    metrics_list = [
        calculate_nugget_scores(record["qid"], record["nuggets"]) for record in records
    ]

    return {
        "qid": "all",
        "strict_vital_score": mean(m.strict_vital_score for m in metrics_list),
        "strict_all_score": mean(m.strict_all_score for m in metrics_list),
        "vital_score": mean(m.vital_score for m in metrics_list),
        "all_score": mean(m.all_score for m in metrics_list),
    }
