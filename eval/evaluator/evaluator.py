import pandas as pd
import logging
from abc import ABC, abstractmethod

try:
    from parsers import Parser
    from evaluator import EvaluationFunction
except ImportError:
    from ..parsers import Parser
    from .enum import EvaluationFunction

logger = logging.getLogger(__name__)


class Evaluator(ABC):
    evaluation_function: EvaluationFunction

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def calculate(self, parsers: list[Parser]) -> pd.DataFrame:
        pass

    @classmethod
    def aggregate(cls, key_to_df: dict[str, pd.DataFrame]) -> pd.DataFrame:
        results = []
        for key, df in key_to_df.items():
            if df.empty:
                results.append(
                    {
                        "baseline_name": key,
                        cls.evaluation_function.value: 0.0,
                    }
                )
                continue
            df[cls.evaluation_function.value] = pd.to_numeric(
                df[cls.evaluation_function.value], errors="coerce"
            )
            results.append(
                {
                    "baseline_name": key,
                    cls.evaluation_function.value: round(
                        df[cls.evaluation_function.value].fillna(0).mean(), 2
                    ),
                }
            )
        return pd.DataFrame(results)
