from enum import Enum
import pandas as pd
import os

try:
    from parsers import Parser
except ImportError:
    from ..parsers import Parser


class EvaluationFunction(Enum):
    ORGANIZATION = "organization"
    NUGGET_COVERAGE = "nugget_coverage"
    REFERENCE_COVERAGE = "reference_coverage"
    DOCUMENT_IMPORTANCE = "document_importance"
    CITE_P = "cite_p"
    CLAIM_COVERAGE = "claim_coverage"
    COVERAGE_RELEVANCE_RATE = "coverage_relevance_rate"

    def to_evaluator(self):
        if self == EvaluationFunction.ORGANIZATION:
            from .organization import OrganizationEvaluator

            return OrganizationEvaluator
        elif self == EvaluationFunction.NUGGET_COVERAGE:
            from .nugget_coverage import NuggetCoverageEvaluator

            return NuggetCoverageEvaluator
        elif self == EvaluationFunction.REFERENCE_COVERAGE:
            from .reference_coverage import ReferenceCoverageEvaluator

            return ReferenceCoverageEvaluator
        elif self == EvaluationFunction.DOCUMENT_IMPORTANCE:
            from .document_importance import DocumentImportanceEvaluator

            return DocumentImportanceEvaluator
        elif self == EvaluationFunction.CITE_P:
            from .cite_p import CitePEvaluator

            return CitePEvaluator
        elif self == EvaluationFunction.CLAIM_COVERAGE:
            from .claim_coverage import ClaimCoverageEvaluator

            return ClaimCoverageEvaluator
        elif self == EvaluationFunction.COVERAGE_RELEVANCE_RATE:
            from .coverage_relevance_rate import CoverageRelevanceRateEvaluator

            return CoverageRelevanceRateEvaluator
        else:
            raise ValueError(f"Invalid evaluation function: {self}")

    def calculate(
        self,
        key_to_parsers: dict[str, list[Parser]],
        output_dir: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        evaluator = self.to_evaluator()(**kwargs)
        results = {
            key: evaluator.calculate(parsers) if parsers else pd.DataFrame()
            for key, parsers in key_to_parsers.items()
        }
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for key, dfs in results.items():
                dfs.to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)
        return results

    def evaluate(self, key: str, parser: list[Parser], **kwargs) -> pd.DataFrame:
        return self.evaluate_all({key: parser}, **kwargs)

    def evaluate_all(
        self,
        key_to_parsers: dict[str, list[Parser]],
        output_dir: str | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        results = self.calculate(key_to_parsers, output_dir, **kwargs)
        aggregated_results = self.to_evaluator().aggregate(results)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            aggregated_results.to_csv(
                os.path.join(output_dir, "aggregated_results.csv"), index=False
            )
        return aggregated_results
