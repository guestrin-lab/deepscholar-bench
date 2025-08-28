import json
import logging
import os
from typing import Dict, List
import pandas as pd
import lotus

from nuggetizer.core.types import ScoredNugget
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.core.metrics import calculate_nugget_scores

try:
    from evaluator import Evaluator
    from parsers import Parser
    from evaluator import EvaluationFunction
except ImportError:
    from .evaluator import Evaluator
    from ..parsers import Parser
    from .enum import EvaluationFunction

logger = logging.getLogger(__name__)


class NuggetCoverageEvaluator(Evaluator):
    evaluation_function = EvaluationFunction.NUGGET_COVERAGE

    def __init__(
        self, nugget_groundtruth_dir_path: str = "gt_nuggets_outputs", **kwargs
    ):
        """
        Initialize the NuggetCoverageEvaluator.

        Args:
            gt_dir: Path to ground truth nuggets directory
            **kwargs: Additional arguments (for compatibility with other evaluators)
        """
        super().__init__()
        self.gt_dir = nugget_groundtruth_dir_path
        self.nuggetizer = Nuggetizer(model=lotus.settings.lm.model, log_level=0)

    def _load_ground_truth_nuggets(
        self, file_id: str
    ) -> tuple[List[ScoredNugget], str]:
        assert file_id is not None, "file_id is required"
        gt_path = os.path.join(self.gt_dir, file_id, "res.json")
        if not os.path.exists(gt_path):
            logger.warning(f"Ground truth file not found: {gt_path}")
            return [], ""

        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            nuggets = gt_data.get("supported_nuggets", [])
            query = gt_data.get("query", "")

            if not nuggets:
                logger.warning(f"No nuggets found in {gt_path}")
                return [], query

            # Convert nuggets to ScoredNugget objects
            nuggets_list = [
                ScoredNugget(text=n["text"], importance=n.get("importance", "vital"))
                for n in nuggets
            ]

            return nuggets_list, query

        except Exception as e:
            logger.error(f"Failed to load ground truth for {file_id}: {e}")
            return [], ""

    def _calculate(self, parser: Parser) -> Dict:
        """Calculate nugget coverage metrics for a single parser."""
        nuggets_list, query = self._load_ground_truth_nuggets(str(parser.file_id))
        generated_text = parser.clean_text or ""

        if not nuggets_list or not generated_text:
            return {
                "strict_vital_score": 0.0,
                "strict_all_score": 0.0,
                "vital_score": 0.0,
                "all_score": 0.0,
            }

        try:
            assigned_nuggets = self.nuggetizer.assign(
                query, generated_text, nuggets_list
            )
            nugget_list = [
                {"text": n.text, "importance": n.importance, "assignment": n.assignment}
                for n in assigned_nuggets
            ]

            metrics = calculate_nugget_scores(parser.file_id, nugget_list)

            return {
                "strict_vital_score": metrics.strict_vital_score,
                "strict_all_score": metrics.strict_all_score,
                "vital_score": metrics.vital_score,
                "all_score": metrics.all_score,
            }

        except Exception as e:
            logger.error(f"Failed to process nuggets for {parser.file_id}: {e}")
            return {
                "strict_vital_score": 0.0,
                "strict_all_score": 0.0,
                "vital_score": 0.0,
                "all_score": 0.0,
            }

    def calculate(self, parsers: list[Parser]) -> pd.DataFrame:
        """Calculate nugget coverage metrics for all parsers."""
        return pd.DataFrame(
            [
                {
                    "folder_path": parser.folder_path,
                    self.evaluation_function.value: self._calculate(parser).get(
                        "strict_all_score", 0.0
                    ),
                }
                for parser in parsers
            ]
        )
