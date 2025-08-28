import pandas as pd
import re
import time
import numpy as np

try:
    from evaluator import Evaluator
    from parsers import Parser
    from evaluator import EvaluationFunction
    from utils import get_citation_count_from_title
except ImportError:
    from .evaluator import Evaluator
    from ..parsers import Parser
    from .enum import EvaluationFunction
    from ..utils import get_citation_count_from_title


class DocumentImportanceEvaluator(Evaluator):
    evaluation_function = EvaluationFunction.DOCUMENT_IMPORTANCE

    def _calculate(self, parser: Parser) -> float:
        citations = [
            re.sub(r"[^\w\s]", "", doc["title"].lower()).strip() for doc in parser.docs
        ]
        citations = list(set([citation for citation in citations if citation]))
        results = []
        for i, citation in enumerate(citations):
            time.sleep(0.05)  # Rate limiting
            count = get_citation_count_from_title(citation)
            if count is not None:
                results.append(count)
        return np.median(results) if results else 0

    def calculate(self, parsers: list[Parser]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "folder_path": [parser.folder_path for parser in parsers],
                self.evaluation_function.value: [
                    self._calculate(parser) for parser in parsers
                ],
            }
        )
