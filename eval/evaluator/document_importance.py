import pandas as pd
import re
import time
import numpy as np
from typing import Optional

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

    def __init__(
        self,
        important_citations: Optional[dict[str, list[dict[str, str]]]] = None,
        important_citations_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        if important_citations is not None and important_citations_path is not None:
            raise ValueError(
                "Only one of important_citations or important_citations_path can be provided"
            )
        if important_citations is not None:
            self.important_citations = important_citations
        elif important_citations_path is not None:
            import os
            if not os.path.exists(important_citations_path):
                # If file doesn't exist, set to empty dict (will use fallback behavior)
                self.important_citations = {}
            else:
                df = pd.read_csv(important_citations_path)
                important_citations = {}
                for _, row in df.iterrows():
                    parent_arxiv_id = str(row["parent_paper_arxiv_id"]).strip()
                    cited_arxiv_link = (
                        str(row["cited_paper_arxiv_link"])
                        if pd.notna(row["cited_paper_arxiv_link"])
                        else ""
                    )

                    # Only include citations that have arXiv links
                    if cited_arxiv_link.strip():
                        if parent_arxiv_id not in important_citations:
                            important_citations[parent_arxiv_id] = []

                        # Handle potential float/NaN values
                        cited_title = (
                            str(row["cited_paper_title"])
                            if pd.notna(row["cited_paper_title"])
                            else ""
                        )
                        cited_abstract = (
                            str(row["cited_paper_abstract"])
                            if pd.notna(row["cited_paper_abstract"])
                            else ""
                        )
                        cited_shorthand = (
                            str(row["citation_shorthand"])
                            if pd.notna(row["citation_shorthand"])
                            else ""
                        )

                        important_citations[parent_arxiv_id].append(
                            {
                                "title": cited_title,
                                "arxiv_link": cited_arxiv_link,
                                "abstract": cited_abstract,
                                "shorthand": cited_shorthand,
                            }
                        )
                self.important_citations = important_citations
        else:
            # If neither provided, set to empty dict (will use fallback behavior)
            self.important_citations = {}

    def _get_groundtruth_median_citations(self, parser: Parser) -> float:
        """Calculate median citation count for ground truth papers."""
        paper_important_citations = self.important_citations.get(
            parser.s_map_groundtruth["arxiv_id"], []
        )
        if not paper_important_citations:
            return None
        
        gt_citation_counts = []
        for gt_cite in paper_important_citations:
            gt_title = str(gt_cite.get("title", "")).strip()
            if not gt_title:
                continue
            time.sleep(0.05)  # Rate limiting
            count = get_citation_count_from_title(gt_title)
            if count is not None:
                gt_citation_counts.append(count)
        
        if not gt_citation_counts:
            return None
        
        return np.median(gt_citation_counts)

    def _calculate(self, parser: Parser) -> float:
        # Get citation counts for generated citations
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
        
        if not results:
            return 0.0
        
        generated_median = np.median(results)
        
        # Get ground truth median and normalize
        gt_median = self._get_groundtruth_median_citations(parser)
        if gt_median is None or gt_median == 0:
            # If no ground truth citations found, return 0 (can't normalize)
            return 0.0
        
        # Normalize by ground truth median and cap at 1
        # This ensures scores are between 0 and 1, comparable to benchmark numbers
        normalized_score = generated_median / gt_median
        return min(normalized_score, 1.0)

    def calculate(self, parsers: list[Parser]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "folder_path": [parser.folder_path for parser in parsers],
                self.evaluation_function.value: [
                    self._calculate(parser) for parser in parsers
                ],
            }
        )
