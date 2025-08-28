import pandas as pd
import re
from typing import Optional
from difflib import SequenceMatcher

try:
    from evaluator import Evaluator
    from parsers import Parser
    from evaluator import EvaluationFunction
except ImportError:
    from .evaluator import Evaluator
    from ..parsers import Parser
    from .enum import EvaluationFunction


class ReferenceCoverageEvaluator(Evaluator):
    evaluation_function = EvaluationFunction.REFERENCE_COVERAGE

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
        if important_citations is None and important_citations_path is None:
            raise ValueError(
                "Either important_citations or important_citations_path must be provided"
            )
        if important_citations is not None:
            self.important_citations = important_citations
        elif important_citations_path is not None:
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
        print(self.important_citations)

    def _calculate(self, parser: Parser) -> float:
        paper_important_citations = self.important_citations.get(
            parser.s_map_groundtruth["arxiv_id"], []
        )
        covered_important = 0

        for important_cite in paper_important_citations:
            print(important_cite)
            important_title = str(important_cite["title"]).strip() or ""
            important_arxiv = str(important_cite["arxiv_link"]).strip() or ""
            match = re.search(r"(\d+\.\d+)(?:v\d+)?", important_arxiv)
            important_arxiv_id = match.group(1) if match else None
            print(important_arxiv_id)
            for citations in parser.docs:
                found_title = str(citations["title"]) or ""
                found_abstract = str(citations["sent"]) or ""

                if important_arxiv_id and (
                    important_arxiv_id in found_title
                    or important_arxiv_id in found_abstract
                ):
                    covered_important += 1
                    break

                similarity = calculate_title_similarity(important_title, found_title)
                print(found_title, similarity)
                if similarity > 0.8:
                    covered_important += 1
                    break

        return (
            covered_important / len(paper_important_citations)
            if len(paper_important_citations) > 0
            else 0
        )

    def calculate(self, parsers: list[Parser]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "folder_path": [parser.folder_path for parser in parsers],
                self.evaluation_function.value: [
                    self._calculate(parser) for parser in parsers
                ],
            }
        )


def calculate_title_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles"""
    if not title1 or not title2:
        return 0.0

    norm1 = normalize_title(title1)
    norm2 = normalize_title(title2)

    if not norm1 or not norm2:
        return 0.0

    return SequenceMatcher(None, norm1, norm2).ratio()


def normalize_title(title: str) -> str:
    """Normalize title for comparison by removing special characters and converting to lowercase"""
    if not title:
        return ""
    # Remove special characters and extra whitespace
    normalized = re.sub(r"[^\w\s]", "", title.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    common_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
    }
    words = [
        word
        for word in normalized.split()
        if word not in common_words and len(word) > 2
    ]
    return " ".join(words)
