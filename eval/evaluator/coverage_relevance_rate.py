import pandas as pd

try:
    from evaluator import Evaluator
    from parsers import Parser
    from evaluator import EvaluationFunction
    from parse_generated_text import extract_arxiv_ids_from_text
    from utils import get_arxiv_title_and_abstract, escape_braces
    from ..prompts.citation_relevance_judge_instruction import (
        citation_relevance_judge_instruction,
        CitationRelevanceResponse,
    )
except ImportError:
    from .evaluator import Evaluator
    from ..parsers import Parser
    from .enum import EvaluationFunction
    from ..parse_generated_text import extract_arxiv_ids_from_text
    from ..utils import get_arxiv_title_and_abstract, escape_braces
    from ..prompts.citation_relevance_judge_instruction import (
        citation_relevance_judge_instruction,
        CitationRelevanceResponse,
    )


class CoverageRelevanceRateEvaluator(Evaluator):
    evaluation_function = EvaluationFunction.COVERAGE_RELEVANCE_RATE

    def _calculate(self, parser: Parser) -> float:
        def enhance_citation(title, sent):
            if not sent or not title or "arxiv" in title.lower():
                _, arxiv_ids, _ = extract_arxiv_ids_from_text(title)
                if arxiv_ids:
                    arxiv_id = arxiv_ids[0]
                    try:
                        title, sent = get_arxiv_title_and_abstract(arxiv_id)
                    except Exception as e:
                        print(f"Error fetching arXiv data for {arxiv_id}: {e}")
                        title, sent = title, sent
            return title, sent

        system_prompt = "You are an intelligent, rigorous, and fair evaluator of scholarly relevance."
        judge_instruction = citation_relevance_judge_instruction.format(
            paper_title=escape_braces(parser.s_map_groundtruth["title"]),
            paper_abstract=escape_braces(parser.s_map_groundtruth["abstract"]),
            paper_related_work=escape_braces(
                parser.s_map_groundtruth["related_works_section"]
            ),
            ref_title="{title}",
            ref_abstract="{sent}",
        )
        df = pd.DataFrame(parser.docs)
        df = df.drop_duplicates(subset=["title"])
        if df.empty:
            return 0
        df["title"], df["sent"] = zip(
            *df.apply(lambda x: enhance_citation(x["title"], x["sent"]), axis=1)
        )
        df = df.dropna(subset=["title", "sent"])
        df = df.llm_as_judge(
            judge_instruction=judge_instruction,
            system_prompt=system_prompt,
            n_trials=2,
            response_format=CitationRelevanceResponse,
        )
        df["graded_rel"] = df["_judge_1"].map(lambda x: x.score)
        df.drop(columns=["_judge_1"], inplace=True)
        return df["graded_rel"].mean()

    def calculate(self, parsers: list[Parser]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "folder_path": [parser.folder_path for parser in parsers],
                self.evaluation_function.value: [
                    self._calculate(parser) for parser in parsers
                ],
            }
        )
