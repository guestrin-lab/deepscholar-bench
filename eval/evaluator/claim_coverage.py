import pandas as pd
import re
from nltk.tokenize import sent_tokenize
import numpy as np

try:
    from evaluator import Evaluator
    from parsers import Parser
    from evaluator import EvaluationFunction
    from prompts.support import get_support
except ImportError:
    from .evaluator import Evaluator
    from ..parsers import Parser
    from .enum import EvaluationFunction
    from ..prompts.support import get_support


class ClaimCoverageEvaluator(Evaluator):
    evaluation_function = EvaluationFunction.CLAIM_COVERAGE

    def __init__(self, window_size: int = 1, **kwargs):
        super().__init__()
        self.window_size = window_size

    def _calculate(self, parser: Parser, window_size: int) -> pd.DataFrame:
        sentences = custom_sent_tokenize(parser.clean_text)
        supports = []
        for sid, sent in enumerate(sentences):
            stripped_sent = _remove_citations(sent)
            if len(stripped_sent) < 50:
                continue

            all_window_docs = []
            support = False
            for i in range(
                max(0, sid - window_size), min(len(sentences), sid + window_size + 1)
            ):
                refs = [int(x[1:]) - 1 for x in re.findall(r"\[\d+", sent)]
                sentence_docs = []
                for ref in refs:
                    if ref > len(parser.docs) - 1 or ref < 0:
                        continue
                    sentence_docs.append(
                        _format_document(parser.citations_for_cite_quality[ref])
                    )
                all_window_docs.extend(sentence_docs)
                if sentence_docs:
                    sentence_docs_str = (
                        " ".join(sentence_docs)
                        + parser.s_map_groundtruth.get("title", "")
                        + parser.s_map_groundtruth.get("abstract", "")
                    )
                    if get_support(sentence_docs_str, stripped_sent):
                        support = True
                        break
            supports.append(1 if support else 0)

        return np.mean(supports)

    def calculate(
        self, parsers: list[Parser], window_size: int | None = None
    ) -> pd.DataFrame:
        window_size = window_size or self.window_size
        return pd.DataFrame(
            {
                "folder_path": [parser.folder_path for parser in parsers],
                self.evaluation_function.value: [
                    self._calculate(parser, window_size) for parser in parsers
                ],
            }
        )


def custom_sent_tokenize(text):
    protected_text = text
    protected_text = re.sub(r"\bet al\.", "ET_AL_PLACEHOLDER", protected_text)
    sentences = sent_tokenize(protected_text)
    sentences = [s.replace("ET_AL_PLACEHOLDER", "et al.") for s in sentences]
    return sentences


def _remove_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text).replace(" |", "").strip()


def _format_document(doc: tuple[str, str]) -> str:
    return f"Title: {doc[0]}\n{doc[1]}"
