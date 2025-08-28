import os
import json
from collections import OrderedDict
import re
import pandas as pd

try:
    from parser import Parser, ParserType
except ImportError:
    from ..parsers.parser import Parser, ParserType


class GroundTruthParser(Parser):
    parser_type = ParserType.GROUNDTRUTH

    @property
    def citation_pattern(self):
        return re.compile(r"\[([0-9,\s]+)\]")

    def _load_file(self):
        self.clean_text, self.docs = self._to_autoais(
            self.s_map_groundtruth["related_works_section"]
        )
        self.raw_generated_text = self.s_map_groundtruth["related_works_section"]

    def _to_autoais(self, updated_text: str):
        citation_path = self.config["citation_path"]
        citations_df = pd.read_csv(citation_path)
        grouped = citations_df.groupby("parent_paper_arxiv_id")
        row = grouped.get_group(self.s_map_groundtruth["arxiv_id"])

        # TODO: confirm this
        return updated_text, [
            {"title": row["cited_paper_title"], "sent": row["search_res_content"]}
        ]
