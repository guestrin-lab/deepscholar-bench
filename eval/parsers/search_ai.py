import os
import json
from collections import OrderedDict
import re

try:
    from parsers.parser import Parser, ParserType
    from parse_generated_text import (
        extract_arxiv_ids_from_text,
        get_arxiv_title_and_abstract,
        process_inline_citations,
        parse_arxiv_references_from_markdown_references_section,
        process_inline_arxiv_titles,
    )
except ImportError:
    from .parser import Parser, ParserType
    from ..parse_generated_text import (
        extract_arxiv_ids_from_text,
        get_arxiv_title_and_abstract,
        process_inline_citations,
        parse_arxiv_references_from_markdown_references_section,
        process_inline_arxiv_titles,
    )


class SearchAIParser(Parser):
    parser_type = ParserType.SEARCH_AI

    @property
    def citation_pattern(self):
        return re.compile(r"\[([0-9,\s]+)\]")

    def _load_file(self):
        md_files = [f for f in os.listdir(self.folder_path) if f.endswith(".md")]
        json_files = [f for f in os.listdir(self.folder_path) if f.endswith(".json")]

        md_file_path = os.path.join(self.folder_path, md_files[0])
        self.raw_generated_text = open(md_file_path, "r", encoding="utf-8").read()

        if self.use_local_reference_map:
            related_work_section = self.raw_generated_text.split("References")[0]
            file_path = os.path.join(self.folder_path, json_files[0])
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            ctxs = data.get("ctxs", [])
            reference_map = {
                str(i): {
                    "url": f"{ctx.get('url', ctx.get('id'))}",
                    "title": ctx.get("title", ""),
                    "text": ctx.get("text", ctx.get("text", "")),
                }
                for i, ctx in enumerate(ctxs)
            }
            self.clean_text, self.docs = self._to_autoais(
                related_work_section, reference_map
            )
        else:
            all_arxiv_ids, _, self.clean_text = extract_arxiv_ids_from_text(
                self.raw_generated_text
            )
            self.docs = []
            for arxiv_id in all_arxiv_ids:
                title, abstract = get_arxiv_title_and_abstract(arxiv_id)
                if title:
                    self.docs.append({"title": title, "sent": abstract})
                else:
                    self.docs.append({"title": f"ArXiv {arxiv_id}", "sent": ""})

        self.citations_for_cite_quality = (
            parse_arxiv_references_from_markdown_references_section(
                self.raw_generated_text
            )
            + process_inline_citations(self.raw_generated_text)
        )
        if not self.citations_for_cite_quality:
            reference_section = self.raw_generated_text.split("References")[1]
            self.citations_for_cite_quality = process_inline_arxiv_titles(
                reference_section
            )

    def _to_autoais(self, updated_text: str, reference_map: dict):
        link2id: OrderedDict[str, int] = OrderedDict()
        docs = []

        def repl(match: re.Match) -> str:
            original_numbers = match.group(1)
            new_numbers = []

            for num in re.split(r",\s*", original_numbers):
                if num not in link2id:
                    link2id[num] = len(link2id) + 1
                    info = reference_map.get(num, {})
                    docs.append(
                        {"title": info.get("title", ""), "sent": info.get("text", "")}
                    )
                new_numbers.append(str(link2id[num]))

            return f"[{','.join(new_numbers)}]"

        clean_text = self.citation_pattern.sub(repl, updated_text).strip()
        return clean_text, docs
