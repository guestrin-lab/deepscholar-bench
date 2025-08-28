from collections import OrderedDict
import re
import csv

try:
    from parsers.parser import Parser, ParserType
except ImportError:
    from .parser import Parser, ParserType


class DeepScholarBaseParser(Parser):
    parser_type = ParserType.DEEPSCHOLAR_BASE

    def _get_file_path(self):
        self.folder_path + "/intro.md"

    @property
    def citation_pattern(self):
        return re.compile(r"\[([^\]]+?)\]\((https?://[^\)]+)\)")

    def _load_file(self):
        self.file_path = self._get_file_path()
        self.raw_generated_text = open(self.file_path, "r", encoding="utf-8").read()
        reference_map = self._reference_parsing(self.folder_path + "/paper.csv")
        self.clean_text, self.docs = self._to_autoais(
            self.raw_generated_text, reference_map
        )

    def _to_autoais(self, updated_text: str, reference_map: dict):
        link2id: OrderedDict[str, int] = OrderedDict()
        docs = []

        def extract_arxiv_id(url: str) -> str | None:
            """Extract arXiv ID from URL."""
            match = re.search(r"arxiv\.org/abs/([^)\s]+)", url)
            return match.group(1) if match else None

        def repl(match: re.Match) -> str:
            visible, url = match.groups()
            arxiv_id = extract_arxiv_id(url)
            if arxiv_id is None:
                return visible

            if arxiv_id not in link2id:
                link2id[arxiv_id] = len(link2id) + 1
                info = reference_map.get(arxiv_id, {})
                docs.append(
                    {"title": info.get("title", ""), "sent": info.get("abstract", "")}
                )

            return f"[{link2id[arxiv_id]}]"

        clean_text = self.citation_pattern.sub(repl, updated_text).strip()
        return clean_text, docs

    def _reference_parsing(self, file_path):
        result = {}
        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                paper_id = row["id"].strip()
                title = row["title"].strip()
                abstract = row["snippet"].strip()
                result[paper_id] = {"title": title, "abstract": abstract}
        return result
