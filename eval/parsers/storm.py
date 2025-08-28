import os
import json
import re

try:
    from parsers.parser import Parser, ParserType
    from parse_generated_text import extract_arxiv_ids_from_text
    from utils import get_arxiv_title_and_abstract
except ImportError:
    from ..parsers.parser import Parser, ParserType
    from ..parse_generated_text import extract_arxiv_ids_from_text
    from ..utils import get_arxiv_title_and_abstract


class StormParser(Parser):
    parser_type = ParserType.STORM

    @property
    def citation_pattern(self):
        return re.compile(r"\[([^\]]+?)\]\((https?://[^\)]+)\)")

    def _load_file(self):
        self.file_path = os.path.join(self.folder_path, "storm_gen_article.md")
        if not os.path.isfile(self.file_path):
            folder_name = os.listdir(self.folder_path)[0]
            self.folder_path = os.path.join(self.folder_path, folder_name)
            self.file_path = os.path.join(self.folder_path, "storm_gen_article.md")
        if not os.path.isfile(self.file_path):
            return
        self.raw_generated_text = open(self.file_path, "r").read()
        if self.use_local_reference_map:
            url_to_info_path = os.path.join(self.folder_path, "url_to_info.json")
            self.clean_text, self.docs = self._to_autoais(
                self.raw_generated_text, url_to_info_path
            )
        else:
            all_arxiv_ids, _, self.clean_text = extract_arxiv_ids_from_text(
                self.raw_generated_text
            )
            docs = []
            for arxiv_id in all_arxiv_ids:
                try:
                    # Remove version number for API call
                    clean_arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
                    title, abstract = get_arxiv_title_and_abstract(clean_arxiv_id)
                    if title:
                        docs.append({"title": title, "sent": abstract or ""})
                    else:
                        docs.append({"title": f"ArXiv {arxiv_id}", "sent": ""})
                except Exception as e:
                    print(f"Error fetching arXiv data for {arxiv_id}: {e}")
                    docs.append({"title": f"ArXiv {arxiv_id}", "sent": ""})
            self.docs = docs

    def _to_autoais(self, updated_text: str, url_to_info_path: str):
        return updated_text, self.extract_title_snippet_list_storm(url_to_info_path)

    def extract_title_snippet_list_storm(self, file_path):
        """
        Reads a JSON file and returns a list of dictionaries with 'title' and 'sent' (first snippet).
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        url_to_info = data.get("url_to_info", {})
        docs = []

        for url, info in url_to_info.items():
            title = info.get("title", "")
            snippets = info.get("snippets", [])
            if snippets:
                sent = snippets[0]
                docs.append({"title": title, "sent": sent})

        return docs
