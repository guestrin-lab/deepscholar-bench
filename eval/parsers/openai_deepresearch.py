import os
from collections import OrderedDict
import re

try:
    from parsers.parser import Parser, ParserType
    from parse_generated_text import (
        extract_arxiv_ids_from_text,
        get_arxiv_title_and_abstract,
        replace_refs,
    )
    from utils import extract_html_content
except ImportError:
    from ..parsers.parser import Parser, ParserType
    from ..parse_generated_text import (
        extract_arxiv_ids_from_text,
        get_arxiv_title_and_abstract,
        replace_refs,
    )
    from ..utils import extract_html_content


class OpenAIDeepResearchParser(Parser):
    parser_type = ParserType.OPENAI_DEEPRESEARCH

    @property
    def citation_pattern(self):
        return re.compile(r"\[([^\]]+?)\]\((https?://[^\)]+)\)")

    def _load_file(self):
        md_files = [f for f in os.listdir(self.folder_path) if f.endswith(".md")]

        md_file_path = os.path.join(self.folder_path, md_files[0])
        self.raw_generated_text = open(md_file_path, "r", encoding="utf-8").read()

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

        _, self.citations_for_cite_quality = self._process_for_cite_quality()

    def _process_for_cite_quality(self) -> tuple[str, list[dict]]:
        # Load the pickle file
        md_files = [f for f in os.listdir(self.folder_path) if f.endswith(".md")]
        md_file_path = os.path.join(self.folder_path, md_files[0])
        content = open(md_file_path, "r", encoding="utf-8").read()

        related_work_section = (
            content.split("References")[0] if "References" in content else content
        )

        link_pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        links_in_text = re.findall(link_pattern, related_work_section)

        arxiv_id_pattern = r"arXiv:(\d+\.\d+(?:v\d+)?)\s*\((\d{4})\)"
        arxiv_ids_in_text = re.findall(arxiv_id_pattern, content)

        all_citations = [(link_text, link_url) for link_text, link_url in links_in_text]

        for arxiv_id, _ in arxiv_ids_in_text:
            arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
            if arxiv_url not in all_citations:
                try:
                    fetched_title, _ = get_arxiv_title_and_abstract(arxiv_id)
                    title = fetched_title if fetched_title else f"arXiv:{arxiv_id}"
                except Exception as _:
                    title = f"arXiv:{arxiv_id}"

                all_citations.append((title, arxiv_url))

        processed_urls = set()
        reference_map = {}

        for i, (title, url) in enumerate(all_citations):
            if url in processed_urls:
                continue
            processed_urls.add(url)

            abstract = ""
            arxiv_match = re.search(
                r"https://arxiv\.org/(?:abs|html|pdf)/(\d+\.\d+(?:v\d+)?)", url
            )
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
                try:
                    __, fetched_abstract = get_arxiv_title_and_abstract(arxiv_id)  # type: ignore
                    if fetched_abstract and len(fetched_abstract) > 10:
                        abstract = fetched_abstract
                except Exception as _:
                    pass
            else:
                html_content = extract_html_content(url)
                if html_content and len(html_content) > 50:
                    abstract = html_content

            if title and abstract:
                reference_map[str(i)] = {
                    "url": f"[{i}]({url})",
                    "title": title,
                    "text": abstract,
                }
            else:
                pass

        updated_text = replace_refs(related_work_section, reference_map)
        answer, docs = self.md_to_autoais_openai_deep_research(
            updated_text, reference_map
        )

        if not docs or len(docs) == 0:
            answer, docs = self.md_to_autoais_openai_deep_research2(
                updated_text, reference_map
            )

        return answer, [(doc["title"], doc["sent"]) for doc in docs]  # type: ignore

    def md_to_autoais_openai_deep_research(
        self, markdown: str, url2abs: dict | None = None
    ) -> tuple[str, list[dict]]:
        """
        Convert OpenAI Deep Research format markdown to AutoAIS format.

        Args:
            markdown: Input markdown text with markdown links
            url2abs: Mapping of indices to document info with title and text

        Returns:
            Tuple of (cleaned_text, list_of_docs)
        """
        url2abs = url2abs or {}
        link2id: dict[str, int] = OrderedDict()
        docs = []

        def repl(match: re.Match) -> str:
            visible, url = match.groups()
            # Extract the index from the URL format [index](url)
            # For OpenAI Deep Research, the index is the key in url2abs
            index = None
            for idx, info in url2abs.items():
                if info.get("url", "").endswith(url.split("/")[-1]) or url in info.get(
                    "url", ""
                ):
                    index = idx
                    break

            if index is None:
                # If we can't find the index, try to extract it from the URL
                url_parts = url.split("/")
                if len(url_parts) > 0:
                    # Try to use the last part of the URL as a key
                    potential_key = url_parts[-1].split("#")[0]  # Remove anchor
                    for idx, info in url2abs.items():
                        if potential_key in info.get("url", ""):
                            index = idx
                            break

            if index is None:
                # If still not found, return the original text
                return visible

            if index not in link2id:
                link2id[index] = len(link2id) + 1
                info = url2abs.get(index, {})
                docs.append(
                    {"title": info.get("title", ""), "sent": info.get("text", "")}
                )

            return f"[{link2id[index]}]"

        clean_text = self.citation_pattern.sub(repl, markdown).strip()
        return clean_text, docs

    def md_to_autoais_openai_deep_research2(self, markdown_text, reference_map):
        """
        Custom AutoAIS function for OpenAI Deep Research mode that properly handles citation mapping
        """
        docs = []
        citation_mapping = {}

        for i, (ref_id, ref_info) in enumerate(reference_map.items()):
            docs.append({"title": ref_info["title"], "sent": ref_info["text"]})
            citation_mapping[ref_id] = str(i + 1)

        answer = markdown_text
        for old_ref, new_ref in citation_mapping.items():
            answer = answer.replace(f"[{old_ref}]", f"[{new_ref}]")

        return answer, docs
