import os
import json
from collections import OrderedDict
import re
import requests  # type: ignore
import xml.etree.ElementTree as ET

try:
    from parsers.parser import Parser, ParserType
    from parse_generated_text import (
        replace_refs,
        replace_latex_cites,
        extract_arxiv_ids_from_text,
        get_arxiv_title_and_abstract,
    )
except ImportError:
    from ..parsers.parser import Parser, ParserType
    from ..parse_generated_text import (
        replace_refs,
        replace_latex_cites,
        extract_arxiv_ids_from_text,
        get_arxiv_title_and_abstract,
    )


class DeepResearcherParser(Parser):
    parser_type = ParserType.DEEPRESEARCHER

    @property
    def citation_pattern(self):
        return re.compile(r"\[([^\]]+?)\]\((https?://[^\)]+)\)")

    def _load_file(self):
        json_files = [f for f in os.listdir(self.folder_path) if f.endswith(".json")]
        if not json_files:
            return
        self.file_path = os.path.join(self.folder_path, json_files[0])
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        raw_text = data.get("output", "")

        if self.use_local_reference_map:
            ctxs = data.get("ctxs", [])
            reference_map = {}
            for i, ctx in enumerate(ctxs):
                reference_url = ""
                year = None
                if "paperId" in ctx:
                    ref_id = ctx["paperId"]
                    title = ctx.get("title", "").strip()
                    text = ctx.get("abstract", "").strip()
                    reference_url = ctx.get("url", "")
                    year = ctx.get("year", "")
                else:
                    ref_id = ctx.get("id")
                    title = ctx.get("title", "").strip()
                    text = ctx.get("text", "").strip()

                reference_map[str(i)] = {
                    "url": f"[{i}]({ref_id})",
                    "title": title,
                    "text": text,
                    "reference_url": reference_url,
                    "year": year,
                }

            updated_text = replace_refs(raw_text, reference_map)
            self.docs = [
                {"title": info["title"], "sent": info["text"]}
                for info in reference_map.values()
            ]
            if not self.docs:
                updated_text = replace_latex_cites(updated_text, reference_map)
                self.clean_text, self.docs = self._to_autoais(updated_text)
            self.raw_generated_text = updated_text.split("References")[0]
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

        _, self.citations_for_cite_quality = self._process_for_cite_quality()

    def _to_autoais(self, updated_text: str, reference_map: dict):
        link2id: OrderedDict[str, int] = OrderedDict()
        docs = []

        def repl(match: re.Match) -> str:
            _, url = match.groups()
            if url not in link2id:
                link2id[url] = len(link2id) + 1
                info = self._get_arxiv_title_and_abstract(url)
                docs.append(
                    {"title": info.get("title", ""), "sent": info.get("abstract", "")}
                )
            return f"[{link2id[url]}]"

        clean_text = self.citation_pattern.sub(repl, updated_text).strip()
        return clean_text, docs

    def _get_arxiv_title_and_abstract(self, arxiv_url: str) -> dict[str, str]:
        """
        Fetch title and abstract from arXiv API.

        Args:
            arxiv_url: arXiv URL or ID

        Returns:
            Dictionary with 'title' and 'abstract' keys, or 'error' if failed
        """
        try:
            arxiv_id = arxiv_url.strip().split("/")[-1]
            api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            entry = root.find("{http://www.w3.org/2005/Atom}entry")

            if entry is None:
                return {"error": "No entry found"}

            title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
            summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")

            title = (
                title_elem.text.strip()
                if title_elem is not None and title_elem.text is not None
                else ""
            )
            abstract = (
                summary_elem.text.strip()
                if summary_elem is not None and summary_elem.text is not None
                else ""
            )

            return {"title": title, "abstract": abstract}
        except Exception as e:
            return {"error": str(e)}

    def _process_for_cite_quality(self) -> tuple[str, list[dict]]:
        md_files = [f for f in os.listdir(self.folder_path) if f.endswith(".md")]

        md_file_path = os.path.join(self.folder_path, md_files[0])
        with open(md_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "<references>" in content:
            related_works_section = content.split("<references>")[0]
        elif "References" in content:
            related_works_section = content.split("References")[0]
        else:
            related_works_section = content

        # Extract the reference section
        reference_section = ""
        if "<references>" in content:
            reference_section = content.split("<references>")[1]
        elif "References" in content:
            reference_section = content.split("References")[1]

        # Extract arXiv IDs from the related work section
        arxiv_pattern = r"\[(\d+\.\d+(?:v\d+)?)\]"
        arxiv_matches = re.findall(arxiv_pattern, related_works_section)

        # Also extract deepresearcher citation pattern: [1] [arxiv_id](http://arxiv.org/abs/arxiv_id)
        deepresearcher_pattern = (
            r"\[(\d+)\]\s*\[(\d+\.\d+(?:v\d+)?)\]\(http://arxiv\.org/abs/\2\)"
        )
        deepresearcher_matches = re.findall(
            deepresearcher_pattern, related_works_section
        )

        # Also search in the reference section for deepresearcher citations
        deepresearcher_matches_ref = re.findall(
            deepresearcher_pattern, reference_section
        )

        # Also try a more flexible pattern for deepresearcher citations
        deepresearcher_pattern2 = (
            r"\[(\d+)\]\s*\[(\d+\.\d+(?:v\d+)?)\]\([^)]*arxiv\.org/abs/\2[^)]*\)"
        )
        deepresearcher_matches2 = re.findall(
            deepresearcher_pattern2, related_works_section
        )
        deepresearcher_matches2_ref = re.findall(
            deepresearcher_pattern2, reference_section
        )

        # Combine matches from both sections, removing duplicates
        all_deepresearcher_matches = (
            deepresearcher_matches
            + deepresearcher_matches_ref
            + deepresearcher_matches2
            + deepresearcher_matches2_ref
        )
        # Remove duplicates based on (ref_num, arxiv_id) pairs
        seen = set()
        deepresearcher_matches = []
        for match in all_deepresearcher_matches:
            if match not in seen:
                seen.add(match)
                deepresearcher_matches.append(match)

        # Extract XML-style references: 1. [2204.06547v2] within <references> tags
        xml_ref_pattern = r"(\d+)\.\s*\[(\d+\.\d+(?:v\d+)?)\]"
        xml_ref_matches = re.findall(xml_ref_pattern, reference_section)

        # Parse the reference section to get numbered references
        ref_pattern = r"(\d+)\.\s*(.+)"
        ref_matches = re.findall(ref_pattern, reference_section)

        # Create reference map and arXiv ID mapping
        reference_map = {}
        arxiv_id_to_ref_idx = {}  # Mapping from arXiv ID to reference index
        docs = []

        # Process XML-style references first (they have explicit numbering)
        for ref_num, arxiv_id in xml_ref_matches:
            arxiv_id_to_ref_idx[arxiv_id] = ref_num

            # Try to get abstract from arXiv
            try:
                arxiv_title, arxiv_abstract = get_arxiv_title_and_abstract(arxiv_id)
                if arxiv_abstract:
                    abstract = arxiv_abstract
                    final_title = arxiv_title
                else:
                    abstract = f"arXiv paper: {arxiv_id}"
                    final_title = f"arXiv:{arxiv_id}"
            except Exception:
                abstract = f"arXiv paper: {arxiv_id}"
                final_title = f"arXiv:{arxiv_id}"

            # Add to reference map
            reference_map[ref_num] = {
                "url": f"[{ref_num}]({final_title})",
                "title": final_title,
                "text": abstract,
            }

            # Add to docs for evaluation
            docs.append({"title": final_title, "sent": abstract})

        # Process deepresearcher citations
        for ref_num, arxiv_id in deepresearcher_matches:
            if arxiv_id not in arxiv_id_to_ref_idx:  # Skip if already processed
                arxiv_id_to_ref_idx[arxiv_id] = ref_num

                # Try to get abstract from arXiv
                try:
                    arxiv_title, arxiv_abstract = get_arxiv_title_and_abstract(arxiv_id)
                    if arxiv_abstract:
                        abstract = arxiv_abstract
                        final_title = arxiv_title
                    else:
                        abstract = f"arXiv paper: {arxiv_id}"
                        final_title = f"arXiv:{arxiv_id}"
                except Exception:
                    abstract = f"arXiv paper: {arxiv_id}"
                    final_title = f"arXiv:{arxiv_id}"

                # Add to reference map
                reference_map[ref_num] = {
                    "url": f"[{ref_num}]({final_title})",
                    "title": final_title,
                    "text": abstract,
                }

                # Add to docs for evaluation
                docs.append({"title": final_title, "sent": abstract})

        # Map arXiv IDs to reference numbers (for the original pattern)
        for i, arxiv_id in enumerate(arxiv_matches):
            if arxiv_id not in arxiv_id_to_ref_idx:  # Skip if already processed
                if i < len(ref_matches):
                    ref_num, ref_content = ref_matches[i]
                    arxiv_id_to_ref_idx[arxiv_id] = ref_num

                    # Extract title and URL from reference content
                    ref_content = ref_content.strip()

                    # Try to extract title from the reference
                    if ". " in ref_content:
                        title = ref_content.split(". ")[0].strip()
                    else:
                        title = ref_content

                    # Try to get abstract from arXiv
                    try:
                        arxiv_title, arxiv_abstract = get_arxiv_title_and_abstract(
                            arxiv_id
                        )
                        if arxiv_abstract:
                            abstract = arxiv_abstract
                            final_title = arxiv_title
                        else:
                            abstract = ref_content
                            final_title = title
                    except Exception:
                        abstract = ref_content
                        final_title = title

                    # Add to reference map
                    reference_map[ref_num] = {
                        "url": f"[{ref_num}]({final_title})",
                        "title": final_title,
                        "text": abstract,
                    }

                    # Add to docs for evaluation
                    docs.append({"title": final_title, "sent": abstract})

        # Use the related work section as the text to process
        raw_text = related_works_section

        # Create a custom function to handle arXiv ID citations in deepresearcher format
        def process_deepresearcher_citations(text, reference_map, arxiv_id_to_ref_idx):
            """Process deepresearcher citations in format [arxiv_id], [1] [arxiv_id](url), and XML-style references"""
            import re
            from collections import OrderedDict

            link2id = OrderedDict()

            def repl(match):
                arxiv_id = match.group(1)
                print(f"Processing citation: [{arxiv_id}]")

                # Check if we have a mapping for this arXiv ID
                if arxiv_id in arxiv_id_to_ref_idx:
                    # ref_idx = arxiv_id_to_ref_idx[arxiv_id]
                    if arxiv_id not in link2id:
                        link2id[arxiv_id] = len(link2id) + 1
                    return f"[{link2id[arxiv_id]}]"
                else:
                    print(f"⚠️  No mapping found for arXiv ID: {arxiv_id}")
                    return match.group(0)

            def repl_deepresearcher(match):
                ref_num, arxiv_id = match.groups()
                print(
                    f"Processing deepresearcher citation: [{ref_num}] [{arxiv_id}](url)"
                )

                # Check if we have a mapping for this arXiv ID
                if arxiv_id in arxiv_id_to_ref_idx:
                    # ref_idx = arxiv_id_to_ref_idx[arxiv_id]
                    if arxiv_id not in link2id:
                        link2id[arxiv_id] = len(link2id) + 1
                    return f"[{link2id[arxiv_id]}]"
                else:
                    print(
                        f"⚠️  No mapping found for deepresearcher arXiv ID: {arxiv_id}"
                    )
                    return match.group(0)

            # Pattern to match [arxiv_id] citations - only match if not already a numbered citation
            pattern = r"\[(\d+\.\d+(?:v\d+)?)\]"
            clean_text = re.sub(pattern, repl, text)

            # Pattern to match deepresearcher citations: [1] [arxiv_id](url)
            deepresearcher_pattern = (
                r"\[(\d+)\]\s*\[(\d+\.\d+(?:v\d+)?)\]\(http://arxiv\.org/abs/\2\)"
            )
            clean_text = re.sub(deepresearcher_pattern, repl_deepresearcher, clean_text)

            # Debug: Check for any remaining arXiv IDs or duplicate citations
            remaining_arxiv = re.findall(r"\[(\d+\.\d+(?:v\d+)?)\]", clean_text)
            if remaining_arxiv:
                print(
                    f"⚠️  Warning: Found remaining arXiv IDs after replacement: {remaining_arxiv}"
                )

            # Check for duplicate citations like [2] [2]
            duplicate_pattern = r"\[(\d+)\]\s*\[(\d+)\]"
            duplicates = re.findall(duplicate_pattern, clean_text)
            if duplicates:
                print(f"⚠️  Warning: Found duplicate citations: {duplicates}")
                # Remove duplicates
                clean_text = re.sub(r"\[(\d+)\]\s*\[(\d+)\]", r"[\1]", clean_text)

            return clean_text

        updated_text = process_deepresearcher_citations(
            raw_text, reference_map, arxiv_id_to_ref_idx
        )
        answer = updated_text
        return answer, [(doc["title"], doc["sent"]) for doc in docs]  # type: ignore
