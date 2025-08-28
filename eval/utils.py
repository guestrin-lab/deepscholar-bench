import pandas as pd
import re
import requests  # type: ignore
from typing import List, Dict
import os
import logging
import urllib.parse
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)


def escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def save_csv_with_append(
    df: pd.DataFrame, output_file: str, key_columns: List[str]
) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file)
            existing_df = existing_df.drop_duplicates(subset=key_columns, keep="last")
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=key_columns, keep="last")
            combined_df.to_csv(output_file, index=False)
            logger.info(f"Appended results to {output_file}")
        except Exception as e:
            logger.error(f"Error loading existing file: {e}")
            df.to_csv(output_file, index=False)
            logger.info(f"Created new results file: {output_file}")
    else:
        df.to_csv(output_file, index=False)
        logger.info(f"Created new results file: {output_file}")


def check_arxiv_id(
    file_id: str, dataset: pd.DataFrame, arxiv_to_mapped_id: Dict
) -> str:
    for idx, row in dataset.iterrows():
        if idx == int(file_id) and row["arxiv_id"] in arxiv_to_mapped_id:
            arxiv_id = row["arxiv_id"]
            break
    return arxiv_id


def jaccard_similarity(str1, str2):
    tokens1 = set(str1.lower().split())
    tokens2 = set(str2.lower().split())
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union) if union else 0


def get_citation_count_from_title(
    title, mailto="your@email.com", similarity_threshold=0.8
):
    try:
        search_url = f"https://api.openalex.org/works?search={title}&mailto={mailto}"
        response = requests.get(search_url, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])

        if results:
            top_result = results[0]
            paper_title = top_result.get("display_name", "")
            citation_count = top_result.get("cited_by_count", 0)

            # Compare similarity
            similarity = jaccard_similarity(title, paper_title)
            if similarity >= similarity_threshold:
                return citation_count
            else:
                return None
    except Exception as e:
        print(f"Error fetching citation count: {e}")
        return None


# === Get valid arXiv links and mapped IDs ===
def get_valid_arxiv_links_and_ids(id_map_path: str, paper_content_path: str):
    id_map_df = pd.read_csv(id_map_path, dtype={"arxiv_id": str})
    paper_df = pd.read_csv(paper_content_path, dtype={"arxiv_id": str})
    valid_id_map_df = id_map_df[id_map_df["arxiv_id"].isin(paper_df["arxiv_id"])]
    valid_links = paper_df["arxiv_link"].tolist()
    valid_mapped_ids = valid_id_map_df.index.tolist()
    return valid_links, valid_mapped_ids, valid_id_map_df


def get_arxiv_title_and_abstract(arxiv_id: str) -> tuple[str, str]:
    """Fetch title and abstract from arXiv by ID"""
    try:
        # First try the direct arXiv page
        url = f"https://arxiv.org/abs/{arxiv_id}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            # Extract title - updated pattern
            title_pattern = r'<h1 class="title">\s*([^<]+)</h1>'
            title_match = re.search(title_pattern, response.text)
            title = title_match.group(1).strip() if title_match else ""

            # Extract abstract - updated pattern
            abstract_pattern = (
                r'<blockquote class="abstract">\s*<span[^>]*>([^<]+)</span>'
            )
            abstract_match = re.search(abstract_pattern, response.text)
            abstract = abstract_match.group(1).strip() if abstract_match else ""

            # If abstract is empty, try alternative pattern
            if not abstract:
                alt_abstract_pattern = (
                    r'<blockquote class="abstract">\s*([^<]+)</blockquote>'
                )
                alt_abstract_match = re.search(alt_abstract_pattern, response.text)
                abstract = (
                    alt_abstract_match.group(1).strip() if alt_abstract_match else ""
                )

            if title and abstract:
                return title, abstract

        # Fallback to arXiv API
        try:
            api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
            api_response = requests.get(api_url, timeout=10)
            if api_response.status_code == 200:
                import xml.etree.ElementTree as ET

                root = ET.fromstring(api_response.text)
                entry = root.find(".//{http://www.w3.org/2005/Atom}entry")
                if entry is not None:
                    title_elem = entry.find(".//{http://www.w3.org/2005/Atom}title")
                    abstract_elem = entry.find(
                        ".//{http://www.w3.org/2005/Atom}summary"
                    )

                    api_title = (
                        title_elem.text.strip()
                        if title_elem is not None and title_elem.text is not None
                        else ""
                    )
                    api_abstract = (
                        abstract_elem.text.strip()
                        if abstract_elem is not None and abstract_elem.text is not None
                        else ""
                    )

                    if api_title and api_abstract:
                        return api_title, api_abstract
        except Exception as api_e:
            print(f"API fallback failed for arXiv {arxiv_id}: {api_e}")

        return "", ""
    except Exception as e:
        print(f"Error fetching title and abstract for arXiv {arxiv_id}: {e}")
        return "", ""


def get_arxiv_abstract_by_title(
    title: str, similarity_threshold: float = 0.8
) -> tuple[str, str]:
    """Search arXiv by title and return the title and abstract if found"""
    try:
        # URL encode the title for search
        encoded_title = urllib.parse.quote(title)
        search_url = f"https://arxiv.org/search/?query={encoded_title}&searchtype=all&source=header"

        response = requests.get(search_url, timeout=10)
        if response.status_code == 200:
            # Extract title and abstract from search results
            import re

            # Look for title in search results - updated pattern
            title_pattern = r'<h3 class="title">\s*<a[^>]*>([^<]+)</a>'
            title_matches = re.findall(title_pattern, response.text)

            if title_matches:
                # Check similarity with the first result
                from difflib import SequenceMatcher

                similarity = SequenceMatcher(
                    None, title.lower(), title_matches[0].lower()
                ).ratio()

                if similarity >= similarity_threshold:
                    # Extract abstract - updated pattern
                    abstract_pattern = (
                        r'<span class="abstract-full">\s*<span[^>]*>([^<]+)</span>'
                    )
                    abstract_matches = re.findall(abstract_pattern, response.text)
                    if abstract_matches:
                        return title_matches[0].strip(), abstract_matches[0].strip()
                    else:
                        # Try alternative abstract pattern
                        alt_abstract_pattern = (
                            r'<span class="abstract-full">\s*([^<]+)</span>'
                        )
                        alt_abstract_matches = re.findall(
                            alt_abstract_pattern, response.text
                        )
                        if alt_abstract_matches:
                            return title_matches[0].strip(), alt_abstract_matches[
                                0
                            ].strip()

        # If search fails, try direct arXiv API
        try:
            # Try to extract arXiv ID from title and use direct API
            api_url = f'https://export.arxiv.org/api/query?search_query=ti:"{title}"&start=0&max_results=1'
            api_response = requests.get(api_url, timeout=10)
            if api_response.status_code == 200:
                # Parse XML response
                import xml.etree.ElementTree as ET

                root = ET.fromstring(api_response.text)
                entry = root.find(".//{http://www.w3.org/2005/Atom}entry")
                if entry is not None:
                    title_elem = entry.find(".//{http://www.w3.org/2005/Atom}title")
                    abstract_elem = entry.find(
                        ".//{http://www.w3.org/2005/Atom}summary"
                    )

                    api_title = (
                        title_elem.text.strip()
                        if title_elem is not None and title_elem.text is not None
                        else ""
                    )
                    api_abstract = (
                        abstract_elem.text.strip()
                        if abstract_elem is not None and abstract_elem.text is not None
                        else ""
                    )

                    if api_title and api_abstract:
                        return api_title, api_abstract
        except Exception as api_e:
            print(f"API fallback failed for '{title}': {api_e}")

        return "", ""
    except Exception as e:
        print(f"Error fetching abstract for title '{title}': {e}")
        return "", ""


def get_arxiv_abstract(
    title: str, arxiv_id: str, check_using_sequence_matcher: bool = True
) -> tuple[str, str]:
    """Fetch arXiv abstract with optional title matching validation."""
    try:
        fetched_title, abstract = get_arxiv_title_and_abstract(arxiv_id)

        if not check_using_sequence_matcher:
            final_title = (
                fetched_title
                if fetched_title and len(fetched_title) > len(title)
                else title
            )
            return final_title, abstract or ""

        # Check title similarity
        title_similarity = 0.0
        if fetched_title:
            from difflib import SequenceMatcher

            title_similarity = SequenceMatcher(
                None, title.lower(), fetched_title.lower()
            ).ratio()
            print(
                f"Title similarity: {title_similarity:.2f} (expected: '{title}' vs fetched: '{fetched_title}')"
            )

        # Use title search if similarity is low
        if title_similarity < 0.7 and fetched_title:
            print(
                f"⚠️  arXiv ID {arxiv_id} returned different title, searching by title instead..."
            )
            try:
                search_title, search_abstract = get_arxiv_abstract_by_title(title)
                if search_abstract:
                    print(f"✅ Found matching paper by title search: '{title}'")
                    return title, search_abstract
                else:
                    print("❌ Title search failed, using arXiv ID result")
            except Exception as search_e:
                print(f"❌ Title search failed: {search_e}")

        # Use fetched results
        final_title = fetched_title or title
        final_abstract = abstract or ""

        if final_abstract:
            print(f"✅ Final result for '{final_title}': {final_abstract[:100]}...")
        else:
            print(f"⚠️  No abstract found for '{final_title}', using title only")

        return final_title, final_abstract

    except Exception as e:
        print(f"❌ Error fetching abstract for arXiv {arxiv_id}: {e}")
        return title, ""


def extract_html_content(url: str, max_retries: int = 3) -> str:
    """Extract text content from HTML page"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")

                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()

                # Extract text content
                text = soup.get_text()

                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (
                    phrase.strip() for line in lines for phrase in line.split("  ")
                )
                text = " ".join(chunk for chunk in chunks if chunk)

                # Limit to first 2000 characters to avoid too long content
                if len(text) > 2000:
                    text = text[:2000] + "..."

                return text
            else:
                print(f"HTTP {response.status_code} for {url}")
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retry

    return ""
