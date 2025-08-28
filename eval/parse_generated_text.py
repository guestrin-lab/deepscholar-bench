import re
from eval.utils import (
    get_arxiv_title_and_abstract,
    get_arxiv_abstract_by_title,
    get_arxiv_abstract,
)


def replace_refs(text: str, ref_map: dict) -> str:
    def repl(match):
        idx = match.group(1)
        return ref_map.get(idx, {}).get("url", match.group(0))

    return re.sub(r"\[(\d+)\]", repl, text)


def replace_latex_cites(text: str, ref_map: dict) -> str:
    def repl(match):
        cite_key = match.group(1)
        for ref_id, ref_info in ref_map.items():
            if cite_key in ref_info.get("url", ""):
                return ref_info.get("url", f"[{cite_key}]")
        return f"[{cite_key}]"

    return re.sub(r"\\cite\{([^}]+)\}", repl, text)


def remove_md_links(text: str) -> str:
    """
    Removes all Markdown-style hyperlinks from the text.
    Examples:
    - [OpenAI](https://openai.com) -> OpenAI
    - https://openai.com -> (removed)
    """
    # Replace [text](link) with just 'text'
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Optionally remove bare URLs (if desired)
    text = re.sub(r"https?://\S+", "", text)

    return text


def extract_reference_section(text: str) -> str:
    """Extract the reference section from the text"""
    # Split by common reference section markers
    reference_sections = []
    for marker in [
        "References",
        "<references>",
        "**References**",
        "## References",
        "# References",
    ]:
        if marker in text:
            parts = text.split(marker)
            if len(parts) > 1:
                reference_sections.append(parts[1])

    if not reference_sections:
        # If no explicit reference section found, look for bracketed references pattern
        bracketed_refs = re.findall(r"\[(\d+)\]\s*[^[]+", text)
        if bracketed_refs:
            # Find the start of the first bracketed reference
            start_match = re.search(r"\[1\]", text)
            if start_match:
                return text[start_match.start() :]

    if reference_sections:
        return reference_sections[0]

    return ""


def extract_reference_lines(text: str) -> list[str]:
    """Extract individual reference lines from the reference section"""
    reference_section = extract_reference_section(text)
    if not reference_section:
        return []

    # Handle different reference formats
    lines = []

    # Pattern 1: Bracketed references [1], [2], etc.
    bracketed_pattern = r"\[(\d+)\]\s*(.*?)(?=\n\s*\[\d+\]|\n\s*$|$)"
    bracketed_matches = re.findall(bracketed_pattern, reference_section, re.DOTALL)
    for ref_num, content in bracketed_matches:
        lines.append(content.strip())

    # Pattern 2: Numbered references 1., 2., etc.
    if not lines:  # Only if no bracketed references found
        numbered_pattern = r"\n\s*(\d+)\.\s*(.*?)(?=\n\s*\d+\.|\n\s*$|$)"
        numbered_matches = re.findall(numbered_pattern, reference_section, re.DOTALL)
        for ref_num, content in numbered_matches:
            lines.append(content.strip())

    # Pattern 3: Simple line-by-line split for other formats
    if not lines:
        # Split by newlines and filter out empty lines
        lines = [line.strip() for line in reference_section.split("\n") if line.strip()]
        # Remove lines that are just numbers or brackets
        lines = [line for line in lines if not re.match(r"^\s*[\d\[\]]+\s*$", line)]

    return lines


def extract_arxiv_ids_from_text(text: str) -> tuple[list[str], list[str], str]:
    """Extract all ArXiv IDs from text in various formats"""
    arxiv_ids = []
    clean_text = text
    # Pattern 1: Full URLs like http://arxiv.org/abs/2311.05822v2
    url_pattern = r"https?://arxiv\.org/(?:abs|html|pdf)/(\d+\.\d+(?:v\d+)?)"
    url_matches = re.findall(url_pattern, text)
    clean_text = re.sub(url_pattern, "", clean_text)
    arxiv_ids.extend(url_matches)

    # Pattern 2: ArXiv format like arXiv:2501.08573v1
    arxiv_format_pattern = r"arXiv:(\d+\.\d+(?:v\d+)?)"
    arxiv_format_matches = re.findall(arxiv_format_pattern, text)
    clean_text = re.sub(arxiv_format_pattern, "", clean_text)
    arxiv_ids.extend(arxiv_format_matches)

    # Pattern 3: Standalone IDs like 2311.05822v2 (with word boundaries to avoid false positives)
    standalone_pattern = r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b"
    standalone_matches = re.findall(standalone_pattern, text)
    clean_text = re.sub(standalone_pattern, "", clean_text)
    arxiv_ids.extend(standalone_matches)

    # Pattern 4: Bracketed IDs like [2311.05822v2]
    bracketed_pattern = r"\[(\d+\.\d+(?:v\d+)?)\]"
    bracketed_matches = re.findall(bracketed_pattern, text)
    clean_text = re.sub(bracketed_pattern, "", clean_text)
    arxiv_ids.extend(bracketed_matches)

    # Normalize ArXiv IDs by removing version numbers for uniqueness
    normalized_ids = []
    unique_base_ids = set()

    for arxiv_id in arxiv_ids:
        # Extract the base ID (without version)
        base_id = re.sub(r"v\d+$", "", arxiv_id)
        if base_id not in unique_base_ids:
            unique_base_ids.add(base_id)
            normalized_ids.append(arxiv_id)  # Keep original ID for display

    return arxiv_ids, normalized_ids, clean_text


def process_inline_arxiv_titles(reference_section: str) -> list[tuple[str, str]]:
    """Extract and process reference titles from a reference section."""
    citations = []
    reference_title_pattern = r"\[(\d+)\]\s*(.+?)(?:[.:]\s*|$)"
    matches = re.findall(reference_title_pattern, reference_section)

    for ref_num, title in matches:
        title = title.strip()
        if len(title) > 10:  # Only process meaningful titles
            try:
                search_title, search_abstract = get_arxiv_abstract_by_title(title)
                final_title = (
                    search_title if search_title and search_abstract else title
                )
                final_abstract = (
                    search_abstract if search_title and search_abstract else ""
                )
                citations.append((final_title, final_abstract))
            except Exception:
                citations.append((title, ""))

    return citations


def process_inline_citations(generated_text: str) -> list[tuple[str, str]]:
    """Extract inline citations from generated text using multiple patterns."""
    # Define patterns with titles
    title_patterns = [
        (
            r"\[(\d+)\]\s*([^(]+)\s*\(http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)\)",
            "parentheses",
        ),
        (
            r"\[(\d+)\]\s*([^h]+?)\s*http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)",
            "no parentheses",
        ),
        (
            r"\[(\d+)\]\s*(.+?)[.:]\s*http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)",
            "SearchAI format",
        ),
        (
            r"\[(\d+)\]\s*(.+?)\s+http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)",
            "SearchAI format no punctuation",
        ),
        (r"\[(\d+)\]\s*(.+?)[.:]\s*arXiv:(\d+\.\d+(?:v\d+)?)", "arXiv prefix"),
        (
            r"\[(\d+)\]\s*(.+?)\s+arXiv:(\d+\.\d+(?:v\d+)?)",
            "arXiv prefix no punctuation",
        ),
        (
            r"\[(\d+)\]\s*(.+?)\s*\(arXiv:(\d+\.\d+(?:v\d+)?)\)",
            "arXiv prefix parentheses",
        ),
        (r"\[(\d+)\]\s*(.+?)\s*\[arXiv:(\d+\.\d+(?:v\d+)?)\]", "arXiv prefix brackets"),
        (
            r"(\d+)\.\s*(.+?)[.:]\s*http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)",
            "numbered list",
        ),
        (
            r"(\d+)\.\s*(.+?)\s+http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)",
            "numbered list no punctuation",
        ),
        (
            r"(\d+)\.\s*(.+?)[.:]\s*arXiv:(\d+\.\d+(?:v\d+)?)",
            "numbered list arXiv prefix",
        ),
        (
            r"(\d+)\.\s*(.+?)\s+arXiv:(\d+\.\d+(?:v\d+)?)",
            "numbered list arXiv prefix no punctuation",
        ),
    ]

    # Patterns without titles
    no_title_patterns = [
        (r"\[(\d+)\]\s*https://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)", "simple SearchAI"),
        (
            r"(\d+)\.\s*http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)",
            "numbered list URL only",
        ),
        (r"(\d+)\.\s*arXiv:(\d+\.\d+(?:v\d+)?)", "numbered list arXiv prefix only"),
    ]

    all_matches = []

    # Process patterns with titles
    for pattern, description in title_patterns:
        matches = re.findall(pattern, generated_text)
        all_matches.extend(matches)

    # Process patterns without titles
    for pattern, description in no_title_patterns:
        matches = re.findall(pattern, generated_text)
        for match in matches:
            ref_num, arxiv_id = match
            try:
                fetched_title, _ = get_arxiv_title_and_abstract(arxiv_id)
                title = fetched_title or f"arXiv:{arxiv_id}"
            except Exception as _:
                title = f"arXiv:{arxiv_id}"
            all_matches.append((ref_num, title, arxiv_id))

    # Fetch abstracts for all matches
    citations = []
    for ref_num, title, arxiv_id in all_matches:
        try:
            fetched_title, abstract = get_arxiv_title_and_abstract(arxiv_id)
            final_title = fetched_title or title.strip()
            final_abstract = abstract or ""
            citations.append((final_title, final_abstract))
        except Exception:
            citations.append((title.strip(), ""))

    return citations


def parse_arxiv_references_from_markdown_references_section(
    content: str,
) -> list[tuple[str, str]]:
    """Parse arXiv references from markdown content and fetch their titles and abstracts."""
    citations = []
    seen_titles: set[str] = set()

    # Extract reference section
    reference_section = content
    for ref_header in ["References:", "References"]:
        if ref_header in content:
            reference_section = content.split(ref_header)[-1]
            break

    # Try structured reference patterns
    pattern_functions = [
        _try_quoted_references,
        _try_unquoted_references,
        _try_bracket_arxiv_references,
        _try_markdown_link_references,
    ]

    for pattern_func in pattern_functions:
        citations.extend(pattern_func(reference_section, seen_titles))
        if citations:
            break

    # Fallback to simple patterns if no structured references found
    if not citations:
        citations.extend(_try_fallback_patterns(content, seen_titles))

    return citations


def _try_quoted_references(
    reference_section: str, seen_titles: set[str]
) -> list[tuple[str, str]]:
    """Try pattern: [1] Authors, "Title." arXiv:ID, Year."""
    pattern = r'\[(\d+)\]\s+([^"]*)"([^"]+)"[^.]*arXiv:(\d+\.\d+(?:v\d+)?)[^.]*(\d{4})'
    matches = re.findall(pattern, reference_section)

    citations: list[tuple[str, str]] = []
    for ref_num, authors, title, arxiv_id, year in matches:
        title = title.strip()
        arxiv_id = arxiv_id.strip()

        if title and arxiv_id and _add_unique_citation(citations, seen_titles, title):
            fetched_title, abstract = get_arxiv_abstract(
                title, arxiv_id, check_using_sequence_matcher=False
            )
            citations.append((fetched_title, abstract))

    return citations


def _try_unquoted_references(
    reference_section: str, seen_titles: set[str]
) -> list[tuple[str, str]]:
    """Try pattern: [1] Authors, Title. arXiv:ID, Year."""
    pattern = r"\[(\d+)\]\s+([^.]*?)\s+arXiv:(\d+\.\d+(?:v\d+)?)[^.]*(\d{4})"
    matches = re.findall(pattern, reference_section)

    citations: list[tuple[str, str]] = []
    for ref_num, authors_title, arxiv_id, year in matches:
        title = _extract_title_from_authors_title(authors_title)
        arxiv_id = arxiv_id.strip()

        if title and arxiv_id and _add_unique_citation(citations, seen_titles, title):
            fetched_title, abstract = get_arxiv_abstract(title, arxiv_id)
            citations.append((fetched_title, abstract))

    return citations


def _try_bracket_arxiv_references(
    reference_section: str, seen_titles: set[str]
) -> list[tuple[str, str]]:
    """Try pattern: [1] Title [ID]"""
    pattern = r"\[(\d+)\]\s+([^[]*?)\s*\[(\d+\.\d+(?:v\d+)?)\](?:v\d+)?"
    matches = re.findall(pattern, reference_section)

    citations: list[tuple[str, str]] = []
    for ref_num, title, arxiv_id in matches:
        title = title.strip()
        arxiv_id = arxiv_id.strip()

        if title and arxiv_id and _add_unique_citation(citations, seen_titles, title):
            fetched_title, abstract = get_arxiv_abstract(title, arxiv_id)
            citations.append((fetched_title, abstract))

    return citations


def _try_markdown_link_references(
    reference_section: str, seen_titles: set[str]
) -> list[tuple[str, str]]:
    """Try pattern: [1] [Title](http://arxiv.org/abs/ID)"""
    pattern = r"\[(\d+)\]\s*\[([^\]]+)\]\(http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)\)"
    matches = re.findall(pattern, reference_section)

    citations: list[tuple[str, str]] = []
    for ref_num, title, arxiv_id in matches:
        title = title.strip()
        arxiv_id = arxiv_id.strip()

        if title and arxiv_id and _add_unique_citation(citations, seen_titles, title):
            fetched_title, abstract = get_arxiv_abstract(title, arxiv_id)
            citations.append((fetched_title, abstract))

    return citations


def _try_fallback_patterns(
    content: str, seen_titles: set[str]
) -> list[tuple[str, str]]:
    """Try various fallback patterns when structured references aren't found."""
    citations: list[tuple[str, str]] = []

    # Traditional citations with quotes
    traditional_pattern = r'\[(\d+)\]\s+([^,]+),\s*"([^"]+)"'
    traditional_matches = re.findall(traditional_pattern, content)

    for ref_num, author, title in traditional_matches:
        title = title.strip()
        if (
            title
            and len(title) > 5
            and _add_unique_citation(citations, seen_titles, title)
        ):
            citations.append((title, ""))

    # ArXiv ID patterns
    arxiv_patterns = [
        (r"arXiv:(\d+\.\d+(?:v\d+)?)", "ArXiv ID"),
        (r"\[(\d+\.\d+(?:v\d+)?)\]", "bracket ArXiv ID"),
        (
            r"\[([^\]]+)\]\(http://arxiv\.org/abs/(\d+\.\d+(?:v\d+)?)\)",
            "markdown link ArXiv ID",
        ),
    ]

    for pattern, pattern_name in arxiv_patterns:
        matches = re.findall(pattern, content)

        for match in matches:
            if pattern_name == "markdown link ArXiv ID":
                title, arxiv_id = match
            else:
                arxiv_id = match

            try:
                fetched_title, abstract = get_arxiv_title_and_abstract(arxiv_id)

                if (
                    fetched_title
                    and len(fetched_title) > 5
                    and _add_unique_citation(citations, seen_titles, fetched_title)
                ):
                    citations.append((fetched_title, abstract or ""))
            except Exception as e:
                print(f"Error fetching title and abstract for {arxiv_id}: {e}")

    return citations


def _extract_title_from_authors_title(authors_title: str) -> str:
    """Extract title from combined authors and title string."""
    parts = authors_title.split(",")
    if len(parts) > 1:
        title = parts[-1].strip()
        # Remove common author patterns
        title = re.sub(r"\s*et al\.?\s*$", "", title)
        title = re.sub(r"\s*and\s+[^,]+$", "", title)
    else:
        title = authors_title.strip()
    return title


def _add_unique_citation(
    citations: list[tuple[str, str]], seen_titles: set[str], title: str
) -> bool:
    """Check if title is unique and add to seen_titles if so."""
    normalized_title = re.sub(r"[^\w\s]", "", title.lower()).strip()
    if normalized_title not in seen_titles:
        seen_titles.add(normalized_title)
        return True
    return False
