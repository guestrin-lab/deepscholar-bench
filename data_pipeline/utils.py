import pandas as pd

try:
    from arxiv_scraper import ArxivPaper
except ImportError:
    from .arxiv_scraper import ArxivPaper
import re


def papers_to_dataframe(papers: list[ArxivPaper] | ArxivPaper) -> pd.DataFrame:
    """
    Convert ArxivPaper object(s) to a pandas DataFrame.

    Args:
        papers: Single ArxivPaper object or list of ArxivPaper objects

    Returns:
        DataFrame containing paper information
    """
    if isinstance(papers, ArxivPaper):
        papers = [papers]

    paper_dicts = [
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": "; ".join(paper.authors),
            "abstract": paper.abstract,
            "categories": "; ".join(paper.categories),
            "published_date": paper.published_date,
            "updated_date": paper.updated_date,
            "abs_url": paper.abs_url,
            "doi": paper.doi,
            "journal_ref": paper.journal_ref,
            "comments": paper.comments,
        }
        for paper in papers
    ]

    return pd.DataFrame(paper_dicts)


def clean_author_name(name: str) -> str:
    """Clean and normalize author name for lookups."""
    cleaned = re.sub(r"\s+", " ", name.strip())
    prefixes = ["Dr.", "Prof.", "Professor", "Dr", "Prof"]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    return cleaned
