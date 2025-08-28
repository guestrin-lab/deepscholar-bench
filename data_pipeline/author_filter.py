"""
Author filtering module for h-index based paper filtering.
"""

import asyncio
import logging
import time
import requests  # type: ignore
from dataclasses import dataclass

try:
    from config import PipelineConfig
    from arxiv_scraper import ArxivPaper
    from utils import clean_author_name
except ImportError:
    from .config import PipelineConfig
    from .arxiv_scraper import ArxivPaper
    from .utils import clean_author_name

logger = logging.getLogger(__name__)


@dataclass
class AuthorInfo:
    """Information about a paper's author."""

    name: str
    affiliation: str | None = None
    hindex: int | None = None
    citations: int | None = None


class AuthorFilter:
    """Filter papers based on author h-index criteria."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._author_cache: dict[str, AuthorInfo] = {}

    async def filter_papers_by_author_hindex(
        self, papers: list[ArxivPaper]
    ) -> list[ArxivPaper]:
        """
        Filter papers based on author h-index criteria.

        Args:
            papers: list of ArxivPaper objects to filter

        Returns:
            Filtered list of papers where at least one author meets h-index criteria
        """
        filtered_papers: list[ArxivPaper] = []

        for paper in papers:
            try:
                meets_criteria = await self._paper_meets_hindex_criteria(paper)
                if meets_criteria:
                    filtered_papers.append(paper)

                await asyncio.sleep(self.config.request_delay)

            except Exception as e:
                logger.warning(
                    f"Error checking h-index for paper {paper.arxiv_id}: {e}"
                )
                continue

        logger.info(
            f"Filtered {len(papers)} papers to {len(filtered_papers)} based on h-index criteria"
        )
        return filtered_papers

    async def _paper_meets_hindex_criteria(self, paper: ArxivPaper) -> bool:
        """Check if a paper meets the h-index criteria."""
        tasks = [self._get_author_info(author_name) for author_name in paper.authors]
        author_infos = await asyncio.gather(*tasks, return_exceptions=True)

        for author_info in author_infos:
            if isinstance(author_info, Exception):
                continue
            assert isinstance(author_info, AuthorInfo)
            if author_info and author_info.hindex is not None:
                if self._author_meets_criteria(author_info):
                    return True
        return False

    def _author_meets_criteria(self, author_info: AuthorInfo) -> bool:
        """Check if an author meets the h-index criteria."""
        if author_info.hindex is None:
            return False
        if author_info.hindex < self.config.min_author_hindex:
            return False
        if (
            self.config.max_author_hindex
            and author_info.hindex > self.config.max_author_hindex
        ):
            return False
        return True

    async def _get_author_info(self, author_name: str) -> AuthorInfo | None:
        """Get author information including h-index."""
        clean_name = clean_author_name(author_name)
        if clean_name in self._author_cache:
            return self._author_cache[clean_name]
        author_info = None

        try:
            author_info = await self._get_author_from_google_scholar(clean_name)
        except Exception as e:
            logger.debug(f"Google Scholar lookup failed for {clean_name}: {e}")

        # If Google Scholar fails, try Semantic Scholar
        if not author_info or author_info.hindex is None:
            try:
                semantic_info = await self._get_author_from_semantic_scholar(clean_name)
                if semantic_info and semantic_info.hindex is not None:
                    author_info = semantic_info
            except Exception as e:
                logger.debug(f"Semantic Scholar lookup failed for {clean_name}: {e}")

        author_info = author_info or AuthorInfo(name=clean_name, hindex=None)
        self._author_cache[clean_name] = author_info
        return author_info

    async def _get_author_from_google_scholar(
        self, author_name: str
    ) -> AuthorInfo | None:
        """
        Get author information from Google Scholar.
        """
        raise NotImplementedError("Google Scholar lookup not implemented")

    async def _get_author_from_semantic_scholar(
        self, author_name: str
    ) -> AuthorInfo | None:
        """Get author information from Semantic Scholar API."""
        try:
            # Sleep to avoid rate limiting
            time.sleep(3)
            # Semantic Scholar API endpoint
            url = "https://api.semanticscholar.org/graph/v1/author/search"
            params = {
                "query": author_name,
                "limit": 1,
                "fields": "name,hIndex,citationCount,affiliations",
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get("data") and len(data["data"]) > 0:
                author_data = data["data"][0]

                # Extract affiliation
                affiliation = None
                if author_data.get("affiliations"):
                    affiliation = (
                        author_data["affiliations"][0]
                        if author_data["affiliations"]
                        else None
                    )

                return AuthorInfo(
                    name=author_data.get("name", author_name),
                    hindex=author_data.get("hIndex"),
                    citations=author_data.get("citationCount"),
                    affiliation=affiliation,
                )

        except Exception as e:
            logger.warning(f"Semantic Scholar API error for {author_name}: {e}")

        return None
