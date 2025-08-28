"""
ArXiv scraper for collecting papers based on specified criteria.
"""

import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
import arxiv

try:
    from config import PipelineConfig
except ImportError:
    from .config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Represents a paper from ArXiv."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_date: datetime
    updated_date: datetime
    abs_url: str
    doi: str | None = None
    journal_ref: str | None = None
    comments: str | None = None

    def __hash__(self):
        return hash(self.arxiv_id)

    def __eq__(self, other):
        return self.arxiv_id == other.arxiv_id


class ArxivScraper:
    """Scraper for collecting papers from ArXiv"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.client = arxiv.Client(
            page_size=100, delay_seconds=self.config.request_delay, num_retries=3
        )

    async def search_papers(self) -> list[ArxivPaper]:
        """
        Search for papers on ArXiv based on configuration.

        Returns:
            list of ArxivPaper objects matching the criteria.
        """
        all_papers = []

        for category in self.config.arxiv_categories:
            logger.info(f"Searching category: {category}")
            category_papers = await self._search_category(category)
            all_papers.extend(category_papers)

            # Add delay between categories to be respectful
            await asyncio.sleep(self.config.request_delay)

        unique_papers = list(set(all_papers))
        logger.info(f"Found {len(unique_papers)} unique papers across all categories")

        return unique_papers

    async def fetch_paper_by_id(self, arxiv_id: str) -> ArxivPaper | None:
        """
        Fetch a single paper by ArXiv ID.

        Args:
            arxiv_id: ArXiv ID (e.g., "2502.07374" or "arxiv:2502.07374")

        Returns:
            ArxivPaper object if found, None otherwise.
        """
        # Clean the ArXiv ID (remove "arxiv:" prefix if present)
        clean_id = arxiv_id.replace("arxiv:", "").strip()

        try:
            # Create search for specific paper ID
            search = arxiv.Search(id_list=[clean_id])
            results = list(self.client.results(search))

            if results:
                result = results[0]
                paper = self._convert_result_to_paper(result)
                logger.info(f"Successfully fetched paper: {clean_id}")
                return paper
            else:
                logger.warning(f"Paper not found: {clean_id}")
                return None

        except Exception as e:
            logger.error(f"Error fetching paper {clean_id}: {e}")
            return None

    async def _search_category(self, category: str) -> list[ArxivPaper]:
        """Search for papers in a specific category."""
        papers = []

        # Build date range query
        start_date_str = self.config.start_date.strftime("%Y%m%d")
        end_date_str = self.config.end_date.strftime("%Y%m%d")

        # Create search query
        search_query = f"cat:{category} AND submittedDate:[{start_date_str}0000 TO {end_date_str}2359]"

        try:
            # Create search object
            search = arxiv.Search(
                query=search_query,
                max_results=self.config.max_papers_per_category,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )

            # Get results
            results = list(self.client.results(search))

            # Convert results to ArxivPaper objects
            for result in results:
                paper = self._convert_result_to_paper(result)
                if paper:
                    papers.append(paper)

        except Exception as e:
            logger.error(f"Error fetching papers for category {category}: {e}")

        return papers[: self.config.max_papers_per_category]

    def _convert_result_to_paper(self, result: arxiv.Result) -> ArxivPaper | None:
        """Convert arxiv.Result to ArxivPaper object."""
        try:
            return ArxivPaper(
                arxiv_id=result.entry_id.split("/")[-1],
                title=result.title,
                authors=[author.name for author in result.authors],
                abstract=result.summary,
                categories=result.categories,
                published_date=result.published,
                updated_date=result.updated,
                abs_url=result.entry_id,
                doi=getattr(result, "doi", None),
                journal_ref=getattr(result, "journal_ref", None),
                comments=getattr(result, "comment", None),
            )

        except Exception as e:
            logger.error(f"Error converting result to paper: {e}")
            return None
