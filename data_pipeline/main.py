"""
Main pipeline script for ArXiv data collection and citation extraction.

This script orchestrates the entire pipeline:
1. Scrape ArXiv papers
2. Extract related works sections
3. Extract citations with metadata lookup
4. Generate comprehensive dataframe outputs
"""

import asyncio
import logging
import os
import pandas as pd
from datetime import datetime
from typing import Dict
import glob

try:
    from argument_parser import parse_args
    from config import PipelineConfig
    from arxiv_scraper import ArxivScraper, ArxivPaper
    from author_filter import AuthorFilter
    from latex_extractor import LatexExtractor
    from utils import papers_to_dataframe
except ImportError:
    from .argument_parser import parse_args
    from .config import PipelineConfig
    from .arxiv_scraper import ArxivScraper, ArxivPaper
    from .author_filter import AuthorFilter
    from .latex_extractor import LatexExtractor
    from .utils import papers_to_dataframe

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataPipeline:
    """Main data collection pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scraper = ArxivScraper(config)
        self.author_filter = AuthorFilter(config)
        self.latex_extractor = LatexExtractor(config)

        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)

    async def run_full_pipeline(
        self, arxiv_id: str | None = None, continue_from_failed_test: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the complete data collection pipeline.

        Args:
            arxiv_id: ArXiv ID (e.g., "2502.07374" or "arxiv:2502.07374")
            continue_from_failed_test: Whether to continue from a failed test

        Returns:
            Dictionary containing all generated dataframes
        """
        logger.info("🚀 Starting ArXiv data collection pipeline...")

        if arxiv_id:
            papers = [await self.scraper.fetch_paper_by_id(arxiv_id)]
            papers_df = papers_to_dataframe(papers)
        elif self.config.existing_papers_csv:
            logger.info(
                f"📄 Step 1: Reading existing papers from {self.config.existing_papers_csv}..."
            )
            papers_df = pd.read_csv(self.config.existing_papers_csv, index_col=0)
            papers = [
                ArxivPaper(
                    arxiv_id=row["arxiv_id"],
                    title=row["title"],
                    authors=row["authors"].split("; "),
                    abstract=row["abstract"],
                    categories=row["categories"].split("; "),
                    published_date=pd.to_datetime(row["published_date"]),
                    updated_date=pd.to_datetime(row["updated_date"]),
                    abs_url=row["abs_url"],
                    doi=row.get("doi"),
                    journal_ref=row.get("journal_ref"),
                    comments=row.get("comments"),
                )
                for _, row in papers_df.iterrows()
            ]
        else:
            # Step 1: Scrape ArXiv papers
            logger.info("📄 Step 1: Scraping ArXiv papers...")
            papers = await self.scraper.search_papers()
            logger.info(f"Found {len(papers)} papers from ArXiv")
            papers_df = papers_to_dataframe(papers)
        papers_df.to_csv(os.path.join(self.config.output_dir, "papers.csv"), index=True)
        logger.info(
            f"Saved papers dataframe to {os.path.join(self.config.output_dir, 'papers.csv')}"
        )
        if not papers:
            logger.warning("No papers found! Exiting pipeline.")
            return {}

        # # Step 2: Filter papers by author h-index
        logger.info("👥 Step 2: Filtering papers by author h-index...")
        filtered_papers = await self.author_filter.filter_papers_by_author_hindex(
            papers
        )

        if not filtered_papers:
            if not continue_from_failed_test:
                logger.warning("No papers passed author filtering! Exiting pipeline.")
                return {}
            else:
                logger.warning(
                    "No papers passed author filtering, but continuing anyway..."
                )
                filtered_papers = papers
        else:
            logger.info(f"After author filtering: {len(filtered_papers)} papers remain")

        # Step 3: Extract related works sections
        logger.info("📚 Step 3: Extracting related works sections...")
        paper_data_list = await self.latex_extractor.extract_papers_content(
            filtered_papers
        )
        if not paper_data_list:
            if not continue_from_failed_test:
                logger.warning(
                    "No papers had extractable related works sections! Exiting pipeline."
                )
                return {}
            else:
                logger.warning(
                    "No papers had extractable related works sections. Skipping related works section extraction..."
                )
                return self._generate_dataframes_for_papers_only(filtered_papers)
        else:
            logger.info(
                f"Successfully extracted related works sections from {len(paper_data_list)} papers"
            )

        # Step 4: Extract citations with metadata lookup
        logger.info("🔗 Step 4: Extracting citations and looking up metadata...")
        citations = await self.latex_extractor.extract_citations_from_papers(
            paper_data_list
        )
        logger.info(f"Extracted {len(citations)} citations total")

        # Step 5: Generate all dataframes
        logger.info("📊 Step 5: Generating output dataframes...")
        dataframes = self._generate_dataframes(
            filtered_papers, paper_data_list, citations
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Step 6: Save dataframes
        if (
            self.config.save_raw_papers
            or self.config.save_extracted_sections
            or self.config.save_citations
        ):
            logger.info("💾 Step 6: Saving dataframes to CSV files...")
            try:
                self._save_dataframes(dataframes, name=timestamp)
            except Exception:
                import pdb

                pdb.set_trace()

        # Step 7: Combine related works
        all_related_works = glob.glob(f"{self.config.output_dir}/related_works/*.csv")
        dfs = []
        for file in all_related_works:
            df = pd.read_csv(file)
            dfs.append(df)

        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df.to_csv(
            f"{self.config.output_dir}/{timestamp}/related_works_combined.csv",
            index=False,
        )
        papers_df = dataframes["papers"]
        joined_df = pd.merge(
            papers_df, combined_df, on=["arxiv_id", "title", "abstract"], how="left"
        )
        joined_df.to_csv(
            f"{self.config.output_dir}/{timestamp}/papers_with_related_works.csv",
            index=False,
        )
        logger.info("✅ Pipeline completed successfully!")
        return dataframes

    def _generate_dataframes(
        self, papers, paper_data_list, citations
    ) -> Dict[str, pd.DataFrame]:
        """Generate all required dataframes."""
        dataframes = {}

        # 1. Papers DataFrame
        papers_data = []
        for paper in papers:
            papers_data.append(
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": ", ".join(paper.authors),
                    "abstract": paper.abstract,
                    "categories": ", ".join(paper.categories),
                    "published_date": paper.published_date
                    if isinstance(paper.published_date, str)
                    else paper.published_date.isoformat()
                    if paper.published_date
                    else None,
                    "updated_date": paper.updated_date
                    if isinstance(paper.updated_date, str)
                    else paper.updated_date.isoformat()
                    if paper.updated_date
                    else None,
                    "abs_url": paper.abs_url,
                }
            )

        dataframes["papers"] = pd.DataFrame(papers_data)
        logger.info(f"Generated papers dataframe: {len(dataframes['papers'])} rows")

        # 2. Paper Content DataFrame
        content_data = []
        for paper_data in paper_data_list:
            content_data.append(
                {
                    "arxiv_link": paper_data.arxiv_link,
                    "arxiv_id": paper_data.arxiv_link.split("/")[-1],
                    "publication_date": paper_data.publication_date
                    if isinstance(paper_data.publication_date, str)
                    else paper_data.publication_date.isoformat()
                    if paper_data.publication_date
                    else None,
                    "paper_title": paper_data.paper_title,
                    "abstract": paper_data.abstract,
                    "related_works_section": paper_data.related_works_section,
                    "related_works_length": len(paper_data.related_works_section)
                    if paper_data.related_works_section
                    else 0,
                }
            )

        dataframes["paper_content"] = pd.DataFrame(content_data)
        logger.info(
            f"Generated paper content dataframe: {len(dataframes['paper_content'])} rows"
        )

        # 3. Citations DataFrame
        citations_data = []
        for citation in citations:
            citations_data.append(
                {
                    "parent_paper_title": citation.parent_paper_title,
                    "parent_paper_arxiv_id": citation.parent_arxiv_link.split("/")[-1],
                    "citation_shorthand": citation.citation_shorthand,
                    "raw_citation_text": citation.raw_citation_text,
                    "cited_paper_title": citation.cited_paper_title,
                    "cited_paper_arxiv_link": citation.cited_paper_arxiv_link,
                    "cited_paper_abstract": citation.cited_paper_abstract,
                    "has_metadata": bool(citation.cited_paper_title),
                    "is_arxiv_paper": bool(
                        citation.cited_paper_arxiv_link
                        and "arxiv" in citation.cited_paper_arxiv_link.lower()
                    ),
                    "bib_paper_authors": citation.bib_paper_authors
                    if citation.bib_paper_authors is not None
                    else None,
                    "bib_paper_year": citation.bib_paper_year
                    if citation.bib_paper_year is not None
                    else None,
                    "bib_paper_month": citation.bib_paper_month
                    if citation.bib_paper_month is not None
                    else None,
                    "bib_paper_url": citation.bib_paper_url
                    if citation.bib_paper_url is not None
                    else None,
                    "bib_paper_doi": citation.bib_paper_doi
                    if citation.bib_paper_doi is not None
                    else None,
                    "bib_paper_journal": citation.bib_paper_journal
                    if citation.bib_paper_journal is not None
                    else None,
                }
            )

        dataframes["citations"] = pd.DataFrame(citations_data)
        logger.info(
            f"Generated citations dataframe: {len(dataframes['citations'])} rows"
        )

        # 4. Citation Statistics DataFrame
        stats_data = []
        for paper_data in paper_data_list:
            paper_citations = [
                c for c in citations if c.parent_paper_title == paper_data.paper_title
            ]

            total_citations = len(paper_citations)
            resolved_citations = len(
                [c for c in paper_citations if c.cited_paper_title]
            )
            arxiv_citations = len(
                [
                    c
                    for c in paper_citations
                    if c.cited_paper_arxiv_link
                    and "arxiv" in c.cited_paper_arxiv_link.lower()
                ]
            )

            stats_data.append(
                {
                    "paper_title": paper_data.paper_title,
                    "arxiv_link": paper_data.arxiv_link,
                    "total_citations": total_citations,
                    "resolved_citations": resolved_citations,
                    "resolution_rate": resolved_citations / total_citations
                    if total_citations > 0
                    else 0.0,
                    "arxiv_citations": arxiv_citations,
                    "arxiv_citation_rate": arxiv_citations / total_citations
                    if total_citations > 0
                    else 0.0,
                }
            )

        dataframes["citation_stats"] = pd.DataFrame(stats_data)
        logger.info(
            f"Generated citation statistics dataframe: {len(dataframes['citation_stats'])} rows"
        )

        return dataframes

    def _generate_dataframes_for_papers_only(self, papers) -> Dict[str, pd.DataFrame]:
        """Generate dataframes when only paper metadata is available (no content extraction)."""
        dataframes = {}

        # Papers DataFrame
        papers_data = []
        for paper in papers:
            papers_data.append(
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "authors": ", ".join(paper.authors),
                    "abstract": paper.abstract,
                    "categories": ", ".join(paper.categories),
                    "published_date": paper.published_date.isoformat()
                    if paper.published_date
                    else None,
                    "updated_date": paper.updated_date.isoformat()
                    if paper.updated_date
                    else None,
                    "abs_url": paper.abs_url,
                }
            )

        dataframes["papers"] = pd.DataFrame(papers_data)
        logger.info(f"Generated papers dataframe: {len(dataframes['papers'])} rows")

        return dataframes

    def _save_dataframes(
        self, dataframes: Dict[str, pd.DataFrame], name: str | None = None
    ):
        """Save all dataframes to CSV files."""
        # Create a folder for each paper's data using its arxiv_id
        arxiv_id = dataframes["paper_content"]["arxiv_link"].iloc[0].split("/")[-1]
        if name:
            paper_dir = os.path.join(self.config.output_dir, name)
        else:
            paper_dir = os.path.join(self.config.output_dir, arxiv_id)
        os.makedirs(paper_dir, exist_ok=True)
        logger.info(f"Created directory {paper_dir}")

        for name, df in dataframes.items():
            if name == "papers" and not self.config.save_raw_papers:
                continue
            if name == "paper_content" and not self.config.save_extracted_sections:
                continue
            if (
                name in ["citations", "citation_stats"]
                and not self.config.save_citations
            ):
                continue

            # filename = f"{name}_{timestamp}.csv"
            filename = f"{name}.csv"
            filepath = os.path.join(paper_dir, filename)
            df.to_csv(filepath, index=False, encoding="utf-8", errors="ignore")
            logger.info(f"Saved {name} dataframe to {filepath}")

    def print_summary(self, dataframes: Dict[str, pd.DataFrame]):
        """Print a summary of the pipeline results."""
        print("\n" + "=" * 60)
        print("📊 PIPELINE SUMMARY")
        print("=" * 60)

        if "papers" in dataframes:
            print(f"📄 Papers collected: {len(dataframes['papers'])}")

        if "paper_content" in dataframes:
            print(f"📚 Papers with related works: {len(dataframes['paper_content'])}")
            avg_length = dataframes["paper_content"]["related_works_length"].mean()
            print(f"📏 Average related works length: {avg_length:.0f} characters")

        if "citations" in dataframes:
            total_citations = len(dataframes["citations"])
            resolved_citations = dataframes["citations"]["has_metadata"].sum()
            arxiv_citations = dataframes["citations"]["is_arxiv_paper"].sum()
            reference_title_extraction_rate = dataframes["citations"][
                (dataframes["citations"]["cited_paper_title"].notna())
                & (dataframes["citations"]["cited_paper_title"] != "")
            ].shape[0] / len(dataframes["citations"])

            print(f"🔗 Total citations extracted: {total_citations}")
            print(
                f"✅ Citations with metadata: {resolved_citations} ({resolved_citations / total_citations * 100:.1f}%)"
            )
            print(
                f"📄 ArXiv citations: {arxiv_citations} ({arxiv_citations / total_citations * 100:.1f}%)"
            )
            print(
                f"📄 Reference title extraction rate: {reference_title_extraction_rate * 100:.1f}%"
            )
        if "citation_stats" in dataframes:
            avg_citations_per_paper = dataframes["citation_stats"][
                "total_citations"
            ].mean()
            avg_resolution_rate = dataframes["citation_stats"]["resolution_rate"].mean()
            print(f"📈 Average citations per paper: {avg_citations_per_paper:.1f}")
            print(f"🎯 Average resolution rate: {avg_resolution_rate * 100:.1f}%")

        print("=" * 60)


async def main():
    """Main entry point for the pipeline."""
    args, config = parse_args()
    pipeline = DataPipeline(config)
    logger.info(f"🔍 Arguments: {args}")
    dataframes = await pipeline.run_full_pipeline(
        arxiv_id=args.paper_id, continue_from_failed_test=args.paper_id is not None
    )

    if dataframes:
        pipeline.print_summary(dataframes)
    else:
        print("❌ Pipeline failed to generate any dataframes.")


if __name__ == "__main__":
    asyncio.run(main())
