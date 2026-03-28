"""
Filter papers based on related works quality and citation count.

This script processes pipeline outputs and filters papers to ensure:
1. Papers have a related works section
2. Related works section is not too short or too long
3. Papers have at least a minimum number of citations

Usage:
    python filter_quality_papers.py --input-folder outputs/20251015_143311/ \
                                     --min-citations 5 \
                                     --min-rw-length 200 \
                                     --max-rw-length 10000
"""

import argparse
import logging
import os
import pandas as pd
from pathlib import Path
from typing import Optional
import shutil
import asyncio
import tqdm
try:
    from author_filter import AuthorFilter
    from config import PipelineConfig
except ImportError:
    from .author_filter import AuthorFilter
    from .config import PipelineConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PaperQualityFilter:
    """Filter papers based on quality criteria."""
    
    def __init__(
        self, 
        input_folder: str,
        min_citations: int = 5,
        min_rw_length: int = 200,
        max_rw_length: int = 10000,
        citations_path: Optional[str] = None,
        related_works_path: Optional[str] = None,
        min_hindex: int = 20,
        max_hindex: Optional[int] = None,
        request_delay: float = 1.0
    ):
        """
        Initialize the quality filter.
        
        Args:
            input_folder: Path to the folder containing paper_content.csv
            min_citations: Minimum number of citations required
            min_rw_length: Minimum length of related works section (characters)
            max_rw_length: Maximum length of related works section (characters)
            citations_path: Path to citation file (default: citations.csv relative to input folder)
            related_works_path: Path to related works file (default: related_works_combined.csv relative to input folder)
        """
        self.input_folder = Path(input_folder)
        self.min_citations = min_citations
        self.min_rw_length = min_rw_length
        self.max_rw_length = max_rw_length
        
        # Set up paths
        self.paper_content_path = self.input_folder / "paper_content.csv"
        self.papers_path = self.input_folder / "papers.csv"
        
        # Use provided paths or default to parent folders
        if citations_path:
            self.citations_path = Path(citations_path)
        else:
            self.citations_path = self.input_folder / "citations.csv"
        if related_works_path:
            self.related_works_path = Path(related_works_path)
        else:
            self.related_works_path = self.input_folder / "related_works_combined.csv"

        self.citations_df = pd.read_csv(self.citations_path)
        self.related_works_df = pd.read_csv(self.related_works_path)
        # Validate paths
        if not self.paper_content_path.exists():
            raise FileNotFoundError(f"paper_content.csv not found at {self.paper_content_path}")
    
        self.author_filter = AuthorFilter(config=PipelineConfig(
            min_author_hindex=min_hindex,
            max_author_hindex=max_hindex,
            request_delay=request_delay
        ))
        

    def count_citations(self, arxiv_id: str) -> int:
        """
        Count citations for a paper by reading its citation CSV.
        
        Args:
            arxiv_id: ArXiv ID (e.g., "2509.11056v1")
            
        Returns:
            Number of citations found
        """
        citations = self.citations_df[self.citations_df["parent_paper_arxiv_id"] == arxiv_id]
        citations = citations.dropna(subset=['cited_paper_title'])
        return len(citations)
    
    def validate_related_works(self, row: pd.Series) -> bool:
        """
        Check if related works section meets quality criteria.
        
        Args:
            row: DataFrame row with paper data
            
        Returns:
            True if related works section is valid
        """
        # Check if related works section exists
        if pd.isna(row.get("related_works_section")) or not row.get("related_works_section"):
            logger.debug(f"Paper {row.get('arxiv_id')} has no related works section")
            return False
        
        # Check related works length
        rw_length = row.get("related_works_length", 0)
        
        # If length column doesn't exist or is NaN, calculate from text
        if pd.isna(rw_length):
            rw_text = str(row.get("related_works_section", ""))
            rw_length = len(rw_text)
        
        if rw_length < self.min_rw_length:
            logger.debug(
                f"Paper {row.get('arxiv_id')} related works too short: {rw_length} chars"
            )
            return False
        
        if rw_length > self.max_rw_length:
            logger.debug(
                f"Paper {row.get('arxiv_id')} related works too long: {rw_length} chars"
            )
            return False
        
        return True
    
    async def filter_papers(self) -> pd.DataFrame:
        """
        Filter papers based on all quality criteria.
        
        Returns:
            Filtered DataFrame with quality papers
        """
        logger.info(f"Reading papers from {self.paper_content_path}")
        df = pd.read_csv(self.paper_content_path)
        papers_df = pd.read_csv(self.papers_path)
        
        initial_count = len(df)
        logger.info(f"Total papers in input: {initial_count}")
        
        # Track filtering statistics
        stats = {
            "no_related_works": 0,
            "rw_too_short": 0,
            "rw_too_long": 0,
            "insufficient_citations": 0,
            "passed": 0,
            "failed_author_filter": 0
        }
        
        # Lists to store filtered data
        filtered_rows = []
        
        pbar = tqdm.tqdm(total=len(df), desc="Filtering papers")
        for idx, row in df.iterrows():
            pbar.update(1)
            pbar.set_postfix({"passed": stats["passed"]})
            arxiv_id = row.get("arxiv_id")
            
            # Check related works section
            if pd.isna(row.get("related_works_section")) or not row.get("related_works_section"):
                stats["no_related_works"] += 1
                continue
            
            # Check related works length
            rw_length = row.get("related_works_length", 0)
            if pd.isna(rw_length):
                rw_text = str(row.get("related_works_section", ""))
                rw_length = len(rw_text)
            
            if rw_length < self.min_rw_length:
                stats["rw_too_short"] += 1
                continue
            
            if rw_length > self.max_rw_length:
                stats["rw_too_long"] += 1
                continue
            
            # Check citation count
            citation_count = self.count_citations(arxiv_id)
            if citation_count < self.min_citations:
                stats["insufficient_citations"] += 1
                logger.debug(
                    f"Paper {arxiv_id} has only {citation_count} citations "
                    f"(minimum: {self.min_citations})"
                )
                continue
            
            paper_authors = papers_df[papers_df["arxiv_id"] == arxiv_id].iloc[0].get("authors", "").split(", ")
            
            if self.author_filter.config.min_author_hindex > 0 or self.author_filter.config.max_author_hindex is not None:
                meets_author_criteria = await self.author_filter.paper_meets_hindex_criteria(authors=paper_authors)
                if not meets_author_criteria:
                    logger.debug(f"Paper {arxiv_id} failed author filter: {paper_authors}")
                    stats["failed_author_filter"] += 1
                    continue
            
            # Paper passed all filters
            stats["passed"] += 1
            filtered_rows.append(row)
            
            if stats["passed"] % 100 == 0:
                logger.info(f"Processed {idx + 1}/{initial_count} papers, {stats['passed']} passed")
        
        # Create filtered dataframe
        filtered_df = pd.DataFrame(filtered_rows)
        
        # Log statistics
        logger.info("\n" + "=" * 60)
        logger.info("FILTERING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total papers processed: {initial_count}")
        logger.info(f"Papers with no related works: {stats['no_related_works']}")
        logger.info(f"Papers with related works too short: {stats['rw_too_short']}")
        logger.info(f"Papers with related works too long: {stats['rw_too_long']}")
        logger.info(f"Papers with insufficient citations: {stats['insufficient_citations']}")
        logger.info(f"Papers that failed author filter: {stats['failed_author_filter']}")
        logger.info(f"Papers that passed filters: {stats['passed']}")
        logger.info(f"Pass rate: {stats['passed'] / initial_count * 100:.2f}%")
        logger.info("=" * 60)
        
        return filtered_df
    
    def save_filtered_papers(self, filtered_df: pd.DataFrame, output_folder: str = "filtered"):
        """
        Save filtered papers.
        
        Args:
            filtered_df: Filtered DataFrame
            output_folder: Folder to save filtered papers
        """
        output_folder = Path(output_folder)
        shutil.copytree(self.input_folder, output_folder, dirs_exist_ok=True)
        filtered_df.to_csv(output_folder / "paper_content.csv", index=False)
        
        citation_paths = ['citations.csv', 'important_citations.csv', 'recovered_citations.csv']
        paper_paths = ['paper_content_with_citations.csv', 'papers.csv', 'papers_with_related_works.csv', 'related_works_combined.csv']
        for csv_file_name in [*paper_paths, *citation_paths]:
            path = self.input_folder / csv_file_name
            if path.exists():
                df = pd.read_csv(path)
                if csv_file_name in paper_paths:
                    print(f"Filtering {csv_file_name} by arxiv_id")
                    df = df[df["arxiv_id"].isin(filtered_df["arxiv_id"].values)]
                elif csv_file_name in citation_paths:
                    print(f"Filtering {csv_file_name} by parent_paper_arxiv_id")
                    df = df[df["parent_paper_arxiv_id"].isin(filtered_df["arxiv_id"].values)]
                print(f"Saving {csv_file_name} to {output_folder / csv_file_name}, {len(df)} papers")
                df.to_csv(output_folder / csv_file_name, index=False)
    
    async def filter_and_save(self, output_folder: str = "filtered"):
        """
        Filter papers and save results.
        
        Args:
            output_suffix: Suffix to add to output filenames
        """
        # Filter papers
        filtered_df = await self.filter_papers()
        self.save_filtered_papers(filtered_df, output_folder)
        return filtered_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter papers based on related works quality and citation count"
    )
    
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to folder containing paper_content.csv (e.g., outputs/20251015_143311/)"
    )
    
    parser.add_argument(
        "--min-citations",
        type=int,
        default=5,
        help="Minimum number of citations required (default: 5)"
    )
    
    parser.add_argument(
        "--min-rw-length",
        type=int,
        default=200,
        help="Minimum length of related works section in characters (default: 200)"
    )
    
    parser.add_argument(
        "--max-rw-length",
        type=int,
        default=10000,
        help="Maximum length of related works section in characters (default: 10000)"
    )
    
    parser.add_argument(
        "--citations-path",
        type=str,
        default=None,
        help="Path to citations file (default: citations.csv relative to input folder)"
    )
    
    parser.add_argument(
        "--related-works-path",
        type=str,
        default=None,
        help="Path to related works file (default: related_works_combined.csv relative to input folder)"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Folder to save filtered papers (default: filtered)"
    )
    parser.add_argument(
        "--min-hindex",
        type=int,
        default=20,
        help="Minimum h-index for at least one author (default: 20)"
    )
    parser.add_argument(
        "--max-hindex",
        type=int,
        default=None,
        help="Maximum h-index for at least one author (default: None)"
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Delay between requests (seconds) (default: 1.0)"
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    logger.info("Starting paper quality filtering...")
    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Minimum citations: {args.min_citations}")
    logger.info(f"Related works length range: {args.min_rw_length} - {args.max_rw_length} chars")
    logger.info(f"Author h-index range: {args.min_hindex} - {args.max_hindex}")
    
    # Create filter instance
    filter_obj = PaperQualityFilter(
        input_folder=args.input_folder,
        min_citations=args.min_citations,
        min_rw_length=args.min_rw_length,
        max_rw_length=args.max_rw_length,
        citations_path=args.citations_path,
        related_works_path=args.related_works_path,
        min_hindex=args.min_hindex,
        max_hindex=args.max_hindex,
        request_delay=args.request_delay
    )
    
    # Filter and save
    output_folder = args.output_folder if args.output_folder else f"{args.input_folder}_filtered"
    asyncio.run(filter_obj.filter_and_save(output_folder=output_folder))
    
    logger.info("✅ Filtering complete!")


if __name__ == "__main__":
    main()



