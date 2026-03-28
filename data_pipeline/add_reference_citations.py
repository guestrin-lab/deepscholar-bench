"""
Script to add reference citation counts to papers.csv.

This script:
1. Reads papers.csv
2. For each paper, finds its citation CSV file
3. Gets citation counts for all references using OpenAlex API
4. Aggregates the results (median citation count)
5. Adds new columns to papers.csv
"""

import argparse
import logging
import os
import pandas as pd
import time
import re
import numpy as np
from typing import Optional
from pathlib import Path
from tqdm import tqdm

# Import the citation count function
try:
    from eval.utils import get_citation_count_from_title
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from eval.utils import get_citation_count_from_title

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def clean_title_for_search(title: str) -> str:
    """Clean title for citation search (similar to document_importance.py)."""
    if not title or pd.isna(title):
        return ""
    # Remove punctuation and normalize (similar to document_importance.py)
    cleaned = re.sub(r"[^\w\s]", "", str(title).lower()).strip()
    return cleaned


def get_reference_citation_counts(
    citations_df: pd.DataFrame, 
    rate_limit_delay: float = 0.05
) -> dict:
    """
    Get citation counts for all references in a citation CSV file.
    
    Args:
        citations_df: DataFrame containing citations
        rate_limit_delay: Delay between API requests (seconds)
        
    Returns:
        Dictionary with:
        - citation_counts: List of citation counts found
        - citation_counts_with_na: List with "N/A" for not found
        - total_references: Total number of references
        - references_with_citations: Number of references with found citations
        - median_citation_count: Median citation count (or "N/A" if none found)
    """
    try:
        # Extract unique cited paper titles
        if 'cited_paper_title' not in citations_df.columns:
            logger.warning(f"No 'cited_paper_title' column in citations dataframe.")
            return {
                'citation_counts': [],
                'citation_counts_with_na': [],
                'total_references': 0,
                'references_with_citations': 0,
                'median_citation_count': 'N/A'
            }
        
        # Get unique titles (non-empty)
        cited_titles = citations_df['cited_paper_title'].dropna().unique()
        cited_titles = [str(title).strip() for title in cited_titles if str(title).strip()]
        
        if not cited_titles:
            return {
                'citation_counts': [],
                'citation_counts_with_na': [],
                'total_references': 0,
                'references_with_citations': 0,
                'median_citation_count': 'N/A'
            }
        
        # Get citation counts for each reference
        citation_counts = []
        citation_counts_with_na = []
        
        for i, title in tqdm(enumerate(cited_titles), total=len(cited_titles), desc="Getting citation counts"):
            if i > 0:
                time.sleep(rate_limit_delay)  # Rate limiting
            
            count = get_citation_count_from_title(title)
            
            if count is not None:
                citation_counts.append(count)
                citation_counts_with_na.append(count)
            else:
                citation_counts_with_na.append('N/A')
        
        # Calculate median
        median_citation_count = (
            np.median(citation_counts) if citation_counts else 'N/A'
        )
        
        return {
            'citation_counts': citation_counts,
            'citation_counts_with_na': citation_counts_with_na,
            'total_references': len(cited_titles),
            'references_with_citations': len(citation_counts),
            'median_citation_count': median_citation_count
        }
        
    except Exception as e:
        logger.error(f"Error processing citations dataframe: {e}")
        return {
            'citation_counts': [],
            'citation_counts_with_na': [],
            'total_references': 0,
            'references_with_citations': 0,
            'median_citation_count': 'N/A'
        }


def process_papers(
    papers_csv_path: str,
    citations_path: str,
    output_csv_path: Optional[str] = None,
    rate_limit_delay: float = 0.05
) -> pd.DataFrame:
    """
    Process papers and add reference citation counts.
    
    Args:
        papers_csv_path: Path to papers.csv
        citations_path: Path to citations.csv file
        output_csv_path: Optional output path (if None, overwrites input)
        rate_limit_delay: Delay between API requests (seconds)
        
    Returns:
        DataFrame with added citation columns
    """
    # Read papers CSV
    logger.info(f"Reading papers from: {papers_csv_path}")
    papers_df = pd.read_csv(papers_csv_path)
    citations_df = pd.read_csv(citations_path)
    logger.info(f"Found {len(papers_df)} papers")
    
    # Initialize new columns
    papers_df['reference_citation_counts'] = None
    papers_df['reference_citation_counts_list'] = None
    papers_df['total_references'] = 0
    papers_df['references_with_citations'] = 0
    papers_df['median_reference_citation_count'] = None
    
    # Process each paper
    for idx, row in papers_df.iterrows():
        # Handle both 'arxiv_id' and 'paper_id' column names
        arxiv_id = row.get('arxiv_id', row.get('paper_id', ''))
        
        if not arxiv_id or pd.isna(arxiv_id):
            logger.warning(f"Row {idx}: No arxiv_id found, skipping")
            continue
        
        # Clean arxiv_id (remove any extra spaces or version info)
        arxiv_id = str(arxiv_id).strip()
        
        # Find citation CSV file
        if (idx + 1) % 10 == 0:
            logger.info(f"Processing paper {idx + 1}/{len(papers_df)}: {arxiv_id}")
        
        # Get citation counts for references
        citation_data = get_reference_citation_counts(
            citations_df[citations_df["parent_paper_arxiv_id"] == arxiv_id], 
            rate_limit_delay=rate_limit_delay
        )
        
        # Update dataframe
        papers_df.at[idx, 'reference_citation_counts'] = (
            ','.join(map(str, citation_data['citation_counts']))
            if citation_data['citation_counts'] else ''
        )
        papers_df.at[idx, 'reference_citation_counts_list'] = (
            ','.join(map(str, citation_data['citation_counts_with_na']))
            if citation_data['citation_counts_with_na'] else ''
        )
        papers_df.at[idx, 'total_references'] = citation_data['total_references']
        papers_df.at[idx, 'references_with_citations'] = citation_data['references_with_citations']
        papers_df.at[idx, 'median_reference_citation_count'] = citation_data['median_citation_count']
    
    # Save results
    output_path = output_csv_path or papers_csv_path
    logger.info(f"Saving results to: {output_path}")
    papers_df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved {len(papers_df)} papers with citation data")
    
    # Print summary statistics
    total_refs = papers_df['total_references'].sum()
    total_with_citations = papers_df['references_with_citations'].sum()
    papers_with_refs = (papers_df['total_references'] > 0).sum()
    
    logger.info("=" * 60)
    logger.info("Summary Statistics:")
    logger.info(f"  Total papers processed: {len(papers_df)}")
    logger.info(f"  Papers with references: {papers_with_refs}")
    logger.info(f"  Total references: {total_refs}")
    logger.info(f"  References with citation counts: {total_with_citations}")
    if total_refs > 0:
        logger.info(f"  Success rate: {total_with_citations/total_refs*100:.1f}%")
    logger.info("=" * 60)
    
    return papers_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Add reference citation counts to papers.csv"
    )
    
    parser.add_argument(
        "--papers-csv",
        type=str,
        required=True,
        help="Path to paper_content.csv file"
    )
    
    parser.add_argument(
        "--citations-path",
        type=str,
        required=True,
        help="Path to citations.csv file"
    )
    
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Output CSV path (if not provided, overwrites input)"
    )
    
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.05,
        help="Delay between API requests in seconds (default: 0.05)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate paths
    if not os.path.exists(args.papers_csv):
        logger.error(f"Papers CSV not found: {args.papers_csv}")
        return
    
    if not os.path.exists(args.citations_path):
        logger.error(f"Citations path not found: {args.citations_path}")
        return
    
    # Process papers
    process_papers(
        papers_csv_path=args.papers_csv,
        citations_path=args.citations_path,
        output_csv_path=args.output_csv,
        rate_limit_delay=args.rate_limit_delay
    )
    
    logger.info("✅ Done!")


if __name__ == "__main__":
    main()

