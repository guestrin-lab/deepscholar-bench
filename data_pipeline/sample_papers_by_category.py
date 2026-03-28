"""
Sample papers from filtered dataset based on category distribution.

This script randomly selects a subset of papers (default: 100) from the filtered
dataset, with selection probability proportional to the category distribution
(stratified random sampling).

Usage:
    python sample_papers_by_category.py --input-folder outputs/20251015_143311/ \
                                         --sample-size 100 \
                                         --output-suffix "_sample100"
"""

import argparse
import logging
import os
import pandas as pd
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PaperSampler:
    """Sample papers based on category distribution."""
    
    def __init__(
        self,
        input_folder: str,
        sample_size: int = 100,
        random_seed: Optional[int] = 42,
        use_filtered: bool = True
    ):
        """
        Initialize the paper sampler.
        
        Args:
            input_folder: Path to folder containing paper CSV files
            sample_size: Number of papers to sample
            random_seed: Random seed for reproducibility (None for random)
            use_filtered: Use filtered files if available, otherwise use originals
        """
        self.input_folder = Path(input_folder)
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Determine which files to use
        if use_filtered and (self.input_folder / "paper_content_filtered.csv").exists():
            self.paper_content_path = self.input_folder / "paper_content_filtered.csv"
            self.papers_path = self.input_folder / "papers_filtered.csv"
            logger.info("Using filtered files for sampling")
        else:
            self.paper_content_path = self.input_folder / "paper_content.csv"
            self.papers_path = self.input_folder / "papers.csv"
            logger.info("Using original (non-filtered) files for sampling")
        
        # Validate paths
        if not self.papers_path.exists():
            raise FileNotFoundError(f"Papers file not found at {self.papers_path}")
    
    def extract_cs_categories(self, categories_str: str) -> List[str]:
        """
        Extract CS categories from category string.
        
        Args:
            categories_str: Semicolon or comma separated category string
            
        Returns:
            List of cs.XX categories
        """
        if pd.isna(categories_str):
            return []
        
        # Split by semicolon or comma
        cats = re.split('[;,]', str(categories_str).strip())
        cats = [cat.strip() for cat in cats if cat.strip()]
        
        # Extract cs.XX categories
        cs_cats = []
        for cat in cats:
            match = re.match(r'(cs\.[A-Z]{2})', cat)
            if match:
                cs_cats.append(match.group(1))
        
        return cs_cats
    
    def get_primary_category(self, categories_str: str) -> Optional[str]:
        """
        Get the primary (first) CS category from category string.
        
        Args:
            categories_str: Category string
            
        Returns:
            Primary cs.XX category or None
        """
        cs_cats = self.extract_cs_categories(categories_str)
        return cs_cats[0] if cs_cats else None
    
    def analyze_category_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Analyze the distribution of primary categories.
        
        Args:
            df: DataFrame with papers and categories
            
        Returns:
            Dictionary mapping category to count
        """
        category_counts = Counter()
        
        for categories_str in df['categories']:
            primary_cat = self.get_primary_category(categories_str)
            if primary_cat:
                category_counts[primary_cat] += 1
        
        return dict(category_counts)
    
    def stratified_sample(
        self,
        df: pd.DataFrame,
        category_dist: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Perform stratified random sampling based on category distribution.
        
        Args:
            df: DataFrame with papers
            category_dist: Category distribution (category -> count)
            
        Returns:
            Sampled DataFrame
        """
        total_papers = len(df)
        
        # Calculate target samples per category
        target_samples = {}
        for cat, count in category_dist.items():
            proportion = count / total_papers
            target_samples[cat] = int(round(proportion * self.sample_size))
        
        # Adjust to ensure we get exactly sample_size papers
        current_total = sum(target_samples.values())
        if current_total < self.sample_size:
            # Add to largest category
            largest_cat = max(category_dist.keys(), key=lambda k: category_dist[k])
            target_samples[largest_cat] += (self.sample_size - current_total)
        elif current_total > self.sample_size:
            # Remove from largest category
            largest_cat = max(category_dist.keys(), key=lambda k: category_dist[k])
            target_samples[largest_cat] -= (current_total - self.sample_size)
        
        logger.info("\nTarget samples per category:")
        logger.info("-" * 60)
        for cat in sorted(target_samples.keys(), key=lambda k: target_samples[k], reverse=True):
            percentage = (target_samples[cat] / self.sample_size) * 100
            logger.info(f"  {cat:<12} {target_samples[cat]:>3} papers ({percentage:>5.1f}%)")
        logger.info("-" * 60)
        
        # Sample papers from each category
        sampled_dfs = []
        
        for cat, target_count in target_samples.items():
            if target_count <= 0:
                continue
            
            # Get papers in this category
            cat_mask = df['categories'].apply(
                lambda x: self.get_primary_category(x) == cat
            )
            cat_papers = df[cat_mask]
            
            # Sample from this category
            if len(cat_papers) <= target_count:
                # Take all papers in this category
                sampled = cat_papers
                logger.warning(
                    f"Category {cat}: Only {len(cat_papers)} papers available, "
                    f"requested {target_count}"
                )
            else:
                # Random sample
                sampled = cat_papers.sample(n=target_count, random_state=self.random_seed)
            
            sampled_dfs.append(sampled)
        
        # Combine all samples
        result = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the final result
        result = result.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        return result
    
    def sample_papers(self, output_suffix: str = "_sample100") -> pd.DataFrame:
        """
        Sample papers and save results.
        
        Args:
            output_suffix: Suffix for output filenames
            
        Returns:
            Sampled DataFrame
        """
        logger.info(f"Reading papers from {self.papers_path}")
        df = pd.read_csv(self.papers_path, index_col=0)
        
        total_papers = len(df)
        logger.info(f"Total papers available: {total_papers}")
        
        if total_papers <= self.sample_size:
            logger.warning(
                f"Sample size ({self.sample_size}) >= total papers ({total_papers}). "
                f"Returning all papers."
            )
            return df
        
        # Analyze category distribution
        logger.info("\nAnalyzing category distribution...")
        category_dist = self.analyze_category_distribution(df)
        
        logger.info(f"\nFound {len(category_dist)} primary categories")
        logger.info("\nOriginal distribution:")
        logger.info("-" * 60)
        for cat in sorted(category_dist.keys(), key=lambda k: category_dist[k], reverse=True):
            percentage = (category_dist[cat] / total_papers) * 100
            logger.info(f"  {cat:<12} {category_dist[cat]:>4} papers ({percentage:>5.2f}%)")
        logger.info("-" * 60)
        
        # Perform stratified sampling
        logger.info(f"\nPerforming stratified sampling for {self.sample_size} papers...")
        sampled_df = self.stratified_sample(df, category_dist)
        
        # Save sampled papers
        papers_output = self.input_folder / f"papers{output_suffix}.csv"
        sampled_df.to_csv(papers_output, index=True)
        logger.info(f"\nSaved sampled papers to {papers_output}")
        
        # Also sample paper_content if it exists
        if self.paper_content_path.exists():
            logger.info(f"\nFiltering paper_content.csv based on sampled papers...")
            content_df = pd.read_csv(self.paper_content_path)
            
            # Get sampled arxiv_ids
            sampled_ids = set(sampled_df['arxiv_id'].values)
            
            # Filter content
            sampled_content = content_df[content_df['arxiv_id'].isin(sampled_ids)]
            
            content_output = self.input_folder / f"paper_content{output_suffix}.csv"
            sampled_content.to_csv(content_output, index=False)
            logger.info(f"Saved sampled paper content to {content_output}")
        
        # Print summary statistics
        logger.info("\n" + "=" * 60)
        logger.info("SAMPLING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Original papers: {total_papers}")
        logger.info(f"Sampled papers: {len(sampled_df)}")
        logger.info(f"Sampling rate: {len(sampled_df) / total_papers * 100:.2f}%")
        logger.info(f"Random seed: {self.random_seed}")
        
        # Verify distribution in sampled data
        sampled_dist = self.analyze_category_distribution(sampled_df)
        logger.info("\nActual sampled distribution:")
        logger.info("-" * 60)
        for cat in sorted(sampled_dist.keys(), key=lambda k: sampled_dist[k], reverse=True):
            percentage = (sampled_dist[cat] / len(sampled_df)) * 100
            logger.info(f"  {cat:<12} {sampled_dist[cat]:>3} papers ({percentage:>5.1f}%)")
        logger.info("=" * 60)
        
        return sampled_df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample papers based on category distribution"
    )
    
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Path to folder containing papers CSV files"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of papers to sample (default: 100)"
    )
    
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42, use -1 for random)"
    )
    
    parser.add_argument(
        "--use-filtered",
        action="store_true",
        default=True,
        help="Use filtered files if available (default: True)"
    )
    
    parser.add_argument(
        "--output-suffix",
        type=str,
        default=None,
        help="Suffix for output filenames (default: _sampleN where N is sample size)"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Handle random seed
    random_seed = None if args.random_seed == -1 else args.random_seed
    
    # Default output suffix
    if args.output_suffix is None:
        args.output_suffix = f"_sample{args.sample_size}"
    
    logger.info("Starting paper sampling...")
    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Sample size: {args.sample_size}")
    logger.info(f"Random seed: {random_seed}")
    
    # Create sampler
    sampler = PaperSampler(
        input_folder=args.input_folder,
        sample_size=args.sample_size,
        random_seed=random_seed,
        use_filtered=args.use_filtered
    )
    
    # Sample papers
    sampler.sample_papers(output_suffix=args.output_suffix)
    
    logger.info("\n✅ Sampling complete!")


if __name__ == "__main__":
    main()


