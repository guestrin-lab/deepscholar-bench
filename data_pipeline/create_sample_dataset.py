"""
Create a complete sample dataset with all related files.

This script samples papers based on category distribution and creates a complete
output folder containing all related data (papers, citations, related works, etc.)
for the sampled papers.

Usage:
    python create_sample_dataset.py --input-base outputs/ \
                                     --input-folder 20251015_143311 \
                                     --output-name outputs_oct \
                                     --sample-size 100
"""

import argparse
import logging
import os
import pandas as pd
import re
import shutil
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Set
import random

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetSampler:
    """Create a complete sampled dataset with all related files."""
    
    def __init__(
        self,
        input_base: str,
        input_folder: str,
        output_name: str,
        sample_size: int = 100,
        random_seed: Optional[int] = 42,
        use_filtered: bool = True
    ):
        """
        Initialize the dataset sampler.
        
        Args:
            input_base: Base path containing outputs folder (e.g., "outputs/")
            input_folder: Folder name with timestamp (e.g., "20251015_143311")
            output_name: Name for output dataset (e.g., "outputs_oct")
            sample_size: Number of papers to sample
            random_seed: Random seed for reproducibility
            use_filtered: Use filtered files if available
        """
        self.input_base = Path(input_base)
        self.input_folder_name = input_folder
        self.input_folder = self.input_base / input_folder
        self.output_folder = Path(output_name)
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.use_filtered = use_filtered
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Determine which files to use for sampling
        if use_filtered and (self.input_folder / "papers_filtered.csv").exists():
            self.papers_file = "papers_filtered.csv"
            self.content_file = "paper_content_filtered.csv"
            logger.info("Using filtered files for sampling")
        else:
            self.papers_file = "papers.csv"
            self.content_file = "paper_content.csv"
            logger.info("Using original files for sampling")
        
        # Validate paths
        papers_path = self.input_folder / self.papers_file
        if not papers_path.exists():
            raise FileNotFoundError(f"Papers file not found at {papers_path}")
    
    def extract_cs_categories(self, categories_str: str) -> List[str]:
        """Extract CS categories from category string."""
        if pd.isna(categories_str):
            return []
        
        cats = re.split('[;,]', str(categories_str).strip())
        cats = [cat.strip() for cat in cats if cat.strip()]
        
        cs_cats = []
        for cat in cats:
            match = re.match(r'(cs\.[A-Z]{2})', cat)
            if match:
                cs_cats.append(match.group(1))
        
        return cs_cats
    
    def get_primary_category(self, categories_str: str) -> Optional[str]:
        """Get the primary (first) CS category."""
        cs_cats = self.extract_cs_categories(categories_str)
        return cs_cats[0] if cs_cats else None
    
    def analyze_category_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """Analyze the distribution of primary categories."""
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
        """Perform stratified random sampling based on category distribution."""
        total_papers = len(df)
        
        # Calculate target samples per category
        target_samples = {}
        for cat, count in category_dist.items():
            proportion = count / total_papers
            target_samples[cat] = int(round(proportion * self.sample_size))
        
        # Adjust to ensure we get exactly sample_size papers
        current_total = sum(target_samples.values())
        if current_total < self.sample_size:
            largest_cat = max(category_dist.keys(), key=lambda k: category_dist[k])
            target_samples[largest_cat] += (self.sample_size - current_total)
        elif current_total > self.sample_size:
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
            
            cat_mask = df['categories'].apply(
                lambda x: self.get_primary_category(x) == cat
            )
            cat_papers = df[cat_mask]
            
            if len(cat_papers) <= target_count:
                sampled = cat_papers
                logger.warning(
                    f"Category {cat}: Only {len(cat_papers)} papers available, "
                    f"requested {target_count}"
                )
            else:
                sampled = cat_papers.sample(n=target_count, random_state=self.random_seed)
            
            sampled_dfs.append(sampled)
        
        # Combine and shuffle
        result = pd.concat(sampled_dfs, ignore_index=True)
        result = result.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        return result
    
    def copy_related_files(self, sampled_ids: Set[str]):
        """
        Copy all related files for sampled papers.
        
        Args:
            sampled_ids: Set of arxiv_ids that were sampled
        """
        # Create output directories
        citations_out = self.output_folder / "citations"
        related_works_out = self.output_folder / "related_works"
        latex_out = self.output_folder / "latex_source"
        
        citations_out.mkdir(parents=True, exist_ok=True)
        related_works_out.mkdir(parents=True, exist_ok=True)
        latex_out.mkdir(parents=True, exist_ok=True)
        
        # Copy citation files
        citations_in = self.input_base / "citations"
        if citations_in.exists():
            logger.info("\nCopying citation files...")
            copied_citations = 0
            for arxiv_id in sampled_ids:
                citation_file = citations_in / f"{arxiv_id}.csv"
                if citation_file.exists():
                    shutil.copy2(citation_file, citations_out / f"{arxiv_id}.csv")
                    copied_citations += 1
            logger.info(f"Copied {copied_citations} citation files")
        
        # Copy related works files
        related_works_in = self.input_base / "related_works"
        if related_works_in.exists():
            logger.info("Copying related works files...")
            copied_rw = 0
            for arxiv_id in sampled_ids:
                rw_file = related_works_in / f"{arxiv_id}.csv"
                if rw_file.exists():
                    shutil.copy2(rw_file, related_works_out / f"{arxiv_id}.csv")
                    copied_rw += 1
            logger.info(f"Copied {copied_rw} related works files")
        
        # Copy LaTeX source files and folders
        latex_in = self.input_base / "latex_source"
        if latex_in.exists():
            logger.info("Copying LaTeX source files...")
            copied_latex = 0
            for arxiv_id in sampled_ids:
                # Copy tar.gz if exists
                latex_archive = latex_in / f"{arxiv_id}.tar.gz"
                if latex_archive.exists():
                    shutil.copy2(latex_archive, latex_out / f"{arxiv_id}.tar.gz")
                    copied_latex += 1
                
                # Copy extracted folder if exists
                latex_folder = latex_in / arxiv_id
                if latex_folder.exists() and latex_folder.is_dir():
                    shutil.copytree(
                        latex_folder,
                        latex_out / arxiv_id,
                        dirs_exist_ok=True
                    )
            logger.info(f"Copied {copied_latex} LaTeX source packages")
    
    def create_sample_dataset(self):
        """Create a complete sampled dataset."""
        logger.info("=" * 70)
        logger.info("CREATING SAMPLED DATASET")
        logger.info("=" * 70)
        logger.info(f"Input: {self.input_folder}")
        logger.info(f"Output: {self.output_folder}")
        logger.info(f"Sample size: {self.sample_size}")
        logger.info(f"Random seed: {self.random_seed}")
        
        # Read papers
        papers_path = self.input_folder / self.papers_file
        logger.info(f"\nReading papers from {papers_path}")
        df = pd.read_csv(papers_path, index_col=0)
        
        total_papers = len(df)
        logger.info(f"Total papers available: {total_papers}")
        
        if total_papers <= self.sample_size:
            logger.warning(
                f"Sample size ({self.sample_size}) >= total papers ({total_papers}). "
                f"Using all papers."
            )
            sampled_df = df
        else:
            # Analyze and sample
            category_dist = self.analyze_category_distribution(df)
            logger.info(f"\nFound {len(category_dist)} primary categories")
            
            logger.info("\nOriginal distribution:")
            logger.info("-" * 60)
            for cat in sorted(category_dist.keys(), key=lambda k: category_dist[k], reverse=True)[:10]:
                percentage = (category_dist[cat] / total_papers) * 100
                logger.info(f"  {cat:<12} {category_dist[cat]:>4} papers ({percentage:>5.2f}%)")
            logger.info("-" * 60)
            
            # Perform sampling
            logger.info(f"\nPerforming stratified sampling...")
            sampled_df = self.stratified_sample(df, category_dist)
        
        # Create output folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Save sampled papers.csv
        logger.info(f"\nSaving papers.csv...")
        sampled_df.to_csv(self.output_folder / "papers.csv", index=True)
        
        # Get sampled IDs
        sampled_ids = set(sampled_df['arxiv_id'].values)
        
        # Filter and save paper_content.csv
        content_path = self.input_folder / self.content_file
        if content_path.exists():
            logger.info(f"Filtering paper_content.csv...")
            content_df = pd.read_csv(content_path)
            sampled_content = content_df[content_df['arxiv_id'].isin(sampled_ids)]
            sampled_content.to_csv(self.output_folder / "paper_content.csv", index=False)
            logger.info(f"Saved {len(sampled_content)} papers to paper_content.csv")
        
        # Copy all related files
        self.copy_related_files(sampled_ids)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("SAMPLING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Original papers: {total_papers}")
        logger.info(f"Sampled papers: {len(sampled_df)}")
        logger.info(f"Output folder: {self.output_folder}")
        logger.info("\nOutput structure:")
        logger.info(f"  {self.output_folder}/")
        logger.info(f"    ├── papers.csv ({len(sampled_df)} papers)")
        
        if content_path.exists():
            logger.info(f"    ├── paper_content.csv ({len(sampled_content)} papers)")
        
        citations_count = len(list((self.output_folder / "citations").glob("*.csv"))) if (self.output_folder / "citations").exists() else 0
        if citations_count > 0:
            logger.info(f"    ├── citations/ ({citations_count} files)")
        
        rw_count = len(list((self.output_folder / "related_works").glob("*.csv"))) if (self.output_folder / "related_works").exists() else 0
        if rw_count > 0:
            logger.info(f"    ├── related_works/ ({rw_count} files)")
        
        latex_count = len(list((self.output_folder / "latex_source").glob("*.tar.gz"))) if (self.output_folder / "latex_source").exists() else 0
        if latex_count > 0:
            logger.info(f"    └── latex_source/ ({latex_count} packages)")
        
        # Verify distribution
        sampled_dist = self.analyze_category_distribution(sampled_df)
        logger.info("\nSampled distribution (top 10):")
        logger.info("-" * 60)
        for cat in sorted(sampled_dist.keys(), key=lambda k: sampled_dist[k], reverse=True)[:10]:
            percentage = (sampled_dist[cat] / len(sampled_df)) * 100
            logger.info(f"  {cat:<12} {sampled_dist[cat]:>3} papers ({percentage:>5.1f}%)")
        logger.info("=" * 70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a complete sample dataset with all related files"
    )
    
    parser.add_argument(
        "--input-base",
        type=str,
        default="outputs/",
        help="Base path containing outputs folder (default: outputs/)"
    )
    
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Input folder name with timestamp (e.g., 20251015_143311)"
    )
    
    parser.add_argument(
        "--output-name",
        type=str,
        default="outputs_oct",
        help="Name for output dataset folder (default: outputs_oct)"
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
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Handle random seed
    random_seed = None if args.random_seed == -1 else args.random_seed
    
    # Create sampler
    sampler = DatasetSampler(
        input_base=args.input_base,
        input_folder=args.input_folder,
        output_name=args.output_name,
        sample_size=args.sample_size,
        random_seed=random_seed,
        use_filtered=args.use_filtered
    )
    
    # Create sample dataset
    sampler.create_sample_dataset()
    
    logger.info("\n✅ Sample dataset created successfully!")


if __name__ == "__main__":
    main()


