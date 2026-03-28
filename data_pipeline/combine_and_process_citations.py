#!/usr/bin/env python3
"""
Script to combine individual citation CSV files, filter important citations,
and add important citations list to each paper.
"""
import argparse
import glob
import json
import os
import pandas as pd
import lotus
from lotus.models import LM

# Import from get_important_citations
from get_important_citations import get_important_citations, query_in


def combine_citation_files(citations_folder: str, output_file: str) -> pd.DataFrame:
    """Combine all individual citation CSV files into one."""
    files = glob.glob(os.path.join(citations_folder, "*.csv"))
    dfs = []
    
    for file in files:
        try:
            df = pd.read_csv(file)
            if len(df) > 0:  # Only add non-empty files
                dfs.append(df)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Skipping empty or invalid file: {file}")
            continue
    
    if not dfs:
        raise ValueError(f"No valid citation files found in {citations_folder}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_file, index=False)
    print(f"✅ Combined {len(dfs)} citation files with {len(combined)} total citations")
    print(f"   Saved to: {output_file}")
    return combined


def add_important_citations_to_papers(
    important_citations_df: pd.DataFrame,
    papers_df: pd.DataFrame,
    output_file: str
) -> pd.DataFrame:
    """Add important citations list to each paper."""
    # Group important citations by parent paper
    grouped = important_citations_df.groupby("parent_paper_title")
    
    # Create a mapping of paper title to list of important citations
    paper_to_citations = {}
    for title, group in grouped:
        citations_list = []
        for _, row in group.iterrows():
            citation = {
                "title": str(row.get("cited_paper_title", "")) if pd.notna(row.get("cited_paper_title")) else "",
                "authors": str(row.get("cited_paper_authors", "")) if pd.notna(row.get("cited_paper_authors")) else "",
                "arxiv_link": str(row.get("cited_paper_arxiv_link", "")) if pd.notna(row.get("cited_paper_arxiv_link")) else "",
                "abstract": str(row.get("cited_paper_abstract", "")) if pd.notna(row.get("cited_paper_abstract")) else "",
                "year": str(row.get("bib_paper_year", "")) if pd.notna(row.get("bib_paper_year")) else "",
            }
            citations_list.append(citation)
        paper_to_citations[title] = citations_list
    
    # Determine which title column to use
    title_col = "paper_title" if "paper_title" in papers_df.columns else "title"
    
    # Add important citations column to papers (as JSON string for CSV)
    papers_df["important_citations"] = papers_df[title_col].apply(
        lambda title: json.dumps(paper_to_citations.get(title, [])) if title in paper_to_citations else "[]"
    )
    
    # Also add count
    papers_df["num_important_citations"] = papers_df[title_col].apply(
        lambda title: len(paper_to_citations.get(title, []))
    )
    
    # Save updated papers
    papers_df.to_csv(output_file, index=False)
    print(f"✅ Added important citations to {len(papers_df)} papers")
    print(f"   Total papers with important citations: {(papers_df['num_important_citations'] > 0).sum()}")
    print(f"   Average important citations per paper: {papers_df['num_important_citations'].mean():.2f}")
    print(f"   Saved to: {output_file}")
    
    return papers_df


def main():
    parser = argparse.ArgumentParser(
        description="Combine citations, filter important ones, and add to papers"
    )
    parser.add_argument(
        "--citations_folder",
        type=str,
        help="Folder containing individual citation CSV files",
    )
    parser.add_argument(
        "--papers_csv",
        type=str,
        help="Path to papers CSV file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model to use",
    )
    parser.add_argument(
        "--skip_combine",
        action="store_true",
        help="Skip combining citations if already done",
    )
    parser.add_argument(
        "--skip_filter",
        action="store_true",
        help="Skip filtering if already done",
    )
    
    args = parser.parse_args()
    
    # Setup lotus
    lotus.settings.configure(lm=LM(model=args.model))
    
    # File paths
    combined_citations_file = os.path.join(args.output_dir, "all_citations.csv")
    important_citations_file = os.path.join(args.output_dir, "important_citations.csv")
    output_papers_file = os.path.join(args.output_dir, "paper_content_filtered_with_citations_and_important.csv")
    
    # Step 1: Combine citation files
    if not args.skip_combine and not os.path.exists(combined_citations_file):
        print("📝 Step 1: Combining citation files...")
        combine_citation_files(args.citations_folder, combined_citations_file)
    else:
        print(f"⏭️  Step 1: Skipping combine (file exists or --skip_combine): {combined_citations_file}")
        if not os.path.exists(combined_citations_file):
            raise FileNotFoundError(f"Combined citations file not found: {combined_citations_file}")
    
    # Step 2: Filter important citations
    if not args.skip_filter and not os.path.exists(important_citations_file):
        print("\n📝 Step 2: Filtering important citations...")
        citation_df = pd.read_csv(combined_citations_file)
        papers_df = pd.read_csv(args.papers_csv)
        
        # Column mapping is handled in get_important_citations function
        important_citations_df = get_important_citations(citation_df, papers_df)
        important_citations_df.to_csv(important_citations_file, index=False)
        print(f"✅ Saved important citations to: {important_citations_file}")
    else:
        print(f"⏭️  Step 2: Skipping filter (file exists or --skip_filter): {important_citations_file}")
        if not os.path.exists(important_citations_file):
            raise FileNotFoundError(f"Important citations file not found: {important_citations_file}")
        important_citations_df = pd.read_csv(important_citations_file)
    
    # Step 3: Add important citations to papers
    print("\n📝 Step 3: Adding important citations to papers...")
    papers_df = pd.read_csv(args.papers_csv)
    
    # Ensure we have the title column
    if "title" not in papers_df.columns and "paper_title" in papers_df.columns:
        papers_df["title"] = papers_df["paper_title"]
    
    updated_papers_df = add_important_citations_to_papers(
        important_citations_df, papers_df, output_papers_file
    )
    
    print("\n✅ All steps completed successfully!")
    print(f"   Output file: {output_papers_file}")


if __name__ == "__main__":
    main()

