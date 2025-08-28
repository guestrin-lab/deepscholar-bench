"""
Test script for single paper processing functionality.

This script demonstrates how to process a single ArXiv paper through the complete pipeline.
"""

import asyncio
import logging
import sys
from pathlib import Path
import os
from datetime import datetime, timedelta

sys.path.append(os.getcwd())
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from data_pipeline.config import PipelineConfig
from data_pipeline.main import DataPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

arxiv_id = "2504.12900"


async def test_single_paper():
    """Test processing a single paper by ArXiv ID."""
    print(f"🧪 Testing single paper processing for ArXiv ID: {arxiv_id}")
    print("=" * 60)

    # Create configuration (minimal settings for single paper)
    config = PipelineConfig(
        start_date=datetime.now() - timedelta(days=365),  # Wide date range
        end_date=datetime.now(),
        arxiv_categories=["cs.AI", "cs.CL", "cs.LG"],
        min_author_hindex=0,  # No h-index filtering for test
        max_author_hindex=None,
        max_papers_per_category=1,
        min_citations_in_related_works=1,  # Low threshold for test
        output_dir=str(Path(__file__).parent / "test_outputs"),
        save_raw_papers=True,
        save_extracted_sections=True,
        save_citations=True,
        concurrent_requests=3,
        request_delay=1.0,
    )

    # Create and run pipeline
    pipeline = DataPipeline(config)

    try:
        dataframes = await pipeline.run_full_pipeline(
            arxiv_id=arxiv_id, continue_from_failed_test=True
        )

        if dataframes:
            print("\n🎉 SUCCESS! Generated dataframes:")
            for name, df in dataframes.items():
                print(f"  📊 {name}: {len(df)} rows")

            # Print detailed summary
            pipeline.print_summary(dataframes)

            # Show sample data
            if "citations" in dataframes and len(dataframes["citations"]) > 0:
                print("\n📋 Sample citations:")
                sample_citations = dataframes["citations"].head(3)
                for idx, row in sample_citations.iterrows():
                    title = row["cited_paper_title"]
                    title_display = (
                        title[:80] + "..."
                        if title and len(title) > 80
                        else (title or "No title found")
                    )
                    print(f"  {idx + 1}. {row['citation_shorthand']}: {title_display}")
        else:
            print("❌ No dataframes generated")

    except Exception as e:
        logger.error(f"Error during single paper processing: {e}")
        raise


async def main():
    """Main test function."""
    await test_single_paper()


if __name__ == "__main__":
    asyncio.run(main())
