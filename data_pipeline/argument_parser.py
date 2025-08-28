import argparse
from datetime import datetime, timedelta

try:
    from config import PipelineConfig
except ImportError:
    from .config import PipelineConfig


def parse_args():
    parser = argparse.ArgumentParser(description="ArXiv Data Collection Pipeline")

    # Single paper processing
    parser.add_argument(
        "--paper-id",
        type=str,
        help="Process a single paper by ArXiv ID (e.g., '2502.07374' or 'arxiv:2502.07374')",
        default=None,
    )

    parser.add_argument(
        "--existing-papers-csv",
        type=str,
        default=None,
        help="Use existing papers from a CSV file",
    )

    # Date range arguments
    parser.add_argument(
        "--start-date",
        type=str,
        default=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        help="Start date for paper search (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date for paper search (YYYY-MM-DD)",
    )

    # ArXiv categories
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["cs.AI", "cs.CL", "cs.LG"],
        help="ArXiv categories to search",
    )

    # Author filtering
    parser.add_argument(
        "--min-hindex",
        type=int,
        default=20,
        help="Minimum h-index for at least one author",
    )
    parser.add_argument(
        "--max-hindex",
        type=int,
        default=None,
        help="Maximum h-index (optional upper bound)",
    )

    # Paper limits
    parser.add_argument(
        "--max-papers-per-category",
        type=int,
        default=1000,  # the rate limit is 1000 papers per category
        help="Maximum papers per category",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=5,
        help="Minimum citations in related works section",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data_pipeline/outputs",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--no-save-papers", action="store_true", help="Don't save raw papers dataframe"
    )
    parser.add_argument(
        "--no-save-content",
        action="store_true",
        help="Don't save paper content dataframe",
    )
    parser.add_argument(
        "--no-save-citations",
        action="store_true",
        help="Don't save citations dataframes",
    )

    # Processing settings
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=5,
        help="Number of concurrent API requests",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Delay between requests (seconds)",
    )

    args = parser.parse_args()
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Create configuration
    config = PipelineConfig(
        start_date=start_date,
        end_date=end_date,
        existing_papers_csv=args.existing_papers_csv,
        arxiv_categories=args.categories,
        min_author_hindex=args.min_hindex,
        max_author_hindex=args.max_hindex,
        max_papers_per_category=args.max_papers_per_category,
        min_citations_in_related_works=args.min_citations,
        output_dir=args.output_dir,
        save_raw_papers=not args.no_save_papers,
        save_extracted_sections=not args.no_save_content,
        save_citations=not args.no_save_citations,
        concurrent_requests=args.concurrent_requests,
        request_delay=args.request_delay,
    )

    return args, config
