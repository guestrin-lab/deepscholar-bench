import argparse
import os
import sys
from datetime import datetime, timedelta

try:
    from config import PipelineConfig
except ImportError:
    from .config import PipelineConfig

# Category sets per field (used by --field shorthand)
FIELD_CATEGORIES: dict[str, list[str]] = {
    "cs": [
        "cs.IR", "cs.CV", "cs.AI", "cs.CL", "cs.LG", "cs.DC", "cs.DB",
        "cs.AR", "cs.SD", "cs.CR", "cs.ET", "cs.GR", "cs.PL", "cs.SY",
        "cs.OS", "cs.PF", "cs.SE", "cs.MM",
    ],
    "bio": [
        "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC",
        "q-bio.OT", "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO",
    ],
    "econ": ["econ.EM", "econ.GN", "econ.TH"],
    "phy": [
        "astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.HE",
        "astro-ph.IM", "astro-ph.SR", "cond-mat.dis-nn", "cond-mat.mes-hall",
        "cond-mat.mtrl-sci", "cond-mat.other", "cond-mat.quant-gas",
        "cond-mat.soft", "cond-mat.stat-mech", "cond-mat.str-el",
        "cond-mat.supr-con", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
        "math-ph", "nlin.AO", "nlin.CD", "nlin.CG", "nlin.PS", "nlin.SI",
        "nucl-ex", "nucl-th", "quant-ph", "physics.acc-ph", "physics.ao-ph",
        "physics.app-ph", "physics.atm-clus", "physics.atom-ph",
        "physics.bio-ph", "physics.chem-ph", "physics.class-ph",
        "physics.comp-ph", "physics.data-an", "physics.ed-ph",
        "physics.flu-dyn", "physics.gen-ph", "physics.geo-ph",
        "physics.hist-ph", "physics.ins-det", "physics.med-ph",
        "physics.optics", "physics.plasm-ph", "physics.pop-ph",
        "physics.soc-ph", "physics.space-ph",
    ],
    "stat": ["stat.ML", "stat.OT", "stat.TH", "stat.AP", "stat.CO", "stat.ME"],
}

_DEFAULT_CONFIG_YAML = "configs/data_pipeline.yaml"


def _load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents as a dict. Returns {} on any error."""
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            print(f"Warning: {path} is not a YAML mapping; ignoring.", file=sys.stderr)
            return {}
        return data
    except Exception as e:
        print(f"Warning: Could not load config {path}: {e}", file=sys.stderr)
        return {}


def parse_args():
    # Pre-parse to get --config-yaml before building the main parser
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config-yaml", default=_DEFAULT_CONFIG_YAML)
    pre_args, _ = pre_parser.parse_known_args()

    yaml_values = _load_yaml(pre_args.config_yaml)

    # 'fields' needs special handling due to action="append"; extract and handle separately
    yaml_fields: list[str] = yaml_values.pop("fields", [])
    if isinstance(yaml_fields, str):
        yaml_fields = [yaml_fields]

    parser = argparse.ArgumentParser(description="ArXiv Data Collection Pipeline")

    parser.add_argument(
        "--config-yaml",
        type=str,
        default=pre_args.config_yaml,
        help=(
            f"Path to YAML config file (default: {_DEFAULT_CONFIG_YAML}). "
            "Provides base values; any explicitly-passed CLI arg overrides the YAML."
        ),
    )

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

    # ArXiv categories — specify either --field or --categories (or both)
    parser.add_argument(
        "--field",
        dest="fields",
        action="append",
        default=[],
        choices=list(FIELD_CATEGORIES.keys()),
        metavar="FIELD",
        help=(
            "Convenience alias: expand a named field into its ArXiv categories "
            f"({', '.join(FIELD_CATEGORIES.keys())}). "
            "Can be repeated to combine fields. Use --categories for custom lists."
        ),
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Explicit ArXiv categories to search (e.g. cs.AI cs.LG). "
             "If --field is also given, both sets are merged.",
    )

    # Author filtering
    parser.add_argument(
        "--min-hindex",
        type=int,
        default=0,
        help="Minimum h-index for at least one author",
    )
    parser.add_argument(
        "--max-hindex",
        type=int,
        default=None,
        help="Maximum h-index (optional upper bound)",
    )

    # Paper filtering
    parser.add_argument(
        "--filter-accepted-papers",
        action="store_true",
        help="Filter only papers with 'published' or 'accepted' in comments",
    )

    # Paper limits
    parser.add_argument(
        "--max-papers-per-category",
        type=int,
        default=1000,
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

    # Apply YAML as defaults so CLI args naturally override them.
    # 'fields' is handled separately below due to action="append".
    if yaml_values:
        parser.set_defaults(**yaml_values)

    args = parser.parse_args()

    # Resolve --field into categories and merge with --categories.
    # If no --field was passed on CLI, fall back to YAML fields.
    cli_fields: list[str] = args.fields  # populated only by explicit CLI --field flags
    effective_fields = cli_fields if cli_fields else yaml_fields

    field_categories: list[str] = []
    for f in effective_fields:
        if f in FIELD_CATEGORIES:
            field_categories.extend(FIELD_CATEGORIES[f])
        else:
            parser.error(
                f"Unknown field '{f}'. Choices: {', '.join(FIELD_CATEGORIES.keys())}"
            )

    if field_categories and args.categories:
        # Both given: merge (deduplicate, preserve order)
        seen: set[str] = set()
        merged: list[str] = []
        for c in field_categories + args.categories:
            if c not in seen:
                seen.add(c)
                merged.append(c)
        args.categories = merged
    elif field_categories:
        args.categories = field_categories
    elif args.categories is None:
        # Neither --field nor --categories nor YAML categories: fall back to default
        args.categories = ["cs.AI", "cs.CL", "cs.LG"]

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    categories = []
    for category in args.categories:
        categories.extend(category.split("\n"))
    print(f"Categories: {categories}")

    # Create configuration
    config = PipelineConfig(
        start_date=start_date,
        end_date=end_date,
        existing_papers_csv=args.existing_papers_csv,
        arxiv_categories=categories,
        min_author_hindex=args.min_hindex,
        max_author_hindex=args.max_hindex,
        filter_accepted_papers=args.filter_accepted_papers,
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
