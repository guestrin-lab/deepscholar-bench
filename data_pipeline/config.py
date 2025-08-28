"""
Configuration module for the data pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import os


@dataclass
class PipelineConfig:
    """Configuration for the arxiv data collection pipeline."""

    # Date filtering (required fields first)
    start_date: datetime  # Start date for paper search
    end_date: datetime  # End date for paper search

    existing_papers_csv: Optional[str] = None

    # ArXiv search parameters
    arxiv_categories: List[str] = field(
        default_factory=lambda: [
            "cs.AI",  # Artificial Intelligence
            "cs.CL",  # Computation and Language
            "cs.LG",  # Machine Learning
            "cs.CV",  # Computer Vision
            "stat.ML",  # Statistics - Machine Learning
        ]
    )

    # Author filtering
    min_author_hindex: int = 20  # Minimum h-index for at least one author
    max_author_hindex: Optional[int] = None  # Maximum h-index (optional upper bound)

    # Paper filtering
    max_papers_per_category: int = (
        100  # Limit papers per category to avoid overwhelming
    )
    min_citations_in_related_works: int = (
        5  # Minimum citations in related works section
    )

    # Section extraction
    related_works_section_names: List[str] = field(
        default_factory=lambda: [
            "Related Work",
            "Related Works",
            # "Background",
            # "Literature Review",
            # "Prior Work",
            # "Previous Work"
        ]
    )

    # common_end_section_names: List[str] = field(default_factory=lambda: [
    #     "Acknowledgments",
    #     "Author Contributions",
    #     "Limitations",
    #     "References"
    # ])
    # Output settings
    output_dir: str = os.getcwd() + "/outputs"
    save_raw_papers: bool = True
    save_extracted_sections: bool = True
    save_citations: bool = True

    # Processing settings
    concurrent_requests: int = 5  # Number of concurrent API requests
    request_delay: float = 1.0  # Delay between requests (seconds)
