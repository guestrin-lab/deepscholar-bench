"""
Unit tests for the LatexExtractor module.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from data_pipeline.latex_extractor import LatexExtractor, PaperData, CitationData
from data_pipeline.config import PipelineConfig
from data_pipeline.arxiv_scraper import ArxivPaper

import logging

logger = logging.getLogger(__name__)

# Configure pytest for async tests
pytestmark = pytest.mark.asyncio


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return PipelineConfig(
        start_date=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_date=datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        arxiv_categories=["cs.AI", "cs.LG"],
        max_papers_per_category=2,
        request_delay=0.1,
        related_works_section_names=[
            "Related Work",
            "Related Works",
            "Background",
            "Literature Review",
        ],
        output_dir="./test_output",
    )


@pytest.fixture
def sample_papers():
    """Create sample papers for testing."""
    return [
        ArxivPaper(
            arxiv_id="2501.00677v1",
            title="Test Paper 1",
            authors=["John Doe", "Jane Smith"],
            abstract="This is a test abstract for paper 1.",
            categories=["cs.AI"],
            published_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            updated_date=datetime(2024, 1, 20, tzinfo=timezone.utc),
            abs_url="http://arxiv.org/abs/2501.00677v1",
        ),
        ArxivPaper(
            arxiv_id="2501.00673v1",
            title="Test Paper 2",
            authors=["Alice Johnson", "Bob Wilson"],
            abstract="This is a test abstract for paper 2.",
            categories=["cs.LG"],
            published_date=datetime(2024, 1, 16, tzinfo=timezone.utc),
            updated_date=datetime(2024, 1, 21, tzinfo=timezone.utc),
            abs_url="http://arxiv.org/abs/2501.00673v1",
        ),
    ]


@pytest.fixture
def sample_latex_content():
    """Sample LaTeX content with related works section."""
    return r"""
\documentclass{article}
\begin{document}
\title{Test Paper}
\author{Test Author}

\section{Introduction}
This is the introduction.

\section{Related Work}
This is the related work section with some citations \cite{author2020} and \cite{smith2021}.
More content here about previous work.

\section{Methodology}
This is the methodology section.
\end{document}
"""


@pytest.fixture
def sample_bib_content():
    """Sample bibliography content."""
    return r"""
@article{author2020,
    title={Test Paper Title},
    author={Author, A. and Smith, B.},
    journal={Test Journal},
    year={2020},
    doi={10.1234/test.2020}
}

@article{smith2021,
    title={Another Test Paper},
    author={Smith, C. and Jones, D.},
    journal={Another Journal},
    year={2021}
}
"""


@pytest.fixture
def sample_pdf_text():
    """Sample PDF text with related works section."""
    return """
Introduction
This is the introduction section.

2. Related Work
This is the related work section extracted from PDF.
It contains information about previous research.
Some citations like (Author et al., 2020) and (Smith, 2021).

3. Methodology
This is the methodology section.
"""


class TestLatexExtractor:
    """Test the LatexExtractor class."""

    def test_init(self, sample_config):
        """Test LatexExtractor initialization."""
        extractor = LatexExtractor(sample_config)
        assert extractor.config == sample_config

    @patch("data_pipeline.latex_extractor.LatexExtractor._download_latex_source")
    @patch(
        "data_pipeline.latex_extractor.LatexExtractor._extract_related_works_section"
    )
    @patch(
        "data_pipeline.latex_extractor.LatexExtractor._download_and_extract_pdf_related_works"
    )
    @patch("data_pipeline.latex_extractor.LatexExtractor._clean_latex_content")
    @patch("data_pipeline.latex_extractor.asyncio.sleep")
    async def test_extract_papers_content_success(
        self,
        mock_sleep,
        mock_clean,
        mock_pdf,
        mock_extract,
        mock_download,
        sample_config,
        sample_papers,
    ):
        """Test successful extraction of papers content."""
        extractor = LatexExtractor(sample_config)

        # Mock successful responses
        mock_download.return_value = "latex content"
        mock_extract.return_value = "related works content"
        mock_pdf.return_value = "pdf related works"
        mock_clean.return_value = "cleaned content"

        result = await extractor.extract_papers_content(sample_papers)

        assert len(result) == 2
        assert isinstance(result[0], PaperData)
        assert result[0].paper_title == "Test Paper 1"
        assert result[0].related_works_section == "pdf related works"
        assert result[1].paper_title == "Test Paper 2"

        # Verify methods were called
        assert mock_download.call_count == 2
        assert mock_extract.call_count == 2
        assert mock_pdf.call_count == 2

    @patch("data_pipeline.latex_extractor.LatexExtractor._extract_citations_from_text")
    @patch("data_pipeline.latex_extractor.asyncio.sleep")
    async def test_extract_citations_from_papers(
        self, mock_sleep, mock_extract_citations, sample_config
    ):
        """Test extraction of citations from papers."""
        extractor = LatexExtractor(sample_config)

        # Create sample paper data
        paper_data = [
            PaperData(
                arxiv_link="http://arxiv.org/abs/2501.00677v1",
                publication_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
                paper_title="Test Paper 1",
                abstract="Test abstract",
                related_works_section="Test related works",
            )
        ]
        paper_data[0]._latex_related_works = "latex related works"
        paper_data[0]._full_latex_content = "full latex content"
        paper_data[0]._paper_object = Mock()
        paper_data[0]._paper_object._project_files = {"references.bib": "bib content"}

        # Mock citation extraction
        mock_extract_citations.return_value = [
            CitationData(
                parent_paper_title="Test Paper 1",
                parent_arxiv_link="http://arxiv.org/abs/2501.00677v1",
                citation_shorthand="author2020",
                raw_citation_text="\\cite{author2020}",
            )
        ]

        result = await extractor.extract_citations_from_papers(paper_data)

        assert len(result) == 1
        assert isinstance(result[0], CitationData)
        assert result[0].citation_shorthand == "author2020"
        assert mock_extract_citations.call_count == 1

    def test_extract_related_works_section_direct(
        self, sample_config, sample_latex_content
    ):
        """Test extraction of related works section from LaTeX content."""
        extractor = LatexExtractor(sample_config)

        result = extractor._extract_related_works_section(sample_latex_content)

        assert result is not None
        assert "related work section with some citations" in result
        assert "\\cite{author2020}" in result

    def test_extract_related_works_section_not_found(self, sample_config):
        """Test extraction when related works section is not found."""
        extractor = LatexExtractor(sample_config)

        latex_content = r"""
\documentclass{article}
\begin{document}
\section{Introduction}
This is the introduction.

\section{Methodology}
This is the methodology section.
\end{document}
"""

        result = extractor._extract_related_works_section(latex_content)

        assert result is None

    @patch("data_pipeline.latex_extractor.requests.get")
    @patch("data_pipeline.latex_extractor.tarfile.open")
    @patch("data_pipeline.latex_extractor.LatexExtractor._find_main_tex_file")
    @patch("data_pipeline.latex_extractor.LatexExtractor._read_all_project_files")
    async def test_download_latex_source_success(
        self,
        mock_read_files,
        mock_find_tex,
        mock_tar,
        mock_get,
        sample_config,
        sample_papers,
    ):
        """Test successful download of LaTeX source."""
        extractor = LatexExtractor(sample_config)

        # Mock successful download
        mock_response = Mock()
        mock_response.content = b"tar.gz content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock tar extraction
        mock_tar_context = Mock()
        mock_tar.return_value.__enter__.return_value = mock_tar_context
        mock_tar.return_value.__exit__.return_value = None

        # Mock finding main tex file
        mock_find_tex.return_value = "main tex content"
        mock_read_files.return_value = {
            "file1.tex": "content1",
            "file2.bib": "content2",
        }

        result = await extractor._download_latex_source(sample_papers[0])

        assert result == "main tex content"
        assert sample_papers[0]._project_files == {
            "file1.tex": "content1",
            "file2.bib": "content2",
        }

    @patch("data_pipeline.latex_extractor.requests.get")
    @patch("data_pipeline.latex_extractor.LatexExtractor._extract_text_from_pdf")
    @patch(
        "data_pipeline.latex_extractor.LatexExtractor._extract_related_works_from_pdf_text"
    )
    async def test_download_and_extract_pdf_related_works_success(
        self,
        mock_extract_pdf_text,
        mock_extract_text,
        mock_get,
        sample_config,
        sample_papers,
        sample_pdf_text,
    ):
        """Test successful PDF download and extraction."""
        extractor = LatexExtractor(sample_config)

        # Mock successful PDF download
        mock_response = Mock()
        mock_response.content = b"pdf content"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Mock text extraction
        mock_extract_text.return_value = sample_pdf_text
        mock_extract_pdf_text.return_value = (
            "2. Related Work\nThis is the related work section extracted from PDF."
        )

        result = await extractor._download_and_extract_pdf_related_works(
            sample_papers[0]
        )

        assert (
            result
            == "2. Related Work\nThis is the related work section extracted from PDF."
        )
        assert mock_extract_text.call_count == 1
        assert mock_extract_pdf_text.call_count == 1

    @patch("data_pipeline.latex_extractor.LatexExtractor._lookup_citation_details")
    async def test_extract_citations_from_text(self, mock_lookup, sample_config):
        """Test extraction of citations from text."""
        extractor = LatexExtractor(sample_config)

        text = r"""
        This is a related works section with citations \cite{author2020} and \cite{smith2021}.
        Also some inline citations like (Author et al., 2020).
        """

        bibliography = {
            "author2020": {
                "title": "Test Paper Title",
                "author": "Author, A. and Smith, B.",
                "year": "2020",
            }
        }

        # Mock citation lookup
        mock_lookup.return_value = None

        result = await extractor._extract_citations_from_text(
            text, "Test Paper", "http://arxiv.org/abs/test", bibliography
        )

        assert len(result) >= 2  # Should find at least the LaTeX citations
        assert any(cite.citation_shorthand == "author2020" for cite in result)
        assert any(cite.citation_shorthand == "smith2021" for cite in result)

    def test_clean_latex_content(self, sample_config):
        """Test cleaning of LaTeX content."""
        extractor = LatexExtractor(sample_config)

        latex_content = r"""
        \section{Related Work}
        This is content with \textbf{bold text} and \textit{italic text}.
        % This is a comment
        \label{sec:related}
        """

        result = extractor._clean_latex_content(latex_content)
        print(result)
        assert "This is content with" in result
        assert "% This is a comment" not in result  # Comments should be removed
        assert "\\label{sec:related}" not in result  # Labels should be removed

    def test_parse_bib_file(self, sample_config, sample_bib_content):
        """Test parsing of bibliography file."""
        extractor = LatexExtractor(sample_config)

        result = extractor._parse_bib_file(sample_bib_content)

        assert len(result) == 2
        assert "author2020" in result
        assert "smith2021" in result
        assert result["author2020"]["title"] == "Test Paper Title"
        assert result["author2020"]["year"] == "2020"

    def test_extract_bibtex_field(self, sample_config):
        """Test extraction of BibTeX fields."""
        extractor = LatexExtractor(sample_config)

        fields = (
            "title={Test Paper Title}, author={Author, A. and Smith, B.}, year={2020}"
        )

        title = extractor._extract_bibtex_field(fields, "title")
        author = extractor._extract_bibtex_field(fields, "author")
        year = extractor._extract_bibtex_field(fields, "year")

        assert title == "Test Paper Title"
        assert author == "Author, A. and Smith, B."
        assert year == "2020"

    def test_remove_latex_comments(self, sample_config):
        """Test removal of LaTeX comments."""
        extractor = LatexExtractor(sample_config)

        text = r"""
        This is content % This is a comment
        This line has no comment
        This line has \% escaped percent % but this is a comment
        """

        result = extractor._remove_latex_comments(text)

        assert "This is content" in result
        assert "% This is a comment" not in result
        assert "This line has no comment" in result
        assert (
            "This line has \\% escaped percent" in result
        )  # Escaped % should remain with backslash
        assert "but this is a comment" not in result

    def test_extract_search_terms(self, sample_config):
        """Test the _extract_search_terms method."""
        extractor = LatexExtractor(sample_config)

        # Test cases with expected search terms
        test_cases = [
            ("guo2025deepseek", ["guo", "deepseek"]),
            ("jaech2024openai", ["jaech", "openai"]),
            ("team2024qwq", ["team", "qwq"]),
            ("snell2024scaling", ["snell", "scaling"]),
            ("brown2024large", ["brown", "large"]),
            ("noYearAuthor", ["noYearAuthor"]),
            ("2023onlyYear", ["onlyYear"]),
            ("", []),  # Edge case: empty string
        ]

        for citation_key, expected_terms in test_cases:
            search_terms = extractor._extract_search_terms(citation_key)
            assert set(search_terms) == set(expected_terms), (
                f"Failed for {citation_key}: {search_terms}"
            )


class TestLatexExtractorIntegration:
    """Integration tests for LatexExtractor with real API calls."""

    @pytest.mark.integration
    async def test_real_arxiv_paper_processing(self, sample_config):
        """Test processing a real ArXiv paper (marked as integration test)."""
        extractor = LatexExtractor(sample_config)

        # Use a real ArXiv paper ID
        real_papers = [
            ArxivPaper(
                arxiv_id="2502.07374",  # Recent paper with likely related works
                title="Test Paper 1",
                authors=["Test Author"],
                abstract="Test abstract",
                categories=["cs.AI"],
                published_date=datetime(2024, 10, 1),
                updated_date=datetime(2024, 10, 1),
                abs_url="https://arxiv.org/pdf/2502.07374",
            ),
        ]

        result = await extractor.extract_papers_content(real_papers)
        citations = await extractor.extract_citations_from_papers(result)
        assert isinstance(result, list)
        # Result could be empty if paper doesn't exist or has no related works
        assert len(result) == len(real_papers)
        assert result[0].related_works_section is not None
        assert len(citations) > 0

    @pytest.mark.integration
    async def test_arxiv_search_for_citation(self, sample_config):
        """Test the _search_arxiv_for_citation method."""
        extractor = LatexExtractor(sample_config)

        # Test cases with expected search terms
        test_cases = [
            ("jaech2024openai", ["2024", "jaech", "openai"]),
            ("snell2024scaling", ["2024", "snell", "scaling"]),
        ]

        for citation_key, expected_terms in test_cases:
            search_terms = extractor._extract_search_terms(citation_key)
            assert set(search_terms) == set(expected_terms), (
                f"Failed for {citation_key}: {search_terms}"
            )

            result = await extractor._search_arxiv_for_citation(expected_terms)
            assert result is not None, f"Failed for {citation_key}: {result}"

    def test_parse_bibliography_entry(self, sample_config):
        """Test the _parse_bibliography_entry method."""
        extractor = LatexExtractor(sample_config)

        # Test cases with expected title and authors
        test_cases = [
            (
                'Smith, "Deep Learning in AI", Journal of AI, 2023',
                ("Deep Learning in AI", ["Smith"]),
            ),
            (
                "Brown. Advanced AI Techniques. AI Journal. 2024.",
                ("Advanced AI Techniques", ["Brown"]),
            ),
            ("NoTitle, Journal of AI, 2023", (None, ["NoTitle"])),  # No title found
            ("", (None, None)),  # Edge case: empty string
        ]

        for bib_entry, expected in test_cases:
            title, authors = extractor._parse_bibliography_entry(bib_entry)
            assert title == expected[0], f"Title failed for {bib_entry}: {title}"
            assert authors == expected[1], f"Authors failed for {bib_entry}: {authors}"


if __name__ == "__main__":
    pytest.main([__file__])
