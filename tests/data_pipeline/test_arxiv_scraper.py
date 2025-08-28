"""
Unit tests for the ArXiv scraper.
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from data_pipeline.arxiv_scraper import ArxivScraper, ArxivPaper
from data_pipeline.config import PipelineConfig


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return PipelineConfig(
        start_date=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        end_date=datetime(2024, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
        arxiv_categories=["cs.AI", "cs.LG"],
        max_papers_per_category=2,
        request_delay=0.1,
    )


@pytest.fixture
def sample_arxiv_paper():
    """Create a sample ArxivPaper for testing."""
    return ArxivPaper(
        arxiv_id="2401.12345",
        title="Test Paper Title",
        authors=["John Doe", "Jane Smith"],
        abstract="This is a test abstract for the paper.",
        categories=["cs.AI", "cs.LG"],
        published_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
        updated_date=datetime(2024, 1, 20, tzinfo=timezone.utc),
        abs_url="http://arxiv.org/abs/2401.12345",
        doi="10.1234/test.2024.12345",
        journal_ref="Test Journal, 2024",
        comments="Test comments",
    )


@pytest.fixture
def mock_arxiv_result():
    """Create a mock arxiv.Result object."""
    mock_result = Mock()
    mock_result.entry_id = "http://arxiv.org/abs/2401.12345"
    mock_result.title = "Test Paper Title"
    mock_result.summary = "This is a test abstract for the paper."
    mock_result.categories = ["cs.AI", "cs.LG"]
    mock_result.published = datetime(2024, 1, 15, tzinfo=timezone.utc)
    mock_result.updated = datetime(2024, 1, 20, tzinfo=timezone.utc)

    # Mock authors
    mock_author1 = Mock()
    mock_author1.name = "John Doe"
    mock_author2 = Mock()
    mock_author2.name = "Jane Smith"
    mock_result.authors = [mock_author1, mock_author2]

    # Mock optional fields
    mock_result.doi = "10.1234/test.2024.12345"
    mock_result.journal_ref = "Test Journal, 2024"
    mock_result.comment = "Test comments"

    return mock_result


class TestArxivScraper:
    """Test the ArxivScraper class."""

    @patch("arxiv.Search")
    def test_fetch_paper_by_id_success(
        self, mock_search_class, sample_config, mock_arxiv_result
    ):
        """Test successfully fetching a paper by ID."""
        scraper = ArxivScraper(sample_config)

        # Mock the search
        mock_search = Mock()
        mock_search_class.return_value = mock_search

        # Mock the client.results() method directly
        mock_results = Mock()
        mock_results.__iter__ = Mock(return_value=iter([mock_arxiv_result]))
        scraper.client.results = Mock(return_value=mock_results)

        # Test with clean ID
        result = asyncio.run(scraper.fetch_paper_by_id("2401.12345"))

        assert result is not None
        assert result.arxiv_id == "2401.12345"
        assert result.title == "Test Paper Title"

        # Verify search was called with correct parameters
        mock_search_class.assert_called_with(id_list=["2401.12345"])

    @patch("arxiv.Search")
    def test_fetch_paper_by_id_with_prefix(
        self, mock_search_class, sample_config, mock_arxiv_result
    ):
        """Test fetching a paper by ID with 'arxiv:' prefix."""
        scraper = ArxivScraper(sample_config)

        # Mock the search
        mock_search = Mock()
        mock_search_class.return_value = mock_search

        # Mock the client.results() method directly
        mock_results = Mock()
        mock_results.__iter__ = Mock(return_value=iter([mock_arxiv_result]))
        scraper.client.results = Mock(return_value=mock_results)

        # Test with arxiv: prefix
        result = asyncio.run(scraper.fetch_paper_by_id("arxiv:2401.12345"))

        assert result is not None
        assert result.arxiv_id == "2401.12345"

        # Verify search was called with cleaned ID
        mock_search_class.assert_called_with(id_list=["2401.12345"])

    @patch("arxiv.Search")
    def test_fetch_paper_by_id_not_found(self, mock_search_class, sample_config):
        """Test fetching a paper by ID that doesn't exist."""
        scraper = ArxivScraper(sample_config)

        # Mock the search
        mock_search = Mock()
        mock_search_class.return_value = mock_search

        # Mock the client.results() method to return empty list
        mock_results = Mock()
        mock_results.__iter__ = Mock(return_value=iter([]))
        scraper.client.results = Mock(return_value=mock_results)

        result = asyncio.run(scraper.fetch_paper_by_id("9999.99999"))

        assert result is None

    @patch("arxiv.Search")
    @patch("data_pipeline.arxiv_scraper.asyncio.sleep")
    def test_search_category(
        self, mock_sleep, mock_search_class, sample_config, mock_arxiv_result
    ):
        """Test searching for papers in a category."""
        scraper = ArxivScraper(sample_config)

        # Mock the search
        mock_search = Mock()
        mock_search_class.return_value = mock_search

        # Mock the client.results() method directly
        mock_results = Mock()
        mock_results.__iter__ = Mock(return_value=iter([mock_arxiv_result]))
        scraper.client.results = Mock(return_value=mock_results)

        result = asyncio.run(scraper._search_category("cs.AI"))

        assert len(result) == 1
        assert result[0].arxiv_id == "2401.12345"
        assert result[0].categories == ["cs.AI", "cs.LG"]

        # Verify search was called with correct parameters
        expected_query = "cat:cs.AI AND submittedDate:[202401010000 TO 202412312359]"
        # Check that Search was called with the expected query and max_results
        mock_search_class.assert_called()
        call_args = mock_search_class.call_args
        assert call_args[1]["query"] == expected_query
        assert call_args[1]["max_results"] == 2

    @patch("arxiv.Search")
    @patch("data_pipeline.arxiv_scraper.asyncio.sleep")
    def test_search_papers(
        self, mock_sleep, mock_search_class, sample_config, mock_arxiv_result
    ):
        """Test the main search_papers method."""
        scraper = ArxivScraper(sample_config)

        # Mock the search
        mock_search = Mock()
        mock_search_class.return_value = mock_search

        # Mock the client.results() method to return the same result for both categories
        mock_results = Mock()
        mock_results.__iter__ = Mock(return_value=iter([mock_arxiv_result]))
        scraper.client.results = Mock(return_value=mock_results)

        result = asyncio.run(scraper.search_papers())

        # Should get at least one paper (may be deduplicated if same paper in both categories)
        assert len(result) >= 1

        # Verify sleep was called between categories
        assert mock_sleep.call_count == 2


class TestArxivScraperIntegration:
    """Integration tests for the ArxivScraper class."""

    def test_search_paper_with_arxiv_id_without_mock(self, sample_config):
        """Test searching for a paper with an ArXiv ID."""
        scraper = ArxivScraper(sample_config)
        arxiv_id = "2407.11418"
        result = asyncio.run(scraper.fetch_paper_by_id(arxiv_id))
        assert result is not None
        assert result.arxiv_id.startswith(arxiv_id)  # Allow for version suffix
        assert (
            result.title
            == "Semantic Operators: A Declarative Model for Rich, AI-based Data Processing"
        )

    def test_search_papers_without_mock(self, sample_config):
        """Test searching for papers without mocking."""
        scraper = ArxivScraper(sample_config)
        result = asyncio.run(scraper.search_papers())
        assert len(result) == 4
        assert sorted([r.arxiv_id for r in result]) == sorted(
            ["2501.00677v1", "2501.00673v1", "2501.00664v3", "2501.00669v1"]
        )
        for r in result:
            assert (
                sample_config.start_date <= r.published_date <= sample_config.end_date
            )


if __name__ == "__main__":
    pytest.main([__file__])
