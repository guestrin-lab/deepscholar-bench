"""
Unit tests for the AuthorFilter module.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch

from data_pipeline.author_filter import AuthorFilter, AuthorInfo
from data_pipeline.config import PipelineConfig
from data_pipeline.arxiv_scraper import ArxivPaper

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
        min_author_hindex=20,
        max_author_hindex=100,
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
        ArxivPaper(
            arxiv_id="2501.00664v3",
            title="Test Paper 3",
            authors=["Charlie Brown", "Diana Prince"],
            abstract="This is a test abstract for paper 3.",
            categories=["cs.AI"],
            published_date=datetime(2024, 1, 17, tzinfo=timezone.utc),
            updated_date=datetime(2024, 1, 22, tzinfo=timezone.utc),
            abs_url="http://arxiv.org/abs/2501.00664v3",
        ),
        ArxivPaper(
            arxiv_id="2501.00669v1",
            title="Test Paper 4",
            authors=["Eve Adams", "Frank Miller"],
            abstract="This is a test abstract for paper 4.",
            categories=["cs.LG"],
            published_date=datetime(2024, 1, 18, tzinfo=timezone.utc),
            updated_date=datetime(2024, 1, 23, tzinfo=timezone.utc),
            abs_url="http://arxiv.org/abs/2501.00669v1",
        ),
    ]


@pytest.fixture
def sample_author_info():
    """Create sample author information."""
    return AuthorInfo(
        name="John Doe", affiliation="Stanford University", hindex=45, citations=1500
    )


class TestAuthorFilter:
    """Test the AuthorFilter class."""

    def test_author_meets_criteria_within_range(self, sample_config):
        """Test author meets criteria when h-index is within range."""
        filter_obj = AuthorFilter(sample_config)
        author1 = AuthorInfo(name="Test Author", hindex=50)
        author2 = AuthorInfo(name="Test Author", hindex=15)
        author3 = AuthorInfo(name="Test Author", hindex=150)
        author4 = AuthorInfo(name="Test Author", hindex=None)
        assert filter_obj._author_meets_criteria(author1) is True
        assert filter_obj._author_meets_criteria(author2) is False
        assert filter_obj._author_meets_criteria(author3) is False
        assert filter_obj._author_meets_criteria(author4) is False

    @patch("data_pipeline.author_filter.AuthorFilter._get_author_from_semantic_scholar")
    async def test_get_author_info_success(self, mock_semantic, sample_config):
        """Test successful author info retrieval."""
        filter_obj = AuthorFilter(sample_config)

        # Mock successful Semantic Scholar response
        mock_semantic.return_value = AuthorInfo(
            name="John Doe",
            hindex=45,
            citations=1500,
            affiliation="Stanford University",
        )

        result = await filter_obj._get_author_info("John Doe")

        assert result is not None
        assert result.name == "John Doe"
        assert result.hindex == 45
        assert result.citations == 1500
        assert result.affiliation == "Stanford University"

        # Check caching
        assert "John Doe" in filter_obj._author_cache

    @patch("data_pipeline.author_filter.AuthorFilter._get_author_from_semantic_scholar")
    async def test_get_author_info_caching(self, mock_semantic, sample_config):
        """Test that author info is cached."""
        filter_obj = AuthorFilter(sample_config)

        # Mock successful response
        mock_semantic.return_value = AuthorInfo(
            name="John Doe", hindex=45, citations=1500
        )

        # First call
        result1 = await filter_obj._get_author_info("John Doe")

        # Second call should use cache
        result2 = await filter_obj._get_author_info("John Doe")

        assert result1 == result2
        assert mock_semantic.call_count == 1  # Only called once

    @patch("data_pipeline.author_filter.AuthorFilter._get_author_from_semantic_scholar")
    async def test_get_author_info_fallback(self, mock_semantic, sample_config):
        """Test fallback to AuthorInfo with None h-index when API fails."""
        filter_obj = AuthorFilter(sample_config)

        # Mock API failure
        mock_semantic.return_value = None

        result = await filter_obj._get_author_info("Unknown Author")

        assert result is not None
        assert result.name == "Unknown Author"
        assert result.hindex is None
        assert result.affiliation is None
        assert result.citations is None

    @patch("data_pipeline.author_filter.AuthorFilter._paper_meets_hindex_criteria")
    @patch("data_pipeline.author_filter.asyncio.sleep")
    async def test_filter_papers_by_author_hindex(
        self, mock_sleep, mock_meets_criteria, sample_config, sample_papers
    ):
        """Test filtering papers based on author h-index criteria."""
        filter_obj = AuthorFilter(sample_config)

        # Mock some papers meeting criteria, some not
        mock_meets_criteria.side_effect = [True, False, True, False]

        result = await filter_obj.filter_papers_by_author_hindex(sample_papers)

        # Should return papers 0 and 2 (indices where criteria was met)
        assert len(result) == 2
        assert result[0].arxiv_id == "2501.00677v1"
        assert result[1].arxiv_id == "2501.00664v3"

        # Verify sleep was called between each paper
        assert mock_sleep.call_count == 4


class TestAuthorFilterIntegration:
    """Integration tests for AuthorFilter with real API calls."""

    @pytest.mark.integration
    async def test_real_semantic_scholar_lookup(self, sample_config):
        """Test real Semantic Scholar API lookup (marked as integration test)."""
        filter_obj = AuthorFilter(sample_config)
        result = await filter_obj._get_author_from_semantic_scholar("Yoshua Bengio")
        assert result is not None
        assert result.name is not None

    @pytest.mark.integration
    async def test_filter_real_papers(self, sample_config):
        """Test filtering real papers with actual API calls."""
        filter_obj = AuthorFilter(sample_config)
        real_papers = [
            ArxivPaper(
                arxiv_id="2501.00677v1",
                title="Deeply Learned Robust Matrix Completion for Large-scale Low-rank Data Recovery",
                authors=["HanQin Cai", "Chandra Kundu", "Jialin Liu", "Wotao Yin"],
                abstract="Robust matrix completion (RMC) is a widely used machine learning tool that\nsimultaneously tackles two critical issues in low-rank data analysis: missing\ndata entries and extreme outliers. This paper proposes a novel scalable and\nlearnable non-convex approach, coined Learned Robust Matrix Completion (LRMC),\nfor large-scale RMC problems. LRMC enjoys low computational complexity with\nlinear convergence. Motivated by the proposed theorem, the free parameters of\nLRMC can be effectively learned via deep unfolding to achieve optimum\nperformance. Furthermore, this paper proposes a flexible\nfeedforward-recurrent-mixed neural network framework that extends deep\nunfolding from fix-number iterations to infinite iterations. The superior\nempirical performance of LRMC is verified with extensive experiments against\nstate-of-the-art on synthetic datasets and real applications, including video\nbackground subtraction, ultrasound imaging, face modeling, and cloud removal\nfrom satellite imagery.",
                categories=[
                    "cs.LG",
                    "cs.CV",
                    "cs.IT",
                    "cs.NA",
                    "math.IT",
                    "math.NA",
                    "stat.ML",
                ],
                published_date=datetime(2024, 12, 31, 23, 22, 12, tzinfo=timezone.utc),
                updated_date=datetime(2024, 12, 31, 23, 22, 12, tzinfo=timezone.utc),
                abs_url="http://arxiv.org/abs/2501.00677v1",
                doi=None,
                journal_ref=None,
                comments="arXiv admin note: substantial text overlap with arXiv:2110.05649",
            ),
            ArxivPaper(
                arxiv_id="2501.00673v1",
                title="Controlled Causal Hallucinations Can Estimate Phantom Nodes in Multiexpert Mixtures of Fuzzy Cognitive Maps",
                authors=["Akash Kumar Panda", "Bart Kosko"],
                abstract="An adaptive multiexpert mixture of feedback causal models can approximate\nmissing or phantom nodes in large-scale causal models. The result gives a\nscalable form of \\emph{big knowledge}. The mixed model approximates a sampled\ndynamical system by approximating its main limit-cycle equilibria. Each expert\nfirst draws a fuzzy cognitive map (FCM) with at least one missing causal node\nor variable. FCMs are directed signed partial-causality cyclic graphs. They mix\nnaturally through convex combination to produce a new causal feedback FCM.\nSupervised learning helps each expert FCM estimate its phantom node by\ncomparing the FCM's partial equilibrium with the complete multi-node\nequilibrium. Such phantom-node estimation allows partial control over these\ncausal hallucinations and helps approximate the future trajectory of the\ndynamical system. But the approximation can be computationally heavy. Mixing\nthe tuned expert FCMs gives a practical way to find several phantom nodes and\nthereby better approximate the feedback system's true equilibrium behavior.",
                categories=["cs.LG"],
                published_date=datetime(2024, 12, 31, 23, 1, 32, tzinfo=timezone.utc),
                updated_date=datetime(2024, 12, 31, 23, 1, 32, tzinfo=timezone.utc),
                abs_url="http://arxiv.org/abs/2501.00673v1",
                doi=None,
                journal_ref=None,
                comments="17 pages, 9 figures, The Ninth International Conference on Data\n  Mining and Big Data 2024 (DMBD 2024), 13 December 2024",
            ),
            ArxivPaper(
                arxiv_id="2501.00664v3",
                title="Grade Inflation in Generative Models",
                authors=[
                    "Phuc Nguyen",
                    "Miao Li",
                    "Alexandra Morgan",
                    "Rima Arnaout",
                    "Ramy Arnaout",
                ],
                abstract='Generative models hold great potential, but only if one can trust the\nevaluation of the data they generate. We show that many commonly used quality\nscores for comparing two-dimensional distributions of synthetic vs.\nground-truth data give better results than they should, a phenomenon we call\nthe "grade inflation problem." We show that the correlation score, Jaccard\nscore, earth-mover\'s score, and Kullback-Leibler (relative-entropy) score all\nsuffer grade inflation. We propose that any score that values all datapoints\nequally, as these do, will also exhibit grade inflation; we refer to such\nscores as "equipoint" scores. We introduce the concept of "equidensity" scores,\nand present the Eden score, to our knowledge the first example of such a score.\nWe found that Eden avoids grade inflation and agrees better with human\nperception of goodness-of-fit than the equipoint scores above. We propose that\nany reasonable equidensity score will avoid grade inflation. We identify a\nconnection between equidensity scores and R\\\'enyi entropy of negative order. We\nconclude that equidensity scores are likely to outperform equipoint scores for\ngenerative models, and for comparing low-dimensional distributions more\ngenerally.',
                categories=["cs.AI", "cs.LG", "stat.ML"],
                published_date=datetime(2024, 12, 31, 22, 34, 54, tzinfo=timezone.utc),
                updated_date=datetime(2025, 1, 22, 21, 15, 18, tzinfo=timezone.utc),
                abs_url="http://arxiv.org/abs/2501.00664v3",
                doi=None,
                journal_ref=None,
                comments="10 pages, 6 figures, 1 table",
            ),
            ArxivPaper(
                arxiv_id="2501.00669v1",
                title="Leaf diseases detection using deep learning methods",
                authors=["El Houcine El Fatimi"],
                abstract="This study, our main topic is to devlop a new deep-learning approachs for\nplant leaf disease identification and detection using leaf image datasets. We\nalso discussed the challenges facing current methods of leaf disease detection\nand how deep learning may be used to overcome these challenges and enhance the\naccuracy of disease detection. Therefore, we have proposed a novel method for\nthe detection of various leaf diseases in crops, along with the identification\nand description of an efficient network architecture that encompasses\nhyperparameters and optimization methods. The effectiveness of different\narchitectures was compared and evaluated to see the best architecture\nconfiguration and to create an effective model that can quickly detect leaf\ndisease. In addition to the work done on pre-trained models, we proposed a new\nmodel based on CNN, which provides an efficient method for identifying and\ndetecting plant leaf disease. Furthermore, we evaluated the efficacy of our\nmodel and compared the results to those of some pre-trained state-of-the-art\narchitectures.",
                categories=["cs.LG", "cs.AI", "cs.CV"],
                published_date=datetime(2024, 12, 31, 22, 56, 19, tzinfo=timezone.utc),
                updated_date=datetime(2024, 12, 31, 22, 56, 19, tzinfo=timezone.utc),
                abs_url="http://arxiv.org/abs/2501.00669v1",
                doi=None,
                journal_ref=None,
                comments="252 pages , 42 images",
            ),
        ]
        result = await filter_obj.filter_papers_by_author_hindex(real_papers)
        assert isinstance(result, list)
        # The result may vary based on API responses, so we just check it's a list
        # and has reasonable length (could be 0-4 depending on author h-indices)
        assert 0 <= len(result) <= len(real_papers)


if __name__ == "__main__":
    pytest.main([__file__])
