import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
import sys

lotus_available = True
try:
    import lotus
    from deepscholar_base.search.agentic_search import _exa_search, _exa_get_contents
    from deepscholar_base.search.recursive_search import _process_single_exa_search
    from deepscholar_base.configs import Configs
    from lotus.models import LM
except ImportError:
    lotus_available = False

skip_reason = "lotus/agents dependencies not installed or deepscholar_base shadowed by test file"


@pytest.mark.skipif(not lotus_available, reason=skip_reason)
class TestExaSearchAgentic:

    @patch("deepscholar_base.search.agentic_search.Exa")
    def test_exa_search_returns_dataframe(self, mock_exa_cls):
        mock_result = MagicMock()
        mock_result.title = "Test Paper"
        mock_result.url = "https://example.com/paper"
        mock_result.text = "Abstract text"
        mock_result.published_date = "2024-01-01"

        mock_search_results = MagicMock()
        mock_search_results.results = [mock_result]

        mock_exa_instance = MagicMock()
        mock_exa_instance.search.return_value = mock_search_results
        mock_exa_cls.return_value = mock_exa_instance

        df = _exa_search(["test query"], max_results=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["title"] == "Test Paper"
        assert df.iloc[0]["url"] == "https://example.com/paper"

    @patch("deepscholar_base.search.agentic_search.Exa")
    def test_exa_search_with_end_date(self, mock_exa_cls):
        mock_search_results = MagicMock()
        mock_search_results.results = []
        mock_exa_instance = MagicMock()
        mock_exa_instance.search.return_value = mock_search_results
        mock_exa_cls.return_value = mock_exa_instance

        end_date = datetime(2024, 6, 1)
        df = _exa_search(["test query"], max_results=5, end_date=end_date)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

        call_kwargs = mock_exa_instance.search.call_args[1]
        assert "end_published_date" in call_kwargs

    @patch("deepscholar_base.search.agentic_search.Exa")
    def test_exa_search_handles_exception(self, mock_exa_cls):
        mock_exa_instance = MagicMock()
        mock_exa_instance.search.side_effect = Exception("API Error")
        mock_exa_cls.return_value = mock_exa_instance

        df = _exa_search(["test query"], max_results=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    @patch("deepscholar_base.search.agentic_search.Exa")
    def test_exa_get_contents_returns_dataframe(self, mock_exa_cls):
        mock_result = MagicMock()
        mock_result.title = "Test Paper"
        mock_result.url = "https://example.com/paper"
        mock_result.text = "Full text content"

        mock_contents = MagicMock()
        mock_contents.results = [mock_result]

        mock_exa_instance = MagicMock()
        mock_exa_instance.get_contents.return_value = mock_contents
        mock_exa_cls.return_value = mock_exa_instance

        df = _exa_get_contents(["https://example.com/paper"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["full_text"] == "Full text content"

    @patch("deepscholar_base.search.agentic_search.Exa")
    def test_exa_get_contents_handles_exception(self, mock_exa_cls):
        mock_exa_instance = MagicMock()
        mock_exa_instance.get_contents.side_effect = Exception("API Error")
        mock_exa_cls.return_value = mock_exa_instance

        df = _exa_get_contents(["https://example.com/paper"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


@pytest.mark.skipif(not lotus_available, reason=skip_reason)
class TestExaSearchRecursive:

    @pytest.mark.asyncio
    @patch("deepscholar_base.search.recursive_search.Exa")
    async def test_process_single_exa_search(self, mock_exa_cls):
        mock_result = MagicMock()
        mock_result.title = "Test Paper"
        mock_result.url = "https://example.com/paper"
        mock_result.text = "Abstract text"
        mock_result.published_date = "2024-01-01"

        mock_search_results = MagicMock()
        mock_search_results.results = [mock_result]

        mock_exa_instance = MagicMock()
        mock_exa_instance.search.return_value = mock_search_results
        mock_exa_cls.return_value = mock_exa_instance

        configs = Configs(lm=LM(model="gpt-4o"))
        df = await _process_single_exa_search(configs, "test query", 5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


@pytest.mark.skipif(not lotus_available, reason=skip_reason)
class TestExaConfigFlag:

    def test_enable_exa_search_default_false(self):
        configs = Configs(lm=LM(model="gpt-4o"))
        assert configs.enable_exa_search is False

    def test_enable_exa_search_can_be_set_true(self):
        configs = Configs(lm=LM(model="gpt-4o"), enable_exa_search=True)
        assert configs.enable_exa_search is True


class TestExaImport:

    def test_exa_py_importable(self):
        import exa_py
        assert hasattr(exa_py, "Exa")


if __name__ == "__main__":
    pytest.main([__file__])
