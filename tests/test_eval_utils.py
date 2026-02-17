import pytest
import os
from unittest.mock import patch


class TestOpenAlexEmailConfig:

    def test_default_email(self):
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import eval.utils as utils_mod
            importlib.reload(utils_mod)
            assert utils_mod.OPENALEX_MAILTO == "your@email.com"

    def test_custom_email_from_env(self):
        with patch.dict(os.environ, {"OPENALEX_MAILTO": "test@test.com"}):
            import importlib
            import eval.utils as utils_mod
            importlib.reload(utils_mod)
            assert utils_mod.OPENALEX_MAILTO == "test@test.com"


if __name__ == "__main__":
    pytest.main([__file__])

