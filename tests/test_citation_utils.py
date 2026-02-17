import pytest
from eval.evaluator.citation_utils import custom_sent_tokenize, remove_citations, format_document


class TestCustomSentTokenize:

    def test_basic_sentences(self):
        text = "This is sentence one. This is sentence two."
        result = custom_sent_tokenize(text)
        assert len(result) == 2

    def test_et_al_not_split(self):
        text = "Smith et al. proposed a method. It was effective."
        result = custom_sent_tokenize(text)
        assert len(result) == 2
        assert "et al." in result[0]

    def test_empty_string(self):
        result = custom_sent_tokenize("")
        assert result == []


class TestRemoveCitations:

    def test_removes_single_citation(self):
        text = "This is a statement [1]."
        result = remove_citations(text)
        assert "[1]" not in result
        assert "This is a statement." == result

    def test_removes_multiple_citations(self):
        text = "Methods [1] were used [2] successfully [3]."
        result = remove_citations(text)
        assert "[1]" not in result
        assert "[2]" not in result
        assert "[3]" not in result

    def test_no_citations(self):
        text = "No citations here."
        result = remove_citations(text)
        assert result == "No citations here."

    def test_removes_pipe(self):
        text = "Some text | more text"
        result = remove_citations(text)
        assert "|" not in result


class TestFormatDocument:

    def test_basic_format(self):
        doc = ("Paper Title", "Paper abstract content")
        result = format_document(doc)
        assert result == "Title: Paper Title\nPaper abstract content"

    def test_empty_fields(self):
        doc = ("", "")
        result = format_document(doc)
        assert result == "Title: \n"


if __name__ == "__main__":
    pytest.main([__file__])

