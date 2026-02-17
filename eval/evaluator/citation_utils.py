import re
from nltk.tokenize import sent_tokenize


def custom_sent_tokenize(text):
    protected_text = text
    protected_text = re.sub(r"\bet al\.", "ET_AL_PLACEHOLDER", protected_text)
    sentences = sent_tokenize(protected_text)
    sentences = [s.replace("ET_AL_PLACEHOLDER", "et al.") for s in sentences]
    return sentences


def remove_citations(text: str) -> str:
    return re.sub(r"\s*\[\d+\]", "", text).replace(" |", "").strip()


def format_document(doc: tuple[str, str]) -> str:
    return f"Title: {doc[0]}\n{doc[1]}"

