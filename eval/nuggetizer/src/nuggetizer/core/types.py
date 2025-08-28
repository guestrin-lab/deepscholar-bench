from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class NuggetMode(Enum):
    ATOMIC = "atomic"
    NOUN_PHRASE = "noun_phrase"
    QUESTION = "question"


class NuggetScoreMode(Enum):
    VITAL_OKAY = "vital_okay"


class NuggetAssignMode(Enum):
    SUPPORT_GRADE_2 = "support_grade_2"
    SUPPORT_GRADE_3 = "support_grade_3"


@dataclass
class Query:
    qid: str
    text: str


@dataclass
class Document:
    docid: str
    segment: str
    title: Optional[str] = None


@dataclass
class Request:
    query: Query
    documents: List[Document]


@dataclass
class Nugget:
    text: str


@dataclass
class ScoredNugget(Nugget):
    importance: str  # "vital" or "okay"


@dataclass
class AssignedNugget(Nugget):
    assignment: str  # "support", "not_support", or "partial_support"


@dataclass
class AssignedScoredNugget(ScoredNugget):
    assignment: str  # "support", "not_support", or "partial_support"
