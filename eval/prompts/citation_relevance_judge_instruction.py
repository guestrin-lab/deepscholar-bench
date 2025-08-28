from pydantic import BaseModel
from typing import Literal

citation_relevance_judge_instruction = """Given the title and abstract of a paper under assessment, the paper's ground-truth related-work section (written by human experts), and the title and abstract of a candidate reference paper, determine the relevance of the candidate reference to the related-work section.

Return a graded relevance score:
'2' – The reference is highly relevant to the related work section, reflecting prior work that directly addresses the problem or premise of the paper.
'1' – The reference is somewhat relevant to the related work section, reflecting prior work that is tangentially on-topic but does not directly address the problem or premises addressed in the paper.
'0' – The reference is irrelevant and should not be included in the related work section.

Instructions:
• Consider the main research topic and themes described in the related-work section.
• Score 2 if the reference directly addresses similar problems, methods, or core concepts.
• Score 1 if the reference is in a related area but doesn't directly tackle the same problem (gray area - optional but reasonable to include).
• Score 0 if the reference is off-topic or unrelated in scope.

Remember: You are only seeing the title and abstract of the reference, so the full content might be more relevant than it appears.

Paper under assessment:
{paper_title}
{paper_abstract}

Ground-truth related-work section:
{paper_related_work}

Candidate reference paper:
{ref_title}
{ref_abstract}

Return only the score in this format:
##final score: <0, 1, or 2>"""


class CitationRelevanceResponse(BaseModel):
    score: Literal[0, 1, 2]
