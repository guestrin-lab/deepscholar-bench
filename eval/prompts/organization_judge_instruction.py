from pydantic import BaseModel
from typing import Literal

organization_judge_instruction = """
    You will receive the **title and abstract** of a research paper, together with 
    two candidate **related-work sections** (A and B) written for that paper.
    Do not consider the formatting of the text e.g., latex , markdown, etc. Only consider the content.
    
    Task: Decide which section—A or B—exhibits better organization and coherence.
    Return only one letter: A or B.

    How to judge (organization only)
    Ignore breadth of coverage, citation accuracy, and analytic depth. Assess:

    Logical structure – Clear introduction, grouping of related themes, and smooth progression of ideas.

    Paragraph cohesion – Each paragraph develops a single topic and flows naturally to the next.

    Clarity & readability – Minimal redundancy or contradictions; transitions guide the reader.

    Signposting – Helpful headings, topic sentences, or discourse markers (if provided).

    Pick the section that is easier to follow and better structured—no ties.

    Output format: return a single token:
    A – Section A is better organized.
    B – Section B is better organized.
    ### Paper under assessment
    
    {paper_title}
    {paper_abstract}
    ### Candidate related-work section A
    {related_work_a}
    ### Candidate related-work section B
    {related_work_b}
    Output your answer as a **JSON dictionary** in the following format:
    "decision": "A" or "B"
    "explanation": "One sentence clearly explaining the key differences between the two options and why the selected one is preferred."
    only output the dictionary, do not output any other text.
"""


class OrganizationResponse(BaseModel):
    explanation: str
    decision: Literal["A", "B"]
