"""
Prompts for nugget creation
"""

from typing import List, Dict
from ..core.types import Request


def create_nugget_prompt(
    request: Request, start: int, end: int, nuggets: List[str]
) -> List[Dict[str, str]]:
    """
    Creates a prompt for nugget creation
    """
    messages = [
        {
            "role": "system",
            "content": "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query.",
        },
        {
            "role": "user",
            "content": get_nugget_prompt_content(request, start, end, nuggets),
        },
    ]
    return messages


def get_nugget_prompt_content(
    request: Request,
    start: int,
    end: int,
    nuggets: List[str],
    creator_max_nuggets: int = 30,
) -> str:
    """
    Gets the content for the nugget creation prompt
    """
    context = "\n".join(
        [
            f"[{i + 1}] {doc.segment}"
            for i, doc in enumerate(request.documents[start:end])
        ]
    )

    return f"""Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process).  Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most {creator_max_nuggets} nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.

Search Query: {request.query.text}
Context:
{context}
Search Query: {request.query.text}
Initial Nugget List: {nuggets}
Initial Nugget List Length: {len(nuggets)}

Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of ".
Updated Nugget List:"""
