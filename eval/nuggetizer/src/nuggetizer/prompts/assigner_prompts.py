"""
Prompts for nugget assignment
"""

from typing import List, Dict
from ..core.types import ScoredNugget, NuggetAssignMode


def create_assign_prompt(
    query: str,
    context: str,
    nuggets: List[ScoredNugget],
    assigner_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3,
) -> List[Dict[str, str]]:
    """
    Creates a prompt for nugget assignment
    """
    messages = [
        {
            "role": "system",
            "content": "You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage.",
        },
        {
            "role": "user",
            "content": get_assign_prompt_content(
                query, context, nuggets, assigner_mode
            ),
        },
    ]
    return messages


def get_assign_prompt_content(
    query: str,
    context: str,
    nuggets: List[ScoredNugget],
    assigner_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3,
) -> str:
    """
    Gets the content for the nugget assignment prompt
    """
    nugget_texts = [nugget.text for nugget in nuggets]

    if assigner_mode == NuggetAssignMode.SUPPORT_GRADE_2:
        instruction = f"""Based on the query and passage, label each of the {len(nuggets)} nuggets either as support or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support; otherwise, label them as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget."""
    else:
        instruction = f"""Based on the query and passage, label each of the {len(nuggets)} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget."""

    return f"""{instruction}

Search Query: {query}
Passage: {context}
Nugget List: {nugget_texts}
Only return the list of labels (List[str]). Do not explain.
Labels:"""
