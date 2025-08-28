import lotus
import re


def get_support(premise: str, hypothesis: str) -> int:
    system_prompt = "You are an intelligent and fair evaluator."
    user_prompt = "You are an Attribution Validator. Your task is to verify whether a given reference can support the given claim.\n\n"
    user_prompt += f"Claim: {hypothesis}\n"
    user_prompt += f"Reference: {premise}\n\n"
    user_prompt += "Does the reference support the claim? Answer '1' if it supports the claim, or '0' if it does not.\n"
    user_prompt += "Do not explain your answer, just return '1' or '0'.\n"
    user_prompt += "Answer:"
    raw_answer = lotus.settings.lm.get_completion(system_prompt, user_prompt)
    answer_cleaned = re.sub(r"[^\w]", "", raw_answer)
    answer = int(answer_cleaned) if answer_cleaned in ["0", "1"] else 0
    return answer
