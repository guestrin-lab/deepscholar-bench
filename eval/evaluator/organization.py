import lotus  # noqa: F401
import pandas as pd

try:
    from evaluator import Evaluator, EvaluationFunction
    from parsers import Parser
    from prompts.organization_judge_instruction import (
        organization_judge_instruction,
        OrganizationResponse,
    )
except ImportError:
    from .evaluator import Evaluator
    from ..parsers import Parser
    from ..prompts.organization_judge_instruction import (
        organization_judge_instruction,
        OrganizationResponse,
    )
    from .enum import EvaluationFunction


class OrganizationEvaluator(Evaluator):
    evaluation_function = EvaluationFunction.ORGANIZATION

    def calculate(self, parsers: list[Parser]) -> pd.DataFrame:
        system_prompt = "You are an intelligent, rigorous, and fair evaluator of scholarly writing quality and relevance."
        infos = [
            parser.get_folder_info(include_related_works_section=True)
            for parser in parsers
        ]
        df = pd.DataFrame(infos)
        df.rename(
            columns={
                "generated_related_works_section": "related_work_a",
                "related_works_section": "related_work_b",
            },
            inplace=True,
        )
        df.head()
        results: pd.DataFrame = df.pairwise_judge(
            col1="related_work_a",
            col2="related_work_b",
            judge_instruction=organization_judge_instruction,
            system_prompt=system_prompt,
            n_trials=2,  # run two trials
            permute_cols=True,  # evaluate both (A,B) and (B,A) orders
            response_format=OrganizationResponse,
        )
        
        # _judge_0 compares (A, B)
        # _judge_1 compares (B, A) due to permute_cols=True
        # decision "A" means A is better, "B" means B is better
        def safe_get_decision(judge_response, default="B"):
            """Safely extract decision from judge response, handling edge cases."""
            if not hasattr(judge_response, "decision"):
                return default
            decision = str(judge_response.decision).strip().upper()
            # Check if "A" or "B" is in the decision string (more robust)
            if "A" in decision:
                return "A"
            elif "B" in decision:
                return "B"
            else:
                return default
        
        results["score_1"] = results["_judge_0"].map(
            lambda x: 1 if safe_get_decision(x, "B") == "A" else 0
        )
        results["score_2"] = results["_judge_1"].map(
            lambda x: 1 if safe_get_decision(x, "B") == "A" else 0
        )
        # Store individual scores as organization_v1 and organization_v2
        results["organization_v1"] = results["score_1"]
        results["organization_v2"] = results["score_2"]
        # Calculate final organization score as average
        results[self.evaluation_function.value] = (
            results["score_1"] + results["score_2"]
        ) / 2
        # Drop intermediate columns but keep organization_v1 and organization_v2
        results.drop(
            columns=["_judge_0", "_judge_1", "score_1", "score_2"], inplace=True
        )

        return results
