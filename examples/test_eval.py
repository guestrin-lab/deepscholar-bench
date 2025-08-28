import logging
import pandas as pd
import os
import dotenv
from eval.parsers import get_parser, ParserType, Parser
from eval.evaluator import EvaluationFunction
import lotus
from lotus.models import LM

dotenv.load_dotenv()

lotus.settings.configure(
    lm=LM(
        model="gpt-4o",
    )
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

OUTPUT_TO_DIR = False


def main():
    parsers: list[Parser] = []
    for file_id in range(2):
        folder_path = f"tests/baselines_results/search_ai_gpt_4.1/{file_id}"
        config = {
            "mode": ParserType.SEARCH_AI,
            "file_id": file_id,
            "dataset": pd.read_csv("dataset/related_works_combined.csv"),
        }

        parser = get_parser(config, folder_path)
        logger.info(f"Type: {parser.parser_type}")
        logger.info(f"Content: {parser.clean_text[:200]}")
        logger.info(f"Docs: {len(parser.docs)}")
        parsers.append(parser)

    if OUTPUT_TO_DIR:
        os.makedirs("results", exist_ok=True)
        for eval in EvaluationFunction:
            try:
                eval.evaluate_all({"SearchAI": parsers}, "results")
            except Exception as e:
                logger.error(f"Error evaluating {eval}: {e}", exc_info=True)
    else:
        logger.info(
            f"Organization: {EvaluationFunction.ORGANIZATION.evaluate_all({'SearchAI': parsers}).to_dict()}"
        )
        logger.info(
            f"CiteP: {EvaluationFunction.CITE_P.evaluate_all({'SearchAI': parsers}).to_dict()}"
        )
        logger.info(
            f"Document importance: {EvaluationFunction.DOCUMENT_IMPORTANCE.evaluate_all({'SearchAI': parsers}).to_dict()}"
        )
        logger.info(
            f"Claim coverage: {EvaluationFunction.CLAIM_COVERAGE.evaluate_all({'SearchAI': parsers}).to_dict()}"
        )
        logger.info(
            f"Coverage relevance rate: {EvaluationFunction.COVERAGE_RELEVANCE_RATE.evaluate_all({'SearchAI': parsers}).to_dict()}"
        )
        logger.info(
            f"Reference coverage: {EvaluationFunction.REFERENCE_COVERAGE.evaluate_all({'SearchAI': parsers}, important_citations_path='dataset/important_citations.csv').to_dict()}"
        )
        logger.info(
            f"Nugget coverage: {EvaluationFunction.NUGGET_COVERAGE.evaluate_all({'SearchAI': parsers}, nugget_groundtruth_dir_path='dataset/gt_nuggets_outputs').to_dict()}"
        )


if __name__ == "__main__":
    main()
