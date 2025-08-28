import argparse
import os
import logging

try:
    from parsers import ParserType
    from eval.evaluator import EvaluationFunction
except ImportError:
    from .parsers import ParserType
    from .evaluator import EvaluationFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modes",
        nargs="+",
        required=True,
        help="Mode to process. Can be: search_ai, openscholar, deepresearcher, deepscholar_base, storm, groundtruth",
        choices=list([mode.value for mode in ParserType]),
    )

    parser.add_argument(
        "--evals",
        nargs="+",
        default=[
            EvaluationFunction.ORGANIZATION.value,
            EvaluationFunction.NUGGET_COVERAGE.value,
            EvaluationFunction.REFERENCE_COVERAGE.value,
            EvaluationFunction.DOCUMENT_IMPORTANCE.value,
            EvaluationFunction.CITE_P.value,
            EvaluationFunction.CLAIM_COVERAGE.value,
            EvaluationFunction.COVERAGE_RELEVANCE_RATE.value,
        ],
        required=True,
        help="Evals to process. Can be: organization, nugget_coverage, reference_coverage, document_importance, cite_p, claim_coverage, coverage_relevance_rate",
        choices=list([eval.value for eval in EvaluationFunction]),
    )

    parser.add_argument(
        "--file_id",
        nargs="+",
        type=str,
        help="Specific file IDs to process. If not provided, processes all files in the input folder",
    )

    parser.add_argument(
        "--input_folder",
        nargs="+",
        type=str,
        required=True,
        help="Input folder containing the results. There should be one folder for each mode.",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="results",
        help="Output directory for citation stats files (default: results)",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/papers_with_related_works.csv",
        help="Path to the dataset CSV file",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o",
        help="Model to use for evaluation (default: gpt-4o)",
    )

    parser.add_argument(
        "--reference_folder",
        type=str,
        default="test/baselines_results/openscholar",
        help="Reference folder to use for getting file id for evaluation of ground truth (default: test/dataset/openscholar)",
    )

    parser.add_argument(
        "--important_citations_path",
        type=str,
        default="dataset/important_citations.csv",
        help="Path to the important citations CSV file",
    )

    parser.add_argument(
        "--nugget_groundtruth_dir_path",
        type=str,
        default="dataset/gt_nuggets_outputs",
        help="Path to the nugget groundtruth directory",
    )

    args = parser.parse_args()
    args.evals = [EvaluationFunction(eval) for eval in args.evals]
    args.modes = [ParserType(mode) for mode in args.modes]

    assert len(args.input_folder) == len(args.modes), (
        "Number of input folders must match number of modes"
    )

    logger.info(f"Processing modes: {args.modes}")
    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Output directory: {args.output_folder}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Evals: {args.evals}")

    if args.file_id:
        logger.info(f"Processing specific file IDs: {args.file_id}")
    else:
        logger.info("Processing all files in input folders")

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(os.path.join(args.output_folder, "tokens"), exist_ok=True)

    return args
