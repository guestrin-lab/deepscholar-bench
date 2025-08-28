# evaluation_suite.py - Combined Citation Stats Evaluation Version
import os
import pandas as pd
import argparse
import logging
from tabulate import tabulate  # type: ignore
import lotus
from lotus.models import LM

try:
    from parsers import get_parser, Parser, ParserType
    from argument_parser import parse_args
except ImportError:
    from .parsers import get_parser, Parser, ParserType
    from .argument_parser import parse_args

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_mode(
    args: argparse.Namespace,
    mode: ParserType,
    input_folder: str,
    dataset: pd.DataFrame,
) -> list[Parser]:
    mode_results: list[Parser] = []

    if not os.path.exists(input_folder) and mode != ParserType.GROUNDTRUTH:
        logger.warning(f"Warning: Mode folder {input_folder} not found, skipping")
        return []

    if args.file_id:
        files_to_process: list[str] = [
            fid
            for fid in args.file_id
            if os.path.exists(os.path.join(input_folder, fid))
        ]
    else:
        if mode == ParserType.GROUNDTRUTH:
            try:
                files_to_process = sorted(os.listdir(args.reference_folder))
            except FileNotFoundError:
                logger.warning(
                    f"Warning: Reference folder {args.reference_folder} not found, skipping"
                )
                return []
        else:
            files_to_process = sorted(os.listdir(input_folder))

    for file_id in files_to_process:
        folder_path = os.path.join(input_folder, file_id)

        try:
            parser = get_parser(
                {
                    "mode": mode,
                    "file_id": file_id.replace(".json", ""),
                    "dataset": dataset,
                },
                folder_path,
            )
        except Exception as e:
            logger.warning(f"❌ Error processing {mode.value} / {file_id}: {e}")
            continue

        if parser.docs:
            mode_results.append(parser)
            logger.info(f"✅ {mode.value} / {file_id} processed")
        else:
            logger.warning(
                f"❌ Error processing {mode.value} / {file_id}: No docs found"
            )

    if mode_results:
        logger.info(f"Processed {len(mode_results)} results for mode: {mode.value}")
    else:
        logger.warning(f"No  results to save for mode: {mode.value}")

    return mode_results


def pretty_print_results(results: pd.DataFrame) -> None:
    print(tabulate(results, headers="keys", tablefmt="grid"))


def main() -> None:
    """Main function to run the citation statistics evaluation"""
    args = parse_args()
    lotus.settings.configure(
        lm=LM(
            model=args.model_name,
        )
    )

    # Load dataset and setup mappings
    dataset = pd.read_csv(args.dataset_path)

    mode_results = {
        mode.value: process_mode(args, mode, input_folder, dataset)
        for mode, input_folder in zip(args.mode, args.input_folder)
    }

    results: pd.DataFrame | None = None
    for eval in args.evals:
        new_results = eval.evaluate_all(
            mode_results,
            output_dir=os.path.join(args.output_folder, eval.value),
            important_citations_path=args.important_citations_path,
            nugget_groundtruth_dir_path=args.nugget_groundtruth_dir_path,
        )
        if results is None:
            results = new_results
        else:
            results = results.merge(new_results, on="baseline_name", how="outer")

    if results is not None:
        results.to_csv(os.path.join(args.output_folder, "results.csv"), index=False)

    pretty_print_results(results)
    logger.info(f"Results saved to: {args.output_folder}")


if __name__ == "__main__":
    main()
