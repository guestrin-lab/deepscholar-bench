#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from nuggetizer.core.types import ScoredNugget
from nuggetizer.models.nuggetizer import Nuggetizer


def setup_logging(log_level: int) -> None:
    """Configure logging based on verbosity level."""
    logging_level = logging.WARNING
    if log_level >= 2:
        logging_level = logging.DEBUG
    elif log_level >= 1:
        logging_level = logging.INFO

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def read_jsonl(file_path: str) -> List[Dict]:
    """Read JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def get_run_id(file_path: str) -> str:
    """Extract run_id from the filename by dropping the .jsonl extension."""
    return Path(file_path).stem


def process_record(
    answer_record: Dict,
    nugget_record: Dict,
    run_id: str,
    nuggetizer: Nuggetizer,
    logger: logging.Logger,
) -> Dict:
    """Process records from answer and nugget files to create output record."""
    # Construct answer text by joining all answer segments
    answer_text = " ".join(a["text"] for a in answer_record["answer"])

    # Convert nuggets to Nugget objects with importance scores
    nuggets = [
        ScoredNugget(text=n["text"], importance=n.get("importance", "vital"))
        for n in nugget_record["nuggets"]
    ]
    query = nugget_record.get("query", "N/A")

    logger.info(
        "Processing query: %s (qid: %s)", query, nugget_record.get("qid", "N/A")
    )
    logger.info(
        "Assigning %d nuggets to answer text (length: %d)",
        len(nuggets),
        len(answer_text),
    )

    assigned_nuggets = nuggetizer.assign(query, answer_text, nuggets=nuggets)

    # Create output record
    output_record = {
        "query": nugget_record["query"],
        "qid": nugget_record["qid"],
        "answer_text": answer_text,
        "response_length": answer_record["response_length"],
        "run_id": run_id,
        "nuggets": [
            {"text": n.text, "importance": n.importance, "assignment": n.assignment}
            for n in assigned_nuggets
        ],
    }

    # Log assignment statistics
    assignment_stats = {
        assignment: sum(1 for n in assigned_nuggets if n.assignment == assignment)
        for assignment in ["support", "partial_support", "not_support"]
    }
    logger.info("Assignment results: %s", assignment_stats)

    return output_record


def get_processed_qids(output_file: str) -> set:
    """Read the output file and return a set of already processed qids."""
    processed_qids = set()
    try:
        with open(output_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_qids.add(record["qid"])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return processed_qids


def main():
    parser = argparse.ArgumentParser(
        description="Assign nuggets to answer text from input JSONL files"
    )
    parser.add_argument(
        "--nugget_file", type=str, required=True, help="Path to nugget JSONL file"
    )
    parser.add_argument(
        "--answer_file", type=str, required=True, help="Path to answer JSONL file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model to use for assignment"
    )
    parser.add_argument(
        "--use_azure_openai", action="store_true", help="Use Azure OpenAI"
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging level: 0=warnings only, 1=info, 2=debug",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Get already processed qids
    processed_qids = get_processed_qids(args.output_file)
    logger.info("Found %d already processed records", len(processed_qids))

    # Get run_id from answer file name
    run_id = get_run_id(args.answer_file)
    logger.info("Using run_id: %s", run_id)

    # Initialize nuggetizer (only using assigner component)
    logger.info("Initializing Nuggetizer with model: %s", args.model)
    nuggetizer = Nuggetizer(
        assigner_model=args.model,
        log_level=args.log_level,
        use_azure_openai=args.use_azure_openai,
    )

    # Read input files
    logger.info("Reading nugget file: %s", args.nugget_file)
    nugget_data = read_jsonl(args.nugget_file)
    logger.info("Reading answer file: %s", args.answer_file)
    answer_data = read_jsonl(args.answer_file)
    qid_to_answer_data = {a["topic_id"]: a for a in answer_data}

    # Process each pair of records
    logger.info("Processing %d record pairs", len(nugget_data))

    with open(args.output_file, "a") as f:
        for i, nugget_record in enumerate(nugget_data, 1):
            answer_record = qid_to_answer_data.get(nugget_record["qid"])
            if answer_record is None:
                answer_record = {
                    "answer": [],
                    "response_length": 0,
                    "qid": nugget_record["qid"],
                }
                # Default to setting each nugget to not_support
            if nugget_record["qid"] in processed_qids:
                logger.info(
                    "Skipping already processed record %s", nugget_record["qid"]
                )
                continue

            logger.info("Processing record pair %d/%d", i, len(nugget_data))
            try:
                processed_record = process_record(
                    answer_record, nugget_record, run_id, nuggetizer, logger
                )
                f.write(json.dumps(processed_record) + "\n")
                f.flush()  # Ensure the record is written immediately
            except Exception as e:
                logger.error(
                    "Error processing record %s: %s", nugget_record["qid"], str(e)
                )
                continue

    logger.info("Processing complete")


if __name__ == "__main__":
    main()
