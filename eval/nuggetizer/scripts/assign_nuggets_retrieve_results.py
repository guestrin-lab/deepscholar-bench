#!/usr/bin/env python3
import argparse
import json
import logging
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


def process_candidate(
    nugget_record: Dict, candidate: Dict, nuggetizer: Nuggetizer, logger: logging.Logger
) -> Dict:
    """Process a single candidate and assign nuggets to it."""
    # Convert nuggets to Nugget objects with importance scores
    nuggets = [
        ScoredNugget(text=n["text"], importance=n.get("importance", "vital"))
        for n in nugget_record["nuggets"]
    ]
    logger.info(
        "Processing query: %s (qid: %s)",
        nugget_record.get("query", "N/A"),
        nugget_record.get("qid", "N/A"),
    )
    logger.info(
        "Assigning %d nuggets to candidate text (length: %d)",
        len(nuggets),
        len(candidate["doc"]["segment"]),
    )

    assigned_nuggets = nuggetizer.assign(
        query=nugget_record.get("query", "N/A"),
        context=candidate["doc"]["segment"],
        nuggets=nuggets,
    )

    # Create output record
    output_record = {
        "text": nugget_record["query"],
        "qid": nugget_record["qid"],
        "candidate_text": candidate["doc"]["segment"],
        "docid": candidate["docid"],
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


def get_processed_entries(output_file: str) -> set:
    """Read the output file and return a set of already processed (qid, docid) pairs."""
    processed_entries = set()
    try:
        with open(output_file, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    processed_entries.add((record["qid"], record["docid"]))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return processed_entries


def main():
    parser = argparse.ArgumentParser(
        description="Assign nuggets to retrieved candidate segments"
    )
    parser.add_argument(
        "--nugget_file", type=str, required=True, help="Path to nugget JSONL file"
    )
    parser.add_argument(
        "--retrieve_results_file",
        type=str,
        required=True,
        help="Path to retrieval results JSONL file",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4", help="Model to use for assignment"
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Logging level: 0=warnings only, 1=info, 2=debug",
    )
    parser.add_argument(
        "--use_azure_openai", action="store_true", help="Use Azure OpenAI"
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Get already processed entries
    processed_entries = get_processed_entries(args.output_file)
    logger.info("Found %d already processed entries", len(processed_entries))

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
    logger.info("Reading retrieval results file: %s", args.retrieve_results_file)
    retrieve_data = read_jsonl(args.retrieve_results_file)

    # Create mapping from qid to nugget record
    qid_to_nuggets = {record["qid"]: record for record in nugget_data}

    # Process each retrieval result
    logger.info("Processing retrieval results")

    with open(args.output_file, "a") as f:
        for i, retrieve_record in enumerate(retrieve_data, 1):
            qid = retrieve_record["query"]["qid"]
            nugget_record = qid_to_nuggets.get(qid)

            if not nugget_record:
                logger.warning(f"No nuggets found for qid {qid}")
                continue

            # Process each candidate for this query
            for candidate in retrieve_record["candidates"]:
                # Skip if already processed
                if (qid, candidate["docid"]) in processed_entries:
                    logger.info(
                        "Skipping already processed entry (qid: %s, docid: %s)",
                        qid,
                        candidate["docid"],
                    )
                    continue

                logger.info(
                    "Processing candidate %d for query %s",
                    retrieve_record["candidates"].index(candidate) + 1,
                    qid,
                )

                try:
                    processed_record = process_candidate(
                        nugget_record, candidate, nuggetizer, logger
                    )
                    f.write(json.dumps(processed_record) + "\n")
                    f.flush()  # Ensure the record is written immediately
                except Exception as e:
                    logger.error(
                        "Error processing candidate for qid %s, docid %s: %s",
                        qid,
                        candidate["docid"],
                        str(e),
                    )
                    continue

    logger.info("Processing complete")


if __name__ == "__main__":
    main()
