#!/usr/bin/env python3
import argparse
import json
import logging
from typing import Dict, List, Optional

from nuggetizer.core.types import Query, Document, Request
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


def process_input_record(record: Dict) -> Request:
    """Convert input JSON record to Request object."""
    query = Query(qid=record["query"]["qid"], text=record["query"]["text"])

    documents = []
    for candidate in record["candidates"]:
        doc = Document(docid=candidate["docid"], segment=candidate["doc"]["segment"])
        if "judgment" not in candidate or (
            "judgment" in candidate and candidate["judgment"] > 0
        ):
            documents.append(doc)

    return Request(query=query, documents=documents)


class ScoredNugget:
    text: str
    importance: str
    assignment: Optional[str] = None


def format_output(request: Request, scored_nuggets: List[ScoredNugget]) -> Dict:
    """Format output according to required schema."""
    return {
        "query": request.query.text,
        "qid": request.query.qid,
        "nuggets": [
            {"text": n.text, "importance": n.importance} for n in scored_nuggets
        ],
    }


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
        description="Extract and score nuggets from input JSONL file"
    )
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to output JSONL file"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o", help="Model to use for all operations"
    )
    parser.add_argument(
        "--creator_model", type=str, help="Model to use for nugget creation"
    )
    parser.add_argument(
        "--scorer_model", type=str, help="Model to use for nugget scoring"
    )
    parser.add_argument("--window_size", type=int, help="Window size for processing")
    parser.add_argument(
        "--max_nuggets", type=int, help="Maximum number of nuggets to extract"
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

    # Get already processed qids
    processed_qids = get_processed_qids(args.output_file)
    logger.info("Found %d already processed records", len(processed_qids))

    # Initialize nuggetizer with all configurations
    logger.info("Initializing Nuggetizer")
    nuggetizer_kwargs = {
        "log_level": args.log_level,
        "use_azure_openai": args.use_azure_openai,
    }

    if args.creator_model or args.scorer_model:
        nuggetizer_kwargs.update(
            {
                "creator_model": args.creator_model or args.model,
                "scorer_model": args.scorer_model or args.model,
            }
        )
    else:
        nuggetizer_kwargs["model"] = args.model

    if args.window_size:
        nuggetizer_kwargs["window_size"] = args.window_size
    if args.max_nuggets:
        nuggetizer_kwargs["max_nuggets"] = args.max_nuggets

    nuggetizer = Nuggetizer(**nuggetizer_kwargs)

    # Process each record
    logger.info("Reading input file: %s", args.input_file)
    input_data = read_jsonl(args.input_file)

    logger.info("Processing %d records", len(input_data))
    with open(args.output_file, "a") as f:
        for i, record in enumerate(input_data, 1):
            if record["query"]["qid"] in processed_qids:
                logger.info(
                    "Skipping already processed record %s", record["query"]["qid"]
                )
                continue

            logger.info("Processing record %d/%d", i, len(input_data))
            try:
                request = process_input_record(record)
                scored_nuggets = nuggetizer.create(request)
                output_record = format_output(request, scored_nuggets)
                f.write(json.dumps(output_record) + "\n")
                f.flush()
                logger.info(
                    "Generated %d nuggets for record %d", len(scored_nuggets), i
                )
            except Exception as e:
                logger.error(
                    "Error processing record %s: %s", record["query"]["qid"], str(e)
                )
                continue

    logger.info("Processing complete")


if __name__ == "__main__":
    main()
