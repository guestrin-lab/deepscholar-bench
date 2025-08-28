#!/usr/bin/env python3
import argparse
import json
from typing import Dict, List

from nuggetizer.core.metrics import calculate_nugget_scores, calculate_global_metrics


def read_jsonl(file_path: str) -> List[Dict]:
    """Read JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def process_records(records: List[Dict]) -> List[Dict]:
    """Process each record to add metrics."""
    for record in records:
        metrics = calculate_nugget_scores(record["qid"], record["nuggets"])
        record["metrics"] = {
            "qid": metrics.qid,
            "strict_vital_score": metrics.strict_vital_score,
            "strict_all_score": metrics.strict_all_score,
            "vital_score": metrics.vital_score,
            "all_score": metrics.all_score,
        }
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Calculate metrics for nugget assignments"
    )
    parser.add_argument(
        "--input_file", type=str, help="Path to input JSONL file with assignments"
    )
    parser.add_argument("--output_file", type=str, help="Path to output JSONL file")
    args = parser.parse_args()

    # Read input data
    records = read_jsonl(args.input_file)

    # Calculate per-response metrics
    processed_records = process_records(records)

    # Calculate global metrics
    global_metrics = calculate_global_metrics(records)

    # Write output with metrics
    with open(args.output_file, "w") as f:
        # Write per-response metrics
        for record in processed_records:
            f.write(json.dumps(record["metrics"]) + "\n")
        print(global_metrics)
        # Write global metrics as final line
        f.write(json.dumps(global_metrics) + "\n")


if __name__ == "__main__":
    main()
