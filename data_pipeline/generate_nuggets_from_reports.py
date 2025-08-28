#!/usr/bin/env python3
import argparse
import json
import logging
import os
import datetime
import pandas as pd

from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.core.metrics import calculate_nugget_scores


def main():
    parser = argparse.ArgumentParser(
        description="Generate nuggets from ground truth reports"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gt_nuggets_outputs",
        help="Path to output directory",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4.1", help="Model to use for nuggetizer"
    )
    parser.add_argument(
        "--log_level",
        type=int,
        default=0,
        help="Log level (0=warning, 1=info, 2=debug)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING - 10 * args.log_level)
    logger = logging.getLogger(__name__)

    # Read the CSV file
    data_path = "../../../scraped_data/20250607_180022/papers_with_related_works.csv"
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows from CSV")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Write config file
    config_path = os.path.join(args.output_dir, "config.txt")
    with open(config_path, "w") as config_file:
        config_file.write(f"model: {args.model}\n")
        config_file.write(f"date: {datetime.datetime.now().strftime('%Y-%m-%d')}\n")

    nuggetizer = Nuggetizer(model=args.model, log_level=args.log_level)

    for i, row in df.iterrows():
        arxiv_id = row.get("arxiv_id", f"paper_{i}")
        abstract = row.get("abstract", "")
        related_works = row.get("clean_latex_related_works", "")

        if pd.isna(related_works) or related_works == "":
            logger.warning(f"Skipping row {i} - no related works content")
            continue

        # Create query and request
        query_text = f"Write a Related Works section for an academic paper given the paper's abstract. Here is the paper abstract:\n{abstract}"
        documents = [Document(docid=f"{arxiv_id}_related_works", segment=related_works)]
        query = Query(qid=arxiv_id, text=query_text)
        request = Request(query=query, documents=documents)

        logger.info(f"Processing paper {i + 1}/{len(df)}: {arxiv_id}")
        try:
            # Generate and assign nuggets
            scored_nuggets = nuggetizer.create(request)
            logger.info(f"Generated {len(scored_nuggets)} nuggets for {arxiv_id}")

            assigned_nuggets = nuggetizer.assign(
                query_text, related_works, scored_nuggets
            )

            # Filter nuggets by assignment type
            valid_nuggets = [n for n in assigned_nuggets if n.assignment == "support"]
            partial_nuggets = [
                n
                for n in assigned_nuggets
                if n.assignment in ["partial_support", "support"]
            ]

            logger.info(
                f"Kept {len(valid_nuggets)} valid nuggets out of {len(scored_nuggets)} for {arxiv_id}"
            )

            # Calculate metrics
            nugget_list = [
                {"text": n.text, "importance": n.importance, "assignment": n.assignment}
                for n in assigned_nuggets
            ]
            metrics = calculate_nugget_scores(request.query.qid, nugget_list)

            # Prepare output
            output = {
                "qid": arxiv_id,
                "query": query_text,
                "nuggets": [
                    {
                        "text": n.text,
                        "importance": n.importance,
                        "assignment": n.assignment,
                    }
                    for n in assigned_nuggets
                ],
                "supported_nuggets": [
                    {
                        "text": n.text,
                        "importance": n.importance,
                        "assignment": n.assignment,
                    }
                    for n in valid_nuggets
                ],
                "partially_supported_nuggets": [
                    {
                        "text": n.text,
                        "importance": n.importance,
                        "assignment": n.assignment,
                    }
                    for n in partial_nuggets
                ],
                "nuggets_metrics": {
                    "strict_vital_score": metrics.strict_vital_score,
                    "strict_all_score": metrics.strict_all_score,
                    "vital_score": metrics.vital_score,
                    "all_score": metrics.all_score,
                },
            }

            # Save output files
            row_id = str(i)
            row_dir = os.path.join(args.output_dir, row_id)
            os.makedirs(row_dir, exist_ok=True)

            # Save JSON
            json_path = os.path.join(row_dir, "res.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            # Save CSV
            csv_path = os.path.join(row_dir, "res.csv")
            output_df = pd.DataFrame([output])
            output_df.to_csv(csv_path, index=False)

            logger.info(f"Logged output for {row_id} to {json_path}")

        except Exception as e:
            logger.error(f"Failed to process {arxiv_id}: {e}")
            exit()

    logger.info(f"Writing output to {args.output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
