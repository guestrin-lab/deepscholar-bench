import pandas as pd
import time
from typing import Dict, List
from tavily import TavilyClient
import arxiv
import tenacity
import os
import argparse

tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))
arxiv_client = arxiv.Client()


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
def search_tavily(title: str) -> Dict[str, str]:
    """
    Search Tavily API for a title and return title, url, and content of the first result
    """
    response = tavily_client.search(query=title, max_results=1)
    return {
        "original_title": title,
        "search_res_title": response["results"][0]["title"],
        "search_res_url": response["results"][0]["url"],
        "search_res_content": response["results"][0]["content"],
    }


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
)
def search_arxiv_api(title: str) -> Dict[str, str] | None:
    """
    Search arXiv API for a title and return title, url, and content of the first result
    """
    search = arxiv.Search(
        query=f'ti:"{title}"',  # Search only in title field
        max_results=1,
    )

    try:
        # Get first result if any exist
        results = list(arxiv_client.results(search))
        normalized_result_title = (
            results[0].title.lower().replace(" ", "").replace("-", "").replace("_", "")
        )
        normalized_title = (
            title.lower().replace(" ", "").replace("-", "").replace("_", "")
        )
        if normalized_result_title == normalized_title:
            return {
                "original_title": title,
                "search_res_title": results[0].title,
                "search_res_url": results[0].pdf_url,
                "search_res_content": results[0].summary,
            }
        else:
            return None
    except Exception as _:
        return None


def save_results(
    results: List[Dict[str, str]], df: pd.DataFrame, output_file_path: str
):
    if not results:
        return
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file_path, index=False, mode="a")


def process_dataset(csv_file_path: str, output_file_path: str):
    """
    Process the dataset and find arXiv matches.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    if os.path.exists(output_file_path):
        existing_df: pd.DataFrame = pd.read_csv(output_file_path)
    else:
        existing_df = pd.DataFrame(columns=df.columns)

    results = []
    for index, row in df.iterrows():
        title = row["cited_paper_title"]  # Assuming 'title' is the column name
        print(f"Processing row {index}: {title[:50]}...")
        if title in existing_df["cited_paper_title"].values:
            existing_row = existing_df[(existing_df["cited_paper_title"] == title)]
            results.append(
                {
                    **row.to_dict(),
                    "original_title": title,
                    "search_res_title": existing_row["search_res_title"].values[0],
                    "search_res_url": existing_row["search_res_url"].values[0],
                    "search_res_content": existing_row["search_res_content"].values[0],
                }
            )
            print(
                f"Skipping {title[:50]} because it already exists in the output file..."
            )
            continue
        # Search arXiv API
        api_match = search_arxiv_api(title)
        if api_match is None:
            print(f"No arXiv match found for {title[:50]}...")
            api_match = search_tavily(title)
            if api_match is None:
                print(f"No match found for {title[:50]}...")
                continue
            else:
                print(f"Tavily match found for {title[:50]}...")
        else:
            print(f"arXiv match found for {title[:50]}...")
        results.append({**row.to_dict(), **api_match})

        if index % 100 == 0:
            save_results(results, df, output_file_path)
            results = []
        # Add a small delay to be respectful to the API
        time.sleep(1)

    save_results(results, df, output_file_path)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate the dataframe based on the cited_paper_title and parent_paper_title columns
    """
    df = df.drop_duplicates(
        subset=["cited_paper_title", "parent_paper_title"], keep="last"
    )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    csv_file_path = args.input_file
    output_file_path = args.output_file

    process_dataset(csv_file_path, output_file_path)
    # df = pd.read_csv(output_file_path)
    # df = deduplicate(df)
    # df.to_csv(output_file_path, index=False)
