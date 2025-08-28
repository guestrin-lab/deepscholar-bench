import argparse
import pandas as pd
import lotus
from lotus.models import LM


query_in = "Carefully read the {title}, {abstract} and {related_work_section} of an academic paper. \
    Then consider the cited paper in question, given the title {cited_paper_title}, the {cited_paper_authors} and a snippet of its content, {cited_paper_content}.\
    Is the cited paper in question an important reference?\
    An important reference reflects a notable prior work that provides key information, which a good related works section for this paper must include.\
    A non-important reference is one that could be omitted or substituted with a different related work.\
    A non-important reference may be a tangential reference, an unimportant reference.\
    Alternatively, a non-important reference may be a relevant reference that reflects an important topic area, but the particular reference could be omitted or substituted with a different related work."


def get_important_citations(
    citation_df: pd.DataFrame, related_works_df: pd.DataFrame
) -> pd.DataFrame:
    joined_df = citation_df.merge(
        related_works_df, left_on="parent_paper_title", right_on="title", how="left"
    )

    joined_df = joined_df.rename(
        columns={
            "clean_latex_related_works": "related_work_section",
            "bib_paper_authors": "cited_paper_authors",
        }
    )

    joined_df["cited_paper_content"] = joined_df.apply(
        lambda row: row["cited_paper_abstract"]
        if row.get("is_arxiv_paper", False)
        else row.get("search_res_content", None),
        axis=1,
    )
    res = joined_df.sem_filter(query_in)
    print_stats(res, joined_df)

    return res


def print_stats(df: pd.DataFrame, joined_df: pd.DataFrame):
    print(f"Total num cites: {len(joined_df)}")
    print(
        f"Total num important cites: {len(df)} ({len(df) / len(joined_df) * 100:.2f}%)"
    )
    print(
        f"Total num important arxiv cites: {len(df[df.is_arxiv_paper])} ({len(df[df.is_arxiv_paper]) / len(df) * 100:.2f}%)"
    )
    print(
        f"Mean citations per paper: {df.groupby('parent_paper_title').agg({'cited_paper_title': 'count'}).mean()}"
    )
    print(
        f"Mean citations per paper: {df.groupby('parent_paper_title').agg({'cited_paper_title': 'count'}).mean()}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--citation_input_file",
        type=str,
        required=True,
        help="Path to the citation file",
    )

    parser.add_argument(
        "--related_works_input_file",
        type=str,
        required=True,
        help="Path to the related works file",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM model to use",
        default="gpt-4o",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output file",
    )

    args = parser.parse_args()

    lotus.settings.configure(lm=LM(model=args.model))

    citation_df = pd.read_csv(args.citation_input_file)
    related_works_df = pd.read_csv(args.related_works_input_file)
    important_citations_df = get_important_citations(citation_df, related_works_df)
    important_citations_df.to_csv(args.output_file, index=False)
