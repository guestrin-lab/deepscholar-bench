from lotus import WebSearchCorpus
import pandas as pd
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import time
from lotus import web_search
from datetime import datetime, timedelta
from exa_py import Exa

try:
    from deepscholar_base.utils.summary_generation import generate_section_summary
    from deepscholar_base.utils.prompts import (
        web_multiquery_system_prompt,
        arxiv_multiquery_system_prompt,
        multiquery_user_prompt,
        background_summarization_instructions,
    )
    from deepscholar_base.configs import Configs
except ImportError:
    from ..utils.summary_generation import generate_section_summary
    from ..utils.prompts import (
        web_multiquery_system_prompt,
        arxiv_multiquery_system_prompt,
        multiquery_user_prompt,
        background_summarization_instructions,
    )
    from ..configs import Configs

async def recursive_search(
    configs: Configs,
    topic: str,
    end_date: datetime | None = None,
) -> tuple[list[str], pd.DataFrame, str]:
    # Run search for arxiv
    results = None
    queries = []
    background = ""
    for i in range(configs.num_search_steps):
        configs.logger.info(f"Running search step {i+1} of {configs.num_search_steps}")
        try:
            queries, results = await _multiquery_search(
                arxiv_multiquery_system_prompt,
                topic,
                background,
                [WebSearchCorpus.ARXIV],
                configs,
                end_date,
            )
        except Exception as e:
            configs.logger.error(f"Error searching for arxiv: {e}")
        if configs.enable_web_search:
            web_queries, web_results = await _multiquery_search(
                web_multiquery_system_prompt,
                topic,
                background,
                configs.web_corpuses,
                configs,
                end_date,
            )
            queries.extend(web_queries)
            if results is None:
                results = web_results
            else:
                results = pd.concat([results, web_results])
                results.fillna("", inplace=True)
        if configs.enable_exa_search:
            exa_queries, exa_results = await _exa_multiquery_search(
                web_multiquery_system_prompt,
                topic,
                background,
                configs,
                end_date,
            )
            queries.extend(exa_queries)
            if results is None:
                results = exa_results
            elif exa_results is not None and not exa_results.empty:
                results = pd.concat([results, exa_results])
                results.fillna("", inplace=True)
        if results is not None:
            results = results.drop_duplicates(subset=["url"])
        else:
            configs.logger.error(f"No results found for search step {i+1} of {configs.num_search_steps}")
            continue
        background = await generate_section_summary(
            topic, 
            results, 
            background_summarization_instructions, 
            background,
            lm=configs.search_lm,
        )
        configs.logger.info(f"Completed search step {i+1} of {configs.num_search_steps}, found {len(results)} results")
    return queries, results, background


########### Search ###########
async def _multiquery_search(
    instruction: str,
    topic: str,
    background: str,
    corpuses: list[WebSearchCorpus],
    configs: Configs,
    end_date: datetime | None = None,
) -> tuple[list[str], pd.DataFrame]:
    queries = await _generate_queries(
        topic, background, instruction, end_date, configs
    )
    configs.logger.info(f"Searching {corpuses} for queries: {queries}")
    results = await _safe_lotus_async_search(configs, queries, configs.per_query_max_search_results_count, corpuses, end_date=end_date)
    return queries, results


async def _safe_lotus_async_search(
    configs: Configs,
    queries: list[str],
    K: int,
    corpuses: list[WebSearchCorpus],
    sort_by_date: bool = False,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    """
    Perform asynchronous searches on the specified corpuses for a list of queries, returning the combined results as a DataFrame.

    Args:
        configs (Configs): Application configs containing logger and configurations.
        queries (list[str]): List of query strings to search for.
        K (int): Number of results to retrieve per query/per corpus.
        corpuses (list[WebSearchCorpus]): List of corpuses to search (e.g., ARXIV, GOOGLE, etc.).
        sort_by_date (bool, optional): Whether to sort results by date. Effective only if `end_date` is provided.
        end_date (datetime | None, optional): Results after this date will be excluded when provided.

    Returns:
        pd.DataFrame: A concatenated DataFrame of search results, deduplicated by URL, with columns ["title", "url", "snippet", "query", "context", "date"].

    Notes:
        - Retries up to 5 times per query/corpus combination on errors.
        - Applies per-source renaming and post-processing for DataFrame compatibility.
        - The "context" column is generated from the title, url, and snippet of each result.
        - Results are deduplicated by the "url" column.
    """
    sort_by_date = sort_by_date and end_date is not None
    search_tasks = [(query, corpus) for query in queries for corpus in corpuses]

    dfs = await asyncio.gather(
        *[_process_single_lotus_search_task(configs, query, corpus, K, sort_by_date, end_date) 
          for query, corpus in search_tasks]
    )

    if dfs:
        result_df = pd.concat(dfs, ignore_index=True).fillna("")
    else:
        result_df = pd.DataFrame()

    if not result_df.empty and "url" in result_df.columns:
        result_df = result_df.drop_duplicates(subset=["url"])

    return result_df

async def _process_single_lotus_search_task(
    configs: Configs, 
    query: str, 
    corpus: WebSearchCorpus, 
    K: int, 
    sort_by_date: bool = False, 
    end_date: datetime | None = None
) -> pd.DataFrame:
    original_query = query

    # Special handling for ARXIV query formatting
    if corpus == WebSearchCorpus.ARXIV:
        split_queries = [q.strip() for q in query.split("AND")]
        if len(split_queries) > 1:
            split_queries = [f"all:{q}" for q in split_queries]
        else:
            split_queries = [f"{query.strip()}"]
        query = " AND ".join(split_queries)

    configs.logger.debug(
        f"Lotus search with query: {query}, corpus: {corpus}, sort_by_date: {sort_by_date}, end_date: {end_date}"
    )

    count = 5
    df = None
    while count > 0:
        try:
            df = web_search(corpus, query, K, sort_by_date=sort_by_date)
            break
        except Exception as e:
            configs.logger.error(
                f"Error searching for query {query}, attempt {6 - count}/5: {e}"
            )
            time.sleep(1)
            count -= 1
            df = None

    if df is None or df.empty:
        return pd.DataFrame(columns=["title", "url", "snippet", "query", "context", "date"])

    # Harmonize column names per corpus
    if corpus == WebSearchCorpus.ARXIV:
        df.rename(
            columns={"abstract": "snippet", "link": "url", "published": "date"},
            inplace=True,
        )
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
    elif corpus in [WebSearchCorpus.GOOGLE, WebSearchCorpus.GOOGLE_SCHOLAR]:
        df.rename(columns={"link": "url"}, inplace=True)
    elif corpus == WebSearchCorpus.BING:
        df.rename(columns={"name": "title"}, inplace=True)
    elif corpus == WebSearchCorpus.TAVILY:
        df.rename(columns={"content": "snippet"}, inplace=True)

    df["query"] = original_query

    # Filter by date if applicable
    if end_date is not None and "date" in df.columns:
        try:
            df["_date"] = pd.to_datetime(df["date"], errors="coerce")
            cutoff_date = end_date - timedelta(days=1)
            df = df[df["_date"].dt.date <= cutoff_date.date()]
            df.drop(columns=["_date"], inplace=True)
        except Exception as e:
            configs.logger.error(f"Error processing date: {e}")

    # Create context string for each row
    def generate_context(row):
        return f"{row.get('title', '')}[{row.get('url', '')}]: {row.get('snippet', '')}"
    df["context"] = df.apply(generate_context, axis=1)

    # Ensure required columns exist, filling with empty strings if missing
    required_columns = ["title", "url", "snippet", "query", "context", "date"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    # Final integrity check
    missing = [col for col in required_columns if col not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    return df


########### Exa Search ###########
async def _exa_multiquery_search(
    instruction: str,
    topic: str,
    background: str,
    configs: Configs,
    end_date: datetime | None = None,
) -> tuple[list[str], pd.DataFrame]:
    queries = await _generate_queries(
        topic, background, instruction, end_date, configs
    )
    configs.logger.info(f"Searching Exa for queries: {queries}")
    results = await _safe_exa_async_search(configs, queries, configs.per_query_max_search_results_count, end_date=end_date)
    return queries, results


async def _safe_exa_async_search(
    configs: Configs,
    queries: list[str],
    K: int,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    dfs = await asyncio.gather(
        *[_process_single_exa_search(configs, query, K, end_date) for query in queries]
    )
    if dfs:
        result_df = pd.concat(dfs, ignore_index=True).fillna("")
    else:
        result_df = pd.DataFrame()
    if not result_df.empty and "url" in result_df.columns:
        result_df = result_df.drop_duplicates(subset=["url"])
    return result_df


async def _process_single_exa_search(
    configs: Configs,
    query: str,
    K: int,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    required_columns = ["title", "url", "snippet", "query", "context", "date"]
    count = 5
    while count > 0:
        try:
            exa = Exa()
            search_kwargs = {
                "query": query,
                "num_results": K,
                "type": "auto",
                "category": "research paper",
            }
            if end_date:
                search_kwargs["end_published_date"] = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            results = await asyncio.get_event_loop().run_in_executor(
                None, lambda: exa.search(**search_kwargs)
            )
            rows = []
            for r in results.results:
                rows.append({
                    "title": r.title or "",
                    "url": r.url or "",
                    "snippet": r.text if hasattr(r, "text") and r.text else "",
                    "date": r.published_date or "",
                    "query": query,
                })
            if not rows:
                return pd.DataFrame(columns=required_columns)
            df = pd.DataFrame(rows)
            df["context"] = df.apply(
                lambda row: f"{row.get('title', '')}[{row.get('url', '')}]: {row.get('snippet', '')}", axis=1
            )
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ""
            if end_date is not None and "date" in df.columns:
                try:
                    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
                    cutoff_date = end_date - timedelta(days=1)
                    df = df[df["_date"].dt.date <= cutoff_date.date()]
                    df.drop(columns=["_date"], inplace=True)
                except Exception as e:
                    configs.logger.error(f"Error processing Exa date: {e}")
            return df
        except Exception as e:
            configs.logger.error(f"Error in Exa search for query {query}, attempt {6 - count}/5: {e}")
            time.sleep(1)
            count -= 1
    return pd.DataFrame(columns=required_columns)


########### Generate Queries ###########
class Queries(BaseModel):
    queries: list[str] = Field(
        description="list of search queries.",
    )

async def _generate_queries(
    topic: str,
    background: str,
    instruction: str,
    end_date: datetime | None,
    configs: Configs,
) -> list[str]:
    end_date = end_date or datetime.now()
    sys_instruction = instruction.format(number_of_queries=configs.num_search_queries_per_step_per_corpus, end_date=end_date.strftime("%Y%m%d%H%M"))
    user_prompt = multiquery_user_prompt.format(topic=topic, background=background)
    if configs.use_structured_output:
        queries = configs.search_lm.get_completion(
            sys_instruction, user_prompt, response_format=Queries
        )
        assert isinstance(queries, Queries)
    else:
        queries_str = configs.search_lm.get_completion(
            sys_instruction, user_prompt
        )
        queries = Queries(queries=_split_queries(queries_str))
    configs.logger.info(f"queries: {queries.queries}")
    return queries.queries

def _split_queries(queries: str) -> list[str]:
    """
    Splits a string containing multiple queries into a list of queries.
    Tries to split by newlines, or by other common separators if needed.
    """
    if not queries or not isinstance(queries, str):
        return []
    # Try splitting by newlines first
    split = [q.strip() for q in queries.splitlines() if q.strip()]
    if len(split) > 1:
        all_queries = [_split_queries(q) for q in split]
        return [q for sublist in all_queries for q in sublist]
    # Try splitting by '\n' (in case it's a literal string)
    if "\\n" in queries:
        split = [q.strip() for q in queries.split("\\n") if q.strip()]
        if len(split) > 1:
            all_queries = [_split_queries(q) for q in split]
            return [q for sublist in all_queries for q in sublist]
    # Try splitting by numbered list (e.g., "1. query", "2. query")
    import re

    numbered = re.split(r"\n?\s*\d+\.\s+", queries)
    numbered = [q.strip() for q in numbered if q.strip()]
    if len(numbered) > 1:
        all_queries = [_split_queries(q) for q in numbered]
        return [q for sublist in all_queries for q in sublist]
    # Fallback: return as single query in list
    return [queries.strip()]
