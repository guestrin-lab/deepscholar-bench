import asyncio
from enum import Enum
from typing import Callable, Coroutine, Any
from datetime import datetime
from agents import function_tool, RunContextWrapper, RunConfig
from openai.types.responses import ResponseInputItemParam
from agents.models.chatcmpl_converter import Converter
from dataclasses import dataclass
from agents import Agent, Runner
from agents.run import ModelInputData, CallModelData
from agents.util._json import _to_dump_compatible
import pandas as pd
from lotus.types import LMStats
import logging
from lotus.models import LM
from openai import AsyncOpenAI
from agents import OpenAIResponsesModel, OpenAIChatCompletionsModel, ModelSettings
from openai.types.shared import Reasoning
import re
from lotus import web_search, web_extract, WebSearchCorpus
from exa_py import Exa

try:
    from deepscholar_base.utils.prompts import (
        openai_sdk_arxiv_search_system_prompt,
        openai_sdk_arxiv_search_system_prompt_without_cutoff,
        openai_sdk_search_system_prompt,
        openai_sdk_search_system_prompt_without_cutoff,
        openai_sdk_exa_search_system_prompt,
        openai_sdk_exa_search_system_prompt_without_cutoff,
    )
    from deepscholar_base.configs import Configs
except ImportError:
    from ..utils.prompts import (
        openai_sdk_arxiv_search_system_prompt,
        openai_sdk_arxiv_search_system_prompt_without_cutoff,
        openai_sdk_search_system_prompt,
        openai_sdk_search_system_prompt_without_cutoff,
        openai_sdk_exa_search_system_prompt,
        openai_sdk_exa_search_system_prompt_without_cutoff,
    )
    from ..configs import Configs

arxiv_logger = logging.getLogger("arxiv")
arxiv_logger.setLevel(logging.WARNING)

@dataclass
class AgentContext:
    configs: Configs
    end_date: datetime | None  # YYYYMMDDhhmm (UTC)
    papers_df: pd.DataFrame | None
    queries: list[list[str]]
    
    def merge_papers_df(self, new: pd.DataFrame) -> None:
        """Deduplicate entries by url; override old rows with new if url matches."""
        if self.papers_df is None or len(self.papers_df) == 0:
            self.papers_df = new.copy()
        url_map = {row["url"]: row for row in self.papers_df.to_dict(orient="records")}
        info = new.to_dict(orient="records")
        info = [{**url_map[row["url"]], **row} for row in info if row["url"] in url_map]
        merged = pd.concat([self.papers_df, pd.DataFrame(info)], ignore_index=True)
        self.papers_df = merged.drop_duplicates(subset=["url"])


# ---------- Search Functions ----------
class ToolTypes(Enum):
    ARXIV = "arxiv"
    WEB = "web"
    EXA = "exa"
    
    def to_web_search_corpus(self) -> WebSearchCorpus:
        if self == ToolTypes.ARXIV:
            return WebSearchCorpus.ARXIV
        elif self == ToolTypes.WEB:
            return WebSearchCorpus.TAVILY
        else:
            raise ValueError(f"Invalid search type: {self}")
        
    def to_rename_map(self) -> dict[str, str]:
        if self == ToolTypes.ARXIV:
            return {"link": "url", "abstract": "snippet", "published": "date"}
        elif self == ToolTypes.WEB:
            return {"content": "snippet"}
        elif self == ToolTypes.EXA:
            return {}
        else:
            raise ValueError(f"Invalid search type: {self}")


def _normalize_search_df(
    df: pd.DataFrame,
    query: str,
    rename_map: dict,
    results_fmt_func: Callable[[pd.Series], str],
    empty_result: str,
) -> tuple[str, pd.DataFrame]:
    if df.empty:
        return empty_result, pd.DataFrame(columns=["title", "url", "snippet", "query", "context", "date"])

    df = df.rename(columns=rename_map)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    required_columns = ["title", "url", "snippet", "query", "context", "date"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
    df["query"] = query
    df["context"] = df.apply(lambda row: f"{row.get('title', '')}[{row.get('url', '')}]: {row.get('snippet', '')}", axis=1)
    results_section = "\n".join([results_fmt_func(row) for _, row in df.iterrows()])
    return (results_section if results_section else empty_result), df

async def _handle_one_search_query(
    ctx: RunContextWrapper[AgentContext],
    search_type: ToolTypes,
    i: int,
    cutoff: datetime | None,
    query: str,
) -> tuple[str, str | None]:
    successful_query = None
    try:
        web_search_df = web_search(
            corpus=search_type.to_web_search_corpus(),
            query=query,
            K=ctx.context.configs.per_query_max_search_results_count,
            end_date=cutoff,
        )
        query_results, df = _normalize_search_df(
            web_search_df,
            query,
            rename_map=search_type.to_rename_map(),
            results_fmt_func=lambda row: f"{row.get('title', 'Untitled')} ({row.get('date', '')}): {row.get('url', '')}",
            empty_result="No results found.",
        )
        ctx.context.merge_papers_df(df)
        successful_query = query
    except Exception as e:
        ctx.context.configs.logger.error(f"Error searching {search_type.value} for query {query}: {e}")
        query_results = f"Error searching {search_type.value} for query {query}: {e}"
    query_section = f"=== QUERY {i}: {query} ===\n{query_results}"
    return query_section, successful_query


async def _search(
    ctx: RunContextWrapper[AgentContext], search_type: ToolTypes, queries: list[str]
) -> str:
    ctx.context.configs.logger.info(f"Searching {search_type.value} for queries: {queries}")
    cutoff = ctx.context.end_date
    all_results_sections = await asyncio.gather(
        *[
            _handle_one_search_query(
                ctx, search_type, i, cutoff, query
            )
            for i, query in enumerate(queries, 1)
        ]
    )
    successful_queries = [f"{search_type.value}_search"] + [
        successful_query
        for _, successful_query in all_results_sections
        if successful_query is not None
    ]
    ctx.context.configs.logger.info(f"Successful queries: {successful_queries}, collected total references: {len(ctx.context.papers_df)}")
    all_results = [query_section for query_section, _ in all_results_sections]
    ctx.context.queries.append(successful_queries)
    return "\n\n".join(all_results)

@function_tool
async def search_arxiv(ctx: RunContextWrapper[AgentContext], queries: list[str]) -> str:
    """
    Search arXiv for literature that matches the provided queries.
    Your query will be passed to the arXiv API, so it should be in the format of the arXiv API syntax.

    Returns up to 25 entries per query, with clear separation showing which results
    correspond to which query. Each entry is formatted as
    "Title (YYYY-MM-DD): https://arxiv.org/abs/<id>".
    Guidelines for Constructing Effective arXiv Search Queries:
    - Use multiple, concise, and focused queries targeting distinct aspects such as:
      key methods, technologies, canonical author names, and major terms central to the target paper.
    - Design queries to maximize breadth and relevance, not just quantity—use at least two different queries
      to ensure comprehensive coverage of prior work.
    - Leverage synonyms or alternative terms to broaden coverage, but avoid redundant or overly broad queries.
    - Do NOT include the word "arXiv" in any search string.

    Args:
        queries: A list of arXiv query strings using the arXiv API syntax.
            Example: ["attention mechanisms", "transformer architecture", "neural machine translation"]
    """
    return await _search(ctx, ToolTypes.ARXIV, queries)

@function_tool
async def search_web(ctx: RunContextWrapper[AgentContext], queries: list[str]) -> str:
    """
    Search the web for literature that matches the provided queries.
    Uses lotus web_search function with Tavily corpus.

    Returns up to 10 entries per query, with clear separation showing which results correspond to which query.
    Each entry is formatted as "Title: URL".
    Guidelines for Constructing Effective Web Search Queries:
    - Use multiple, concise, and focused queries targeting distinct aspects such as:
      key methods, technologies, canonical author names, and major terms central to the target paper.
    - Design queries to maximize breadth and relevance, not just quantity—use at least two different queries to ensure comprehensive coverage of prior work.
    - Leverage synonyms or alternative terms to broaden coverage, but avoid redundant or overly broad queries.
    - Keep the queries under 400 characters.

    Args:
        queries: A list of web search query strings.
            Example: ["attention mechanisms", "transformer architecture", "neural machine translation"]
    """
    return await _search(ctx, ToolTypes.WEB, queries)

# ---------- Generic "Read" Extraction Helpers ----------
def _extract_contents(
    configs: Configs,
    keys: list[str],
    display_fmt: Callable[[str, pd.DataFrame], str],
    corpus: WebSearchCorpus,
    error_prefix: str,
) -> tuple[str, pd.DataFrame]:
    extracted_dfs = []
    results_text = []
    for key in keys:
        try:
            extract_df = web_extract(corpus=corpus, doc_id=key)
            # Defensive
            if not extract_df.empty and extract_df.iloc[0].get("full_text"):
                display_text = display_fmt(key, extract_df)
                results_text.append(display_text)
                row = {
                    "title": "",
                    "url": extract_df.iloc[0]["url"],
                    "snippet": extract_df.iloc[0]["full_text"],
                    "query": f"{corpus.value}_read",
                    "context": f"[{key}]: {extract_df.iloc[0]['full_text'][:1000]}",
                    "date": "",
                }
                extracted_dfs.append(pd.DataFrame([row]))
            else:
                results_text.append(f"{key}: Error extracting content")
        except Exception as e:
            configs.logger.error(f"{error_prefix} from {key}: {e}")
            results_text.append(f"{key}: Error extracting content")
    if extracted_dfs:
        result_df = pd.concat(extracted_dfs, ignore_index=True)
    else:
        result_df = pd.DataFrame(columns=["title", "url", "snippet", "query", "context", "date"])
    answer = "\n\n---\n\n".join(results_text) if results_text else (f"No valid results found." if corpus==WebSearchCorpus.ARXIV else "No content found")
    return answer, result_df

# ---------- Read Tool Extraction Implementations ----------
async def _read_content(
    ctx: RunContextWrapper[AgentContext], tool_type: ToolTypes, input: list[str]
) -> str:
    ctx.context.configs.logger.info(f"Reading content from {tool_type.value} for input: {input}")
    cutoff = ctx.context.end_date
    successful_inputs = [f"{tool_type.value}_read"]
    try:
        answer, df = _extract_contents(
            ctx.context.configs,
            input,
            display_fmt=lambda pid, extract_df: f"{extract_df.iloc[0]['url']}\n\n{extract_df.iloc[0]['full_text'].strip()[:1000]}",
            corpus=tool_type.to_web_search_corpus(),
            error_prefix=f"Error extracting {tool_type.value} content",
        )
        if df is not None and not df.empty:
            ctx.context.merge_papers_df(df)
            successful_inputs.extend(input)
    except Exception as e:
        answer = f"Error extracting content from {tool_type.value} for {input}: {e}"
    ctx.context.configs.logger.info(f"Successful inputs: {successful_inputs}, collected total references: {len(ctx.context.papers_df)}")
    ctx.context.queries.append(successful_inputs)
    return answer

@function_tool
async def read_arxiv_abstracts(
    ctx: RunContextWrapper[AgentContext], paper_ids: list[str]
) -> str:
    """
    Retrieve the titles and abstracts for a list of arXiv papers.
    There may be a cutoff automatically enforced server-side so works released after the target paper's release date may not be included.

    Use this after `search_arxiv` surfaces promising identifiers to extract
    verifiable details for synthesis. The output includes each paper's title,
    followed by a blank line and the abstract text, separated by ---.

    Args:
        paper_ids: A list of arXiv identifiers, such as ["2408.14717", "2407.11418v3"].
    """
    return await _read_content(ctx, ToolTypes.ARXIV, paper_ids)

@function_tool
async def read_webpage_full_text(
    ctx: RunContextWrapper[AgentContext], urls: list[str]
) -> str:
    """
    Retrieve the full text of a list of web pages.
    Use this after `search_web` surfaces promising URLs to extract
    verifiable details for synthesis. The output includes each webpage's URL,
    followed by a blank line and the full text of the webpage, separated by ---.

    Args:
        urls: A list of web page URLs. Example: ["https://www.google.com", "https://www.wikipedia.org"]
    """
    return await _read_content(ctx, ToolTypes.WEB, urls)


# ---------- Exa Search Functions ----------
def _exa_search(
    queries: list[str],
    max_results: int = 10,
    end_date: datetime | None = None,
) -> pd.DataFrame:
    exa = Exa()
    all_results = []
    for query in queries:
        try:
            search_kwargs = {
                "query": query,
                "num_results": max_results,
                "type": "auto",
                "category": "research paper",
            }
            if end_date:
                search_kwargs["end_published_date"] = end_date.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            results = exa.search(**search_kwargs)
            for r in results.results:
                all_results.append({
                    "title": r.title or "",
                    "url": r.url or "",
                    "snippet": r.text if hasattr(r, "text") and r.text else "",
                    "date": r.published_date or "",
                    "query": query,
                })
        except Exception:
            continue
    if not all_results:
        return pd.DataFrame(columns=["title", "url", "snippet", "date", "query"])
    return pd.DataFrame(all_results)


def _exa_get_contents(
    urls: list[str],
) -> pd.DataFrame:
    exa = Exa()
    all_results = []
    try:
        results = exa.get_contents(urls, text=True)
        for r in results.results:
            all_results.append({
                "title": r.title or "",
                "url": r.url or "",
                "full_text": r.text if hasattr(r, "text") and r.text else "",
            })
    except Exception:
        pass
    if not all_results:
        return pd.DataFrame(columns=["title", "url", "full_text"])
    return pd.DataFrame(all_results)


@function_tool
async def search_exa(ctx: RunContextWrapper[AgentContext], queries: list[str]) -> str:
    """
    Search for academic papers and research using Exa's neural search engine.
    Exa excels at finding research papers, technical blog posts, and scholarly content
    that may not be indexed on arXiv.

    Returns up to 10 entries per query, with clear separation showing which results
    correspond to which query. Each entry is formatted as "Title (date): URL".

    Guidelines for Constructing Effective Exa Search Queries:
    - Use natural language queries that describe the concept you are looking for.
    - Be specific about the research area, methodology, or finding.
    - Use different queries to cover distinct aspects of the topic.

    Args:
        queries: A list of search query strings.
            Example: ["retrieval augmented generation for question answering", "dense passage retrieval methods"]
    """
    ctx.context.configs.logger.info(f"Searching Exa for queries: {queries}")
    cutoff = ctx.context.end_date
    all_results_sections = []
    successful_queries_list = []

    exa_df = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: _exa_search(
            queries,
            max_results=ctx.context.configs.per_query_max_search_results_count,
            end_date=cutoff,
        ),
    )

    for query in queries:
        query_df = exa_df[exa_df["query"] == query] if not exa_df.empty else pd.DataFrame()
        if query_df.empty:
            all_results_sections.append(f"=== QUERY: {query} ===\nNo results found.")
            continue

        query_df["context"] = query_df.apply(
            lambda row: f"{row.get('title', '')}[{row.get('url', '')}]: {row.get('snippet', '')}", axis=1
        )
        required_columns = ["title", "url", "snippet", "query", "context", "date"]
        for col in required_columns:
            if col not in query_df.columns:
                query_df[col] = ""

        ctx.context.merge_papers_df(query_df[required_columns])
        successful_queries_list.append(query)
        results_text = "\n".join(
            f"{row.get('title', 'Untitled')} ({row.get('date', '')}): {row.get('url', '')}"
            for _, row in query_df.iterrows()
        )
        all_results_sections.append(f"=== QUERY: {query} ===\n{results_text}")

    ctx.context.queries.append(["exa_search"] + successful_queries_list)
    ctx.context.configs.logger.info(
        f"Exa search successful queries: {successful_queries_list}, "
        f"collected total references: {len(ctx.context.papers_df) if ctx.context.papers_df is not None else 0}"
    )
    return "\n\n".join(all_results_sections)


@function_tool
async def read_exa_contents(
    ctx: RunContextWrapper[AgentContext], urls: list[str]
) -> str:
    """
    Retrieve the full text content for a list of URLs using Exa's content extraction.
    Use this after `search_exa` surfaces promising URLs to get detailed content
    for synthesis. The output includes each page's URL, title, and extracted text.

    Args:
        urls: A list of URLs to extract content from.
            Example: ["https://arxiv.org/abs/2301.00001", "https://example.com/paper"]
    """
    ctx.context.configs.logger.info(f"Reading Exa content for urls: {urls}")
    successful_inputs = ["exa_read"]
    try:
        contents_df = await asyncio.get_event_loop().run_in_executor(
            None, lambda: _exa_get_contents(urls)
        )
        if contents_df.empty:
            ctx.context.queries.append(successful_inputs)
            return "No content found for the provided URLs."

        results_text = []
        for _, row in contents_df.iterrows():
            full_text = row.get("full_text", "")
            truncated = full_text[:1000] if full_text else ""
            results_text.append(f"{row.get('url', '')}\n\n{truncated}")
            paper_row = pd.DataFrame([{
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "snippet": full_text,
                "query": "exa_read",
                "context": f"[{row.get('url', '')}]: {truncated}",
                "date": "",
            }])
            ctx.context.merge_papers_df(paper_row)
            successful_inputs.append(row.get("url", ""))

        ctx.context.queries.append(successful_inputs)
        ctx.context.configs.logger.info(
            f"Exa read successful inputs: {successful_inputs}, "
            f"collected total references: {len(ctx.context.papers_df) if ctx.context.papers_df is not None else 0}"
        )
        return "\n\n---\n\n".join(results_text) if results_text else "No content found"
    except Exception as e:
        ctx.context.queries.append(successful_inputs)
        return f"Error extracting content from Exa for {urls}: {e}"


def _call_model_input_filter(input: CallModelData[AgentContext]) -> ModelInputData:
    """
    This function is used to trim input to less than search_lm's max_ctx_len.
    """
    configs = input.context.configs
    instructions = input.model_data.instructions or ""
    input_items = input.model_data.input

    configs.logger.debug(f"Instructions: {instructions}")
    configs.logger.debug(f"Input: {input_items}")

    input_allowed_length = (
        configs.search_lm.max_ctx_len
        - len(instructions)
        - configs.search_lm.max_tokens
    )
    if input_allowed_length < 0:
        raise ValueError(
            f"Input is too long. Max allowed length is {input_allowed_length} tokens. {configs.search_lm.max_ctx_len} is the max context length and {configs.search_lm.max_tokens} is the max tokens."
        )

    user_message_index = [
        index
        for index, message in enumerate(input_items)
        if (
            Converter.maybe_input_message(message)
            or Converter.maybe_easy_input_message(message)
        )
        and message["role"] == "user"
    ]
    configs.logger.debug(f"User message index: {user_message_index}")
    if user_message_index:
        last_user_message_index = user_message_index[-1]
        input_allowed_length = input_allowed_length - len(
            str(input_items[last_user_message_index])
        )
    else:
        last_user_message_index = -1

    final_input: list[ResponseInputItemParam] = []
    for i, message in enumerate(reversed(input_items)):
        if i == len(input_items) - 1 - last_user_message_index:
            final_input.append(message)
            continue
        message_length = len(str(_to_dump_compatible(message)))
        configs.logger.debug(f"Message length: {message_length}")
        if message_length > input_allowed_length:
            continue
        final_input.append(message)
        input_allowed_length -= message_length
    final_input.reverse()
    configs.logger.debug(f"Final input: {final_input}")
    return ModelInputData(instructions=instructions, input=final_input)

async def agentic_search(
    configs: Configs,
    topic: str,
    end_date: datetime | None = None,
) -> tuple[list[list[str]], pd.DataFrame, str]:
    context = AgentContext(
        configs=configs,
        end_date=end_date,
        papers_df=None,
        queries=[],
    )
    model, model_configs = _lotus_lm_to_openai_lm(configs, configs.search_lm)
    tools = [search_arxiv, read_arxiv_abstracts]
    prompt = (
        openai_sdk_arxiv_search_system_prompt_without_cutoff
        if not end_date
        else openai_sdk_arxiv_search_system_prompt
    )
    if configs.enable_web_search:
        configs.logger.info("Web search is enabled, adding web search tools and prompt.")
        tools.append(search_web)
        tools.append(read_webpage_full_text)
        prompt = (
            openai_sdk_search_system_prompt_without_cutoff
            if not end_date
            else openai_sdk_search_system_prompt
        )
    if configs.enable_exa_search:
        configs.logger.info("Exa search is enabled, adding Exa search tools and prompt.")
        tools.append(search_exa)
        tools.append(read_exa_contents)
        prompt = (
            openai_sdk_exa_search_system_prompt_without_cutoff
            if not end_date
            else openai_sdk_exa_search_system_prompt
        )
    agent = Agent(
        name="Research Assistant",
        instructions=prompt,
        tools=tools,
        model=model,
        model_settings=model_configs,
    )
    result = await Runner.run(
        agent,
        input=topic,
        context=context,
        max_turns=100,
        run_config=RunConfig(call_model_input_filter=_call_model_input_filter),
    )
    docs_df = (
        result.context_wrapper.context.papers_df
        if result.context_wrapper.context.papers_df is not None
        else pd.DataFrame(
            columns=["title", "url", "snippet", "query", "context", "date"]
        )
    )
    configs.search_lm.stats.virtual_usage = LMStats.TotalUsage(
        prompt_tokens=result.context_wrapper.usage.input_tokens,
        completion_tokens=result.context_wrapper.usage.output_tokens,
        total_tokens=result.context_wrapper.usage.total_tokens,
    )
    configs.search_lm.stats.physical_usage = LMStats.TotalUsage(
        prompt_tokens=result.context_wrapper.usage.input_tokens,
        completion_tokens=result.context_wrapper.usage.output_tokens,
        total_tokens=result.context_wrapper.usage.total_tokens,
    )
    if len(docs_df) > 0:
        docs_df = docs_df.drop_duplicates(subset=["url"])
    return result.context_wrapper.context.queries, docs_df, result.final_output

def _lotus_lm_to_openai_lm(
    configs: Configs, lm: LM,
) -> tuple[OpenAIResponsesModel | OpenAIChatCompletionsModel, ModelSettings]:
    client = AsyncOpenAI(
        base_url=lm.kwargs.get("api_base"),
    )
    if _is_responses_model(configs, lm.model):
        configs.logger.info(
            f"Converting Lotus LM to OpenAI Responses Model: {lm.model}"
        )
        model = OpenAIResponsesModel(
            model=lm.model,
            openai_client=client,
        )
    else:
        configs.logger.info(
            f"Converting Lotus LM to OpenAI Chat Completions Model: {lm.model}"
        )
        model = OpenAIChatCompletionsModel(
            model=lm.model,
            openai_client=client,
        )

    model_configs = ModelSettings(
        temperature=lm.kwargs.get("temperature"),
        top_p=lm.kwargs.get("top_p"),
        frequency_penalty=lm.kwargs.get("frequency_penalty"),
        presence_penalty=lm.kwargs.get("presence_penalty"),
        reasoning=Reasoning(
            effort=lm.kwargs.get("reasoning_effort", "low"),
        )
        if lm.kwargs.get("reasoning_effort")
        else None,
        truncation="auto",
    )
    configs.logger.info(
        f"Using OpenAI LM: {lm.model} {model_configs}"
    )
    return model, model_configs

def _is_responses_model(configs: Configs, model: str) -> bool:
    if configs.use_responses_model is not None:
        return configs.use_responses_model
    if "gpt" in model.lower():
        if "oss" in model.lower():
            return True
        match = re.search(r"(\d+(?:\.\d+)?)", model)
        if match:
            try:
                model_number = float(match.group())
                return model_number >= 5.0
            except ValueError:
                return False
        return False

