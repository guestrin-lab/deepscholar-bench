from pathlib import Path
from pydantic import BaseModel, model_validator
import logging
from lotus.models import LM
from pydantic import Field
from lotus import WebSearchCorpus
from lotus.types import ReasoningStrategy
from typing import Any

logger = logging.getLogger("deepscholar_base")

class Configs(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    logger: logging.Logger

    use_agentic_search: bool = True
    max_search_retries: int = 3
    use_structured_output: bool = True

    # Common for both agentic and recursive search
    enable_web_search: bool = True
    enable_exa_search: bool = False
    per_query_max_search_results_count: int = 10

    # Only for agentic search
    use_responses_model: bool | None = None

    # Only for recursive search
    num_search_steps: int = 3
    num_search_queries_per_step_per_corpus: int = 2
    web_corpuses: list[WebSearchCorpus] = Field(
        default_factory=lambda: [WebSearchCorpus.TAVILY]
    )
    
    # Filtering configs
    use_sem_filter: bool = True
    use_sem_topk: bool = True
    final_max_results_count: int = 30
    # Defaulted in initialize_sem_filter_kwargs and initialize_sem_topk_kwargs
    sem_filter_kwargs: dict[str, Any]
    sem_topk_kwargs: dict[str, Any]

    # Taxonomization configs
    categorize_references: bool = True
    generate_category_summary: bool = True

    # Generation configs
    generate_insights: bool = True

    # LM configs (defaulted in initialize_lms)
    filter_lm: LM
    search_lm: LM
    taxonomize_lm: LM
    generation_lm: LM

    @model_validator(mode="before")
    def initialize_logger(value: dict[str, Any]):
        if not value.get("logger"):
            value["logger"] = logger
        return value

    @model_validator(mode="before")
    def resolve_web_corpuses(value: dict[str, Any]):
        """Model validator for web_corpuses to ensure proper enum conversion."""
        raw = value.get("web_corpuses")
        if raw is None:
            value["web_corpuses"] = [WebSearchCorpus.TAVILY]
        elif isinstance(raw, list):
            out = []
            for item in raw:
                if isinstance(item, WebSearchCorpus):
                    out.append(item)
                else:
                    out.append(WebSearchCorpus[item.upper()] if isinstance(item, str) else item)
            value["web_corpuses"] = out
        else:
            value["web_corpuses"] = [WebSearchCorpus.TAVILY]
        return value

    @model_validator(mode="before")
    def initialize_sem_filter_kwargs(value: dict[str, Any]):
        # Default to COT reasoning and return explanations
        value["sem_filter_kwargs"] = {
            "strategy": ReasoningStrategy.COT,
            "return_explanations": True,
            **value.get("sem_filter_kwargs", {}),
        }
        return value

    @model_validator(mode="before")
    def initialize_sem_topk_kwargs(value: dict[str, Any]):
        # Default to COT reasoning and return explanations
        value["sem_topk_kwargs"] = {
            "strategy": ReasoningStrategy.COT,
            "return_explanations": True,
            **value.get("sem_topk_kwargs", {}),
        }
        return value

    @model_validator(mode="before")
    def initialize_lms(value: dict[str, Any]):
        raw_lm = value.get("lm")
        if raw_lm is None:
            configured_lm = LM(model="gpt-5-mini", temperature=1.0, reasoning_effort="low", max_tokens=10000)
        elif isinstance(raw_lm, dict):
            configured_lm = LM(**raw_lm)
            value["lm"] = configured_lm
        else:
            configured_lm = raw_lm
        assert isinstance(configured_lm, LM), "configured_lm must be a Lotus LM"
        configured_lm_kwargs = configured_lm.kwargs
        configured_lm_kwargs.pop("max_completion_tokens")
        for lm_key in ["filter_lm", "search_lm", "taxonomize_lm", "generation_lm"]:
            existing = value.get(lm_key)
            if existing is None:
                value[lm_key] = LM(
                    model=configured_lm.model,
                    max_ctx_len=configured_lm.max_ctx_len,
                    max_batch_size=configured_lm.max_batch_size,
                    rate_limit=configured_lm.rate_limit,
                    tokenizer=configured_lm.tokenizer,
                    cache=configured_lm.cache,
                    physical_usage_limit=configured_lm.physical_usage_limit,
                    virtual_usage_limit=configured_lm.virtual_usage_limit,
                    **configured_lm_kwargs,
                )
            elif isinstance(existing, dict):
                value[lm_key] = LM(**existing)
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Configs":
        """Load Configs from a YAML file.

        LM may be given as a dict; all stage LMs (filter_lm, search_lm, etc.) are
        derived from it if not set. Example YAML:

          lm:
            model: gpt-4o
            temperature: 1.0
            max_tokens: 10000
          search_mode: agentic
          enable_web_search: true
          web_corpuses: [TAVILY, ARXIV]
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for Configs.from_yaml(). Install with: pip install pyyaml")
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("YAML config must be a mapping (dict).")
        
        if data.get("search_mode") == "agentic":
            data["use_agentic_search"] = True
        elif data.get("search_mode") == "recursive":
            data["use_agentic_search"] = False
        elif data.get("search_mode") is None:
            data["use_agentic_search"] = True
        else:
            raise ValueError(f"Invalid search mode: {data.get('search_mode')}")
        return cls(**data)

    def log(self):
        model_dump = self.model_dump(
            mode="json",
            exclude={"logger", "search_lm", "taxonomize_lm", "generation_lm", "filter_lm"},
        )
        for lm in ["search_lm", "taxonomize_lm", "generation_lm", "filter_lm"]:
            lm_instance = getattr(self, lm)
            model_dump[lm] = {
                "model": lm_instance.model,
                "max_ctx_len": lm_instance.max_ctx_len,
                "kwargs": lm_instance.kwargs,
            }
        return model_dump