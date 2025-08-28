import os
from typing import Dict, Optional

from dotenv import load_dotenv


def get_openai_api_key() -> Optional[str]:
    load_dotenv(dotenv_path=".env")
    openai_api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return openai_api_key


def get_azure_openai_args() -> Dict[str, Optional[str]]:
    load_dotenv(dotenv_path=".env")
    azure_args = {
        "api_type": "azure",
        "api_version": os.getenv("AZURE_OPENAI_API_VERSION"),
        "api_base": os.getenv("AZURE_OPENAI_API_BASE"),
        "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
    }

    # Sanity check
    assert all(list(azure_args.values())), (
        "Ensure that `AZURE_OPENAI_API_BASE`, `AZURE_OPENAI_API_VERSION` are set"
    )
    for key, value in azure_args.items():
        if value is None:
            raise ValueError(f"{key} not found in environment variables")
    else:
        return azure_args


def get_cohere_api_key() -> Optional[str]:
    load_dotenv(dotenv_path=".env.local")
    co_api_key = os.getenv("CO_API_KEY")
    return co_api_key


def get_anyscale_api_key() -> Optional[str]:
    load_dotenv(dotenv_path=".env.local")
    anyscale_api_key = os.getenv("ANYSCALE_API_KEY")
    return anyscale_api_key


def get_vllm_args() -> Dict[str, Optional[str]]:
    load_dotenv(dotenv_path=".env")
    vllm_args = {
        "api_type": "vllm",
        "api_base": os.getenv("VLLM_API_BASE"),
        "api_key": os.getenv(
            "VLLM_API_KEY", "dummy-key"
        ),  # vLLM often doesn't require real API keys
    }

    # Sanity check - only api_base is required for vLLM
    if not vllm_args["api_base"]:
        raise ValueError("VLLM_API_BASE not found in environment variables")

    return vllm_args
