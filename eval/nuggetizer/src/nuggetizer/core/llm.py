import time
from typing import Dict, List, Optional, Union, Tuple
import tiktoken
from openai import AzureOpenAI, OpenAI
from ..utils.api import get_azure_openai_args, get_openai_api_key, get_vllm_args


class LLMHandler:
    def __init__(
        self,
        model: str,
        api_keys: Optional[Union[str, List[str]]] = None,
        context_size: int = 8192,
        api_type: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        use_azure_openai: bool = False,
        use_vllm: bool = False,
    ):
        self.model = model
        self.context_size = context_size

        # Auto-configure API keys and settings if not provided
        if use_vllm or api_type == "vllm":
            vllm_args = get_vllm_args()
            api_type = "vllm"
            api_base = vllm_args.get("api_base")
            api_keys = vllm_args.get("api_key", api_keys)
        elif use_azure_openai and (
            api_keys is None
            or (api_type == "azure" or (api_type is None and "gpt" in model.lower()))
        ):
            azure_args = get_azure_openai_args()
            api_type = "azure"
            api_base = azure_args.get("api_base")
            api_version = azure_args.get("api_version")
            api_keys = azure_args.get("api_key", api_keys) or get_openai_api_key()
        else:
            api_type = "openai"
            api_keys = api_keys or get_openai_api_key()
        self.api_keys = [api_keys] if isinstance(api_keys, str) else api_keys
        self.current_key_idx = 0
        self.client = self._initialize_client(api_type, api_base, api_version)

    def _initialize_client(self, api_type, api_base, api_version):
        if api_type == "vllm" and api_base:
            return OpenAI(api_key=self.api_keys[0], base_url=api_base)
        elif api_type == "azure" and all([api_base, api_version]):
            return AzureOpenAI(
                api_key=self.api_keys[0],
                api_version=api_version,
                azure_endpoint=api_base,
            )
        elif api_type == "openai":
            return OpenAI(api_key=self.api_keys[0])
        else:
            raise ValueError(f"Invalid API type: {api_type}")

    def run(
        self, messages: List[Dict[str, str]], temperature: float = 0
    ) -> Tuple[str, int]:
        remaining_retry = 5
        while remaining_retry > 0:
            if "o1" in self.model:
                # System message is not supported for o1 models
                new_messages = messages[1:]
                new_messages[0]["content"] = (
                    messages[0]["content"] + "\n" + messages[1]["content"]
                )
                messages = new_messages[:]
                temperature = 1.0
            try:
                completion = None
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=4096,
                    timeout=30,
                )
                response = completion.choices[0].message.content
                try:
                    # For newer models like gpt-4o that may not have specific encodings yet
                    if "gpt-4o" in self.model or "gpt-4.1" in self.model:
                        encoding = tiktoken.get_encoding("o200k_base")
                    else:
                        encoding = tiktoken.get_encoding(self.model)
                except Exception as e:
                    print(f"Error: {str(e)}")
                    encoding = tiktoken.get_encoding("cl100k_base")
                return response, len(encoding.encode(response))
            except Exception as e:
                print(f"LLM Inference Error: {str(e)}")
                remaining_retry -= 1
                if remaining_retry <= 0:
                    raise RuntimeError("Reached max of 5 retries.")
                # Don't retry in case of safety trigger.
                if (
                    completion
                    and completion.choices
                    and completion.choices[0].finish_reason == "content_filter"
                ):
                    raise ValueError("Request blocked by content filter.")
                if self.api_keys is not None:
                    self.current_key_idx = (self.current_key_idx + 1) % len(
                        self.api_keys
                    )
                    self.client.api_key = self.api_keys[self.current_key_idx]
                time.sleep(0.1)
        return "", 0
