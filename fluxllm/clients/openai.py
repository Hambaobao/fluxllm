import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.completion import Completion

from fluxllm.clients.base import BaseClient


class FluxOpenAIChat(BaseClient):
    """
    A client that can handle multiple requests concurrently for OpenAI chat completions.
    """

    SUPPORT_ARGS = {
        "model",
        "messages",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "max_tokens",
        "n",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "stream",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "user",
        "function_call",
        "functions",
        "tenant",
        "max_completion_tokens",
    }

    progress_prompt: str = "Generating..."

    def __init__(
        self,
        cache_file: str | None = None,
        max_retries: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_qps: Optional[float] = None,
        max_qpm: float = 100,
    ):
        """
        Initialize the client.

        Args:
            - cache_file: the file to cache the results.
            - max_retries: the maximum number of retries to make, defaults to `None`.
            - base_url: OpenAI Compatible API base URL.
            - api_key: OpenAI API key.
            - max_qps: maximum queries per second (rate limit), defaults to `None` (falls back to max_qpm).
            - max_qpm: maximum queries per minute (rate limit), defaults to `100`.
        """

        super().__init__(
            cache_file=cache_file, 
            max_retries=max_retries, 
            max_qps=max_qps,
            max_qpm=max_qpm
        )

        # initialize the client
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("OPENAI_API_BASE"),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    async def make_request_async(self, request: Dict[str, Any], **kwargs) -> Dict | None:
        """
        Make a single request to the OpenAI API and cache the response.

        Args:
            request: the request to make
            **kwargs: Additional arguments that will be passed to the OpenAI API call if they are in SUPPORT_ARGS

        Returns:
            bool: True if the request was successful and cached, False if it failed
        """

        try:

            assert "messages" in request, "`messages` are required in the request"
            assert "model" in request, "`model` is required in the request"

            gen_kwargs = {}
            for key in self.SUPPORT_ARGS:
                if key in request:
                    gen_kwargs[key] = request[key]
                else:
                    if key in kwargs and kwargs[key] is not None:
                        gen_kwargs[key] = kwargs[key]
            # add timeout if it is specified
            if "timeout" in kwargs:
                gen_kwargs["timeout"] = kwargs["timeout"]
            # request for the response
            response: ChatCompletion = await self.client.chat.completions.create(**gen_kwargs)
            response_dict = response.model_dump()
            return response_dict

        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            return None

    def collect_responses(self, requests: List[Dict]) -> List[ChatCompletion]:
        """
        Get the cached responses for a list of requests and convert them to `ChatCompletion` objects.

        Args:
            requests: list of requests to get responses for
        Returns:
            list of ChatCompletion objects containing the cached responses
        """

        # First get the cached responses as dictionaries
        cached_responses = [self.cache.collect_result(request) for request in requests]

        # Then convert dictionaries to ChatCompletion objects
        chat_completions = []
        for cached_response in cached_responses:
            if cached_response is None:
                chat_completion = None
            else:
                chat_completion = ChatCompletion.model_validate(cached_response)
            chat_completions.append(chat_completion)

        return chat_completions


class FluxOpenAICompletion(BaseClient):
    """
    A client that can handle multiple requests concurrently for OpenAI text completions.
    """

    SUPPORT_ARGS = {
        "model",
        "prompt",
        "best_of",
        "echo",
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_tokens",
        "n",
        "presence_penalty",
        "seed",
        "stop",
        "stream",
        "suffix",
        "temperature",
        "top_p",
        "user",
    }

    progress_prompt: str = "Generating..."

    def __init__(
        self,
        cache_file: str | None = None,
        max_retries: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_qps: Optional[float] = None,
        max_qpm: float = 100,
    ):
        """
        Initialize the client.

        Args:
            - cache_file: the file to cache the results.
            - max_retries: the maximum number of retries to make, defaults to `None`.
            - base_url: OpenAI Compatible API base URL.
            - api_key: OpenAI API key.
            - max_qps: maximum queries per second (rate limit), defaults to `None` (falls back to max_qpm).
            - max_qpm: maximum queries per minute (rate limit), defaults to `100`.
        """

        super().__init__(
            cache_file=cache_file, 
            max_retries=max_retries, 
            max_qps=max_qps,
            max_qpm=max_qpm
        )

        # initialize the client
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("OPENAI_API_BASE"),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    async def make_request_async(self, request: Dict[str, Any], **kwargs) -> Dict | None:
        """
        Make a single request to the OpenAI API and cache the response.

        Args:
            request: the request to make
            **kwargs: Additional arguments that will be passed to the OpenAI API call if they are in SUPPORT_ARGS

        Returns:
            bool: True if the request was successful and cached, False if it failed
        """

        try:
            assert "prompt" in request, "`prompt` is required in the request"
            assert "model" in request, "`model` is required in the request"

            gen_kwargs = {}
            for key in self.SUPPORT_ARGS:
                if key in request:
                    gen_kwargs[key] = request[key]
                else:
                    if key in kwargs and kwargs[key] is not None:
                        gen_kwargs[key] = kwargs[key]
            # add timeout if it is specified
            if "timeout" in kwargs:
                gen_kwargs["timeout"] = kwargs["timeout"]
            # request for the response
            response: Completion = await self.client.completions.create(**gen_kwargs)
            response_dict = response.model_dump()
            return response_dict

        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            return None

    def collect_responses(self, requests: List[Dict]) -> List[Completion]:
        """
        Get the cached responses for a list of requests and convert them to `Completion` objects.

        Args:
            requests: list of requests to get responses for
        Returns:
            list of Completion objects containing the cached responses
        """

        # First get the cached responses as dictionaries
        cached_responses = [self.cache.collect_result(request) for request in requests]

        # Then convert dictionaries to Completion objects
        completions = []
        for cached_response in cached_responses:
            if cached_response is None:
                completion = None
            else:
                completion = Completion.model_validate(cached_response)
            completions.append(completion)

        return completions
