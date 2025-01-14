from typing import List, Dict, Callable
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

import json
import os
import copy

from cc_clients.clients.base import ConcurrentClient

from .utils import DUMMY_CHAT_COMPLETION, make_valid_response


class ConcurrentOpenAIClient(ConcurrentClient):
    """
    A client that can handle multiple requests concurrently.
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
        collate_fn: Callable[[Dict, Dict], Dict],
        cache_file: str | None = None,
        max_retries: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ):
        """
        Initialize the client.

        Args:
            - samples: the samples to generate responses for.
            - collate_fn: the function to collate the messages for each sample.
            - max_retries: the maximum number of retries to make, defaults to `None`.
            - base_url: OpenAI Compatible API base URL.
            - api_key: OpenAI API key.
        """

        super().__init__(cache_file=cache_file, collate_fn=collate_fn, max_retries=max_retries)

        # initialize the client
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("OPENAI_API_BASE"),
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    async def request_async(self, sample: Dict, **kwargs) -> Dict | None:
        """
        Make a single request to the OpenAI API and cache the response.

        Args:
            sample: Dictionary containing the input data to generate a response for
            **kwargs: Additional arguments that will be passed to the OpenAI API call if they are in SUPPORT_ARGS

        Returns:
            bool: True if the request was successful and cached, False if it failed
        """

        try:
            # collate the messages
            messages = self.collate_fn(sample, **kwargs)
            # prepare the arguments for the API call
            gen_kwargs = {k: v for k, v in kwargs.items() if k in self.SUPPORT_ARGS and v is not None}
            # add the messages
            gen_kwargs["messages"] = messages
            # add timeout if it is specified
            if "timeout" in kwargs:
                gen_kwargs["timeout"] = kwargs.get("timeout")
            # request for the response
            response = await self.client.chat.completions.create(**gen_kwargs)
            # the content is necessary for the a valid response
            _ = response.choices[0].message.content
            # make sure the response is a valid ChatCompletion object
            response = make_valid_response(response, **kwargs)
            response_dict = response.model_dump()
            return response_dict
        except Exception as e:
            print(f"{type(e).__name__}: {e}")
            return None

    def collect_responses(self, samples: List[Dict]) -> List[ChatCompletion]:
        """
        Get the cached responses for a list of samples and convert them to `ChatCompletion` objects.

        Args:
            samples: list of samples to get responses for
        Returns:
            list of ChatCompletion objects containing the cached responses
        """

        # First get the cached responses as dictionaries
        dict_responses = []
        for sample in samples:
            cached_response = self.cache.collect_result(sample)
            dict_responses.append(cached_response)

        # Then convert dictionaries to ChatCompletion objects
        chat_completions = []
        for dict_response in dict_responses:
            chat_completion = ChatCompletion.model_validate(dict_response)
            chat_completions.append(chat_completion)

        return chat_completions

    def create_dummy_response(self, **kwargs) -> ChatCompletion:
        """
        Return a dummy `ChatCompletion` response, which is used when the request fails.
        """
        dummy_response = copy.deepcopy(DUMMY_CHAT_COMPLETION)
        dummy_response = make_valid_response(dummy_response, **kwargs)
        dummy_response_dict = dummy_response.model_dump()
        return dummy_response_dict
