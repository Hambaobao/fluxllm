# FluxLLM

A blazing-fast gateway for asynchronous, high-concurrency LLM requests (OpenAI, Claude, etc.).
Dynamically caches responses to slash latency and costs while scaling seamlessly across AI providers.
Built for developers who demand speed without compromise.

## Features

- Asynchronous, high-concurrency requests
- Dynamically caches responses to slash latency and costs
- Seamlessly scales across AI providers
- Simple to use
- Extensible to add new AI providers

## Installation

```bash
pip install fluxllm
```

## Usage

```python
from fluxllm.clients import FluxOpenAI

client = FluxOpenAI(
    base_url="https://api.openai.com/v1", # base url of the ai provider
    api_key="sk-...", # api key of the ai provider
    cache_file="/path/to/cache.jsonl", # path to the cache file
    max_retries=3, # max retries for a request, set to None will retry infinitely
    max_parallel_size=1024, # max 1024 requests concurrently
)

# request is a object that passed to the endpoint of the ai provider
request = {
    "messages": [
        {"role": "user", "content": "Hello, world!"},
    ],
    "model": "gpt-4o",
    "max_tokens": 100,
    "temperature": 0.5,
    "top_p": 1,
}
# requests is a list of requests
requests = [request] * 1000

# The list of responses maintains the same order as the input requests.
# If a request fails, its corresponding response will be None.
responses = client.request(requests)

# post-process the responses to get what you want
contents = [response.choices[0].message.content for response in responses]
```
