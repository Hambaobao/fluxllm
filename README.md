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

### Chat Completions API

```python
from fluxllm.clients import FluxOpenAIChat

client = FluxOpenAIChat(
    base_url="https://api.openai.com/v1", # base url of the ai provider
    api_key="sk-...", # api key of the ai provider
    cache_file="/path/to/cache.jsonl", # path to the cache file
    max_retries=3, # max retries for a request, set to None will retry infinitely
    max_qps=None, # maximum queries per second (rate limit), defaults to None (falls back to max_qpm)
    max_qpm=100, # maximum queries per minute (rate limit), defaults to 100
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

### Text Completions API

```python
from fluxllm.clients import FluxOpenAICompletion

client = FluxOpenAICompletion(
    base_url="https://api.openai.com/v1", # base url of the ai provider
    api_key="sk-...", # api key of the ai provider
    cache_file="/path/to/cache.jsonl", # path to the cache file
    max_retries=3, # max retries for a request, set to None will retry infinitely
    max_qps=None, # maximum queries per second (rate limit), defaults to None (falls back to max_qpm)
    max_qpm=100, # maximum queries per minute (rate limit), defaults to 100
)

# request is a object that passed to the endpoint of the ai provider
request = {
    "prompt": "Complete this sentence: The quick brown fox",
    "model": "text-davinci-003",
    "max_tokens": 50,
    "temperature": 0.7,
}
# requests is a list of requests
requests = [request] * 100

# The list of responses maintains the same order as the input requests.
# If a request fails, its corresponding response will be None.
responses = client.request(requests, save_request=True)

# post-process the responses to get what you want
contents = [response.choices[0].text for response in responses if response is not None]
```

## Available Parameters

### Client Initialization Parameters

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|-------------|
| `cache_file` | Path to the file where responses will be cached | `"cache.jsonl"` | Any valid file path string |
| `max_retries` | Maximum number of retries for failed requests | `None` (infinite retries) | `None` or any positive integer |
| `base_url` | Base URL for the AI provider's API | `None` (uses `OPENAI_API_BASE` env var) | Any valid URL string |
| `api_key` | API key for the AI provider | `None` (uses `OPENAI_API_KEY` env var) | Any valid API key string |
| `max_qps` | Maximum queries per second (rate limit) | `None` (falls back to max_qpm) | `None` or any positive float |
| `max_qpm` | Maximum queries per minute (rate limit) | `100` | Any positive float |
| `progress_msg` | Message to display in the progress bar | `"Requesting..."` | Any string |

### Request Method Parameters

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|-------------|
| `requests` | List of request dictionaries to process | Required | List of dictionaries |
| `save_request` | Whether to save the request in the cache alongside the response | `False` | `True` or `False` |
| `**kwargs` | Additional arguments passed to the AI provider's API | - | Depends on the client type |

### FluxOpenAIChat Supported Arguments

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|-------------|
| `model` | ID of the model to use | Required | String (e.g., "gpt-4o", "gpt-3.5-turbo") |
| `messages` | List of messages in the conversation | Required | List of message objects |
| `frequency_penalty` | Penalty for token frequency | - | -2.0 to 2.0 |
| `logit_bias` | Modify likelihood of specified tokens | - | Dictionary mapping token IDs to bias values |
| `logprobs` | Whether to return log probabilities | - | `True` or `False` |
| `top_logprobs` | Number of most likely tokens to return | - | Integer |
| `max_tokens` | Maximum number of tokens to generate | - | Integer |
| `n` | Number of completions to generate | - | Integer |
| `presence_penalty` | Penalty for token presence | - | -2.0 to 2.0 |
| `response_format` | Format of the response | - | Dictionary (e.g., `{"type": "json_object"}`) |
| `seed` | Seed for deterministic sampling | - | Integer |
| `stop` | Sequences where generation should stop | - | String or list of strings |
| `stream` | Whether to stream responses | - | `True` or `False` |
| `temperature` | Sampling temperature | - | 0.0 to 2.0 |
| `top_p` | Nucleus sampling parameter | - | 0.0 to 1.0 |
| `tools` | List of tools the model may call | - | List of tool objects |
| `tool_choice` | Controls which tool is called | - | String or object |
| `user` | User identifier | - | String |
| `function_call` | Controls function calling | - | String or object |
| `functions` | List of functions the model may call | - | List of function objects |
| `timeout` | Request timeout in seconds | - | Float |

### FluxOpenAICompletion Supported Arguments

| Parameter | Description | Default | Valid Values |
|-----------|-------------|---------|-------------|
| `model` | ID of the model to use | Required | String (e.g., "text-davinci-003") |
| `prompt` | Text prompt to complete | Required | String |
| `best_of` | Number of completions to generate and return the best | - | Integer |
| `echo` | Whether to echo the prompt in the response | - | `True` or `False` |
| `frequency_penalty` | Penalty for token frequency | - | -2.0 to 2.0 |
| `logit_bias` | Modify likelihood of specified tokens | - | Dictionary mapping token IDs to bias values |
| `logprobs` | Whether to return log probabilities | - | Integer |
| `max_tokens` | Maximum number of tokens to generate | - | Integer |
| `n` | Number of completions to generate | - | Integer |
| `presence_penalty` | Penalty for token presence | - | -2.0 to 2.0 |
| `seed` | Seed for deterministic sampling | - | Integer |
| `stop` | Sequences where generation should stop | - | String or list of strings |
| `stream` | Whether to stream responses | - | `True` or `False` |
| `suffix` | Text to append to the prompt | - | String |
| `temperature` | Sampling temperature | - | 0.0 to 2.0 |
| `top_p` | Nucleus sampling parameter | - | 0.0 to 1.0 |
| `user` | User identifier | - | String |
| `timeout` | Request timeout in seconds | - | Float |

## Examples

### Using save_request Parameter

When `save_request` is set to `True`, both the request and response are saved in the cache file. This can be useful for debugging or analyzing the requests later.

```python
# Save both request and response in the cache
responses = client.request(requests, save_request=True)
```

### Setting Custom Timeout

You can set a custom timeout for requests:

```python
responses = client.request(requests, timeout=30.0)  # 30 seconds timeout
```
