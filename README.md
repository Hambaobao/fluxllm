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
    base_url=args.base_url,
    api_key=args.api_key,
    cache_file=args.cache_file,
    max_retries=args.max_retries,
    max_parallel_size=args.max_parallel_size,
)

responses = client.request(
    requests=requests,
    model=args.model,
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    top_p=args.top_p,
)

contents = [response.choices[0].message.content for response in responses]
```
