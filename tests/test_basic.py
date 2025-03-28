from fluxllm.clients import FluxOpenAI


def test_basic():

    client = FluxOpenAI(
        base_url="http://127.0.0.1:8008/v1",
        api_key="test",
        cache_file="cache.jsonl",
        max_retries=3,
        max_parallel_size=10,
    )

    request = {
        "messages": [{
            "role": "user",
            "content": "Hello, world!"
        },],
        "model": "gpt-4o",
        "max_tokens": 100,
        "temperature": 0.5,
        "top_p": 1,
    }
    # requests is a list of requests
    requests = [{**request, "id": f"id-{i}"} for i in range(10)]

    responses = client.request(
        requests=requests,
        model="gpt-4o",
        max_tokens=100,
        temperature=0.5,
        top_p=1.0,
    )

    assert len(responses) == len(requests)
