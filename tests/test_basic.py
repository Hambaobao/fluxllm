from fluxllm.clients import FluxOpenAIChat


def test_basic():

    client = FluxOpenAIChat(
        base_url="http://127.0.0.1:8008/v1",
        api_key="test",
        cache_file="cache.jsonl",
        max_retries=3,
        max_qps=5,  # Add rate limiting of 5 requests per second
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


def test_qpm_rate_limiting():
    """Test rate limiting using QPM instead of QPS"""
    
    client = FluxOpenAIChat(
        base_url="http://127.0.0.1:8008/v1",
        api_key="test",
        cache_file="cache.jsonl",
        max_retries=3,
        max_qps=None,  # Disable QPS rate limiting
        max_qpm=60,    # Set QPM rate limiting to 60 requests per minute
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
    requests = [{**request, "id": f"id-{i}"} for i in range(5)]

    responses = client.request(
        requests=requests,
        model="gpt-4o",
        max_tokens=100,
        temperature=0.5,
        top_p=1.0,
    )

    assert len(responses) == len(requests)


def test_dual_rate_limiting():
    """Test rate limiting using both QPS and QPM simultaneously"""
    
    client = FluxOpenAIChat(
        base_url="http://127.0.0.1:8008/v1",
        api_key="test",
        cache_file="cache.jsonl",
        max_retries=3,
        max_qps=10,    # 10 requests per second
        max_qpm=120,   # 120 requests per minute (2 per second)
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
    requests = [{**request, "id": f"id-{i}"} for i in range(15)]

    responses = client.request(
        requests=requests,
        model="gpt-4o",
        max_tokens=100,
        temperature=0.5,
        top_p=1.0,
    )

    assert len(responses) == len(requests)
    # Both rate limits should be enforced, with the more restrictive one (QPM in this case)
    # determining the actual concurrency
