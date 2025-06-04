import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fluxllm.clients import FluxOpenAICompletion


@pytest.fixture
def mock_openai_completion():
    """Mock the OpenAI Completion API response"""
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1677858242,
        "model": "text-davinci-003",
        "choices": [
            {
                "text": "This is a test completion response.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12
        }
    }
    return mock_response


@pytest.mark.asyncio
async def test_flux_openai_completion(mock_openai_completion):
    """Test the FluxOpenAICompletion client"""
    
    # Create a temporary cache file
    cache_file = "test_cache.jsonl"
    if os.path.exists(cache_file):
        os.remove(cache_file)
    
    # Mock the AsyncOpenAI client
    with patch("fluxllm.clients.openai.AsyncOpenAI") as mock_async_openai:
        # Set up the mock
        mock_client = MagicMock()
        mock_completions = MagicMock()
        mock_completions.create = AsyncMock(return_value=mock_openai_completion)
        mock_client.completions = mock_completions
        mock_async_openai.return_value = mock_client
        
        # Create the client with a dummy API key
        client = FluxOpenAICompletion(cache_file=cache_file, api_key="dummy_key")
        
        # Create a test request
        request = {
            "model": "text-davinci-003",
            "prompt": "Complete this sentence:"
        }
        
        # Make the request
        response = await client.make_request_async(request)
        
        # Check that the request was made with the correct parameters
        mock_client.completions.create.assert_called_once_with(
            model="text-davinci-003",
            prompt="Complete this sentence:"
        )
        
        # Check that the response was returned correctly
        assert response["choices"][0]["text"] == "This is a test completion response."
    
    # Clean up
    if os.path.exists(cache_file):
        os.remove(cache_file)