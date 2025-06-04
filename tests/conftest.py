import multiprocessing
import time

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    user_msg = data.get("messages", [{}])[-1].get("content", "unknown")

    return JSONResponse(
        content={
            "id": "chatcmpl-B9MHDbslfkBeAs8l4bebGdFOJ6PeG",
            "object": "chat.completion",
            "created": 1741570283,
            "model": "gpt-4o-2024-08-06",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"echo, {user_msg}",
                    "refusal": None,
                    "annotations": []
                },
                "logprobs": None,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 1117,
                "completion_tokens": 46,
                "total_tokens": 1163,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "audio_tokens": 0
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0
                }
            },
            "service_tier": "default",
            "system_fingerprint": "fp_fc9f1d7035"
        })


@app.post("/v1/completions")
async def completions(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "unknown")

    return JSONResponse(
        content={
            "id": "cmpl-123",
            "object": "text_completion",
            "created": 1677858242,
            "model": data.get("model", "text-davinci-003"),
            "choices": [
                {
                    "text": f"Response to: {prompt}",
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
        })


def run_mock_server():
    uvicorn.run(app, host="127.0.0.1", port=8008, log_level="error")


@pytest.fixture(scope="session", autouse=True)
def mock_openai_server():

    p = multiprocessing.Process(target=run_mock_server, daemon=True)
    p.start()
    time.sleep(2)
    yield
    p.terminate()
