import argparse
import json
from typing import Dict, List

from fluxllm.clients import FluxOpenAI


def parse_args():

    def none_or_int(value):
        if value in ["none", "None"]:
            return None
        return int(value)

    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--input-file", type=str, required=True, help="Input file for questions")
    parser.add_argument("--output-file", type=str, required=True, help="Output file for generated data")
    parser.add_argument("--cache-file", type=str, default="cache.jsonl", help="Cache file for API responses")
    parser.add_argument("--max-retries", type=none_or_int, default=None, help="Maximum number of retries for API requests")
    parser.add_argument("--max-parallel-size", type=int, default=1, help="Maximum number of parallel requests")

    # OpenAI API parameters
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for the OpenAI API")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the OpenAI API")
    parser.add_argument("--model", type=str, required=True, help="Name of the OpenAI model to use for generation (e.g. gpt-3.5-turbo, gpt-4)")
    parser.add_argument("--max-tokens", type=none_or_int, default=None, help="Maximum number of tokens to generate in each response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature between 0 and 2. Lower values make output more focused/deterministic")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling: consider tokens with top_p cumulative probability. Lower values make output more focused")
    parser.add_argument("--top-k", type=int, default=50, help="Only sample from the top K most likely tokens. Lower values make output more focused")
    parser.add_argument("--stop", type=List[str], default=None, help="List of sequences where the API will stop generating further tokens")

    return parser.parse_args()


def load_samples(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def collate_request(sample: Dict, **kwargs) -> List[Dict]:
    """
    Create messages from the sample for API request.

    Args:
        sample: input sample to create messages from
        **kwargs: additional arguments passed from generate() call that can be used if needed

    Returns:
        List of message dictionaries in the format required by the OpenAI API
    """

    messages = [{"role": "user", "content": "Hello, how are you?"}]
    return messages


if __name__ == "__main__":

    args = parse_args()

    # initialize client
    client = FluxOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        cache_file=args.cache_file,
        max_retries=args.max_retries,
        max_parallel_size=args.max_parallel_size,
    )

    # load the samples
    samples = load_samples(args.data_path)

    # collate the requests
    requests = [collate_request(sample) for sample in samples]

    # request responses
    responses = client.request(
        requests=requests,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    contents = [response.choices[0].message.content for response in responses]

    print(contents[0])
