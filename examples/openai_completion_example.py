import argparse
import json
from typing import Dict, List

from fluxllm.clients import FluxOpenAICompletion


def parse_args():

    def none_or_int(value):
        if value in ["none", "None"]:
            return None
        return int(value)
    
    def none_or_float(value):
        if value in ["none", "None"]:
            return None
        return float(value)

    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument("--input-file", type=str, required=True, help="Input file for prompts")
    parser.add_argument("--output-file", type=str, required=True, help="Output file for generated data")
    parser.add_argument("--cache-file", type=str, default="cache.jsonl", help="Cache file for API responses")
    parser.add_argument("--max-retries", type=none_or_int, default=None, help="Maximum number of retries for API requests")
    parser.add_argument("--max-qps", type=none_or_float, default=None, help="Maximum queries per second (rate limit)")
    parser.add_argument("--max-qpm", type=float, default=100, help="Maximum queries per minute (rate limit)")

    # OpenAI API parameters
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for the OpenAI API")
    parser.add_argument("--api-key", type=str, default=None, help="API key for the OpenAI API")
    parser.add_argument("--model", type=str, required=True, help="Name of the OpenAI model to use for generation (e.g. text-davinci-003)")
    parser.add_argument("--max-tokens", type=none_or_int, default=None, help="Maximum number of tokens to generate in each response")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature between 0 and 2. Lower values make output more focused/deterministic")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling: consider tokens with top_p cumulative probability. Lower values make output more focused")
    parser.add_argument("--stop", type=List[str], default=None, help="List of sequences where the API will stop generating further tokens")

    return parser.parse_args()


def load_samples(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def collate_request(sample: Dict, **kwargs) -> Dict:
    """
    Create a request from the sample for API request.

    Args:
        sample: input sample to create request from
        **kwargs: additional arguments passed from generate() call that can be used if needed

    Returns:
        Dictionary in the format required by the OpenAI Completions API
    """

    # Assuming sample has a 'prompt' field
    return {"prompt": sample.get("prompt", "Complete this text:")}


if __name__ == "__main__":

    args = parse_args()

    # initialize client
    client = FluxOpenAICompletion(
        base_url=args.base_url,
        api_key=args.api_key,
        cache_file=args.cache_file,
        max_retries=args.max_retries,
        max_qps=args.max_qps,
        max_qpm=args.max_qpm,
    )

    # load the samples
    samples = load_samples(args.input_file)

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

    # Extract the text from the completions
    contents = [response.choices[0].text for response in responses if response is not None]

    # Print the first response as an example
    if contents:
        print(contents[0])
    
    # Save the results to the output file
    with open(args.output_file, "w") as f:
        for content in contents:
            f.write(json.dumps({"completion": content}) + "\n")