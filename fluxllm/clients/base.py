import asyncio
import os
import random
from typing import Any, Dict, List

from aiolimiter import AsyncLimiter
from openai.types.chat import ChatCompletion

from fluxllm.client_utils import FluxCache, create_progress


class BaseClient:
    """
    A client that allows for concurrent requests to the same API.
    """

    def __init__(
        self,
        cache_file: str | None = None,
        max_retries: int | None = None,
        max_parallel_size: int = 1,
        max_qps: float = 0,  # 0 means no rate limiting
        progress_msg: str = "Requesting...",
    ):
        """
        Initialize the client.

        Args:
            cache_file: the cache file to use.
            max_retries: the maximum number of retries to make, defaults to `None`.
            max_parallel_size: the number of requests to make concurrently, defaults to `1`.
            max_qps: maximum queries per second (rate limit), defaults to `0` (no rate limiting).
            progress_msg: the message to display in the progress bar, defaults to `Requesting...`.
        """
        if cache_file is None:
            cache_file = os.getenv("CACHE_FILE", "cache.jsonl")

        # initialize the cache client
        self.cache = FluxCache(cache_file)
        self.lock = asyncio.Lock()
        self.max_retries = max_retries
        self.max_parallel_size = max_parallel_size
        self.max_qps = max_qps
        self.rate_limiter = AsyncLimiter(max_rate=max_qps, time_period=1.0) if max_qps > 0 else None
        self.progress_msg = progress_msg

    async def save_to_cache_thread_safe(self, sample: Dict, response: Dict, save_request: bool = False):
        async with self.lock:
            self.cache.save_to_cache(sample, response, save_request=save_request)

    async def request_async(
        self,
        requests: List[Dict[str, Any]],
        save_request: bool = False,
        **kwargs,
    ) -> None:
        """
        Make requests for all uncached samples in given list.
        This function is designed to handle the generation of multiple responses in a batch.
        It uses a semaphore to control the number of concurrent requests and a rate limiter
        to control the request rate (if max_qps > 0).

        Args:
            requests: list of requests to generate responses for
            save_request: whether to save the request in the cache
            **kwargs: additional arguments to pass to request_async
        """

        # create a queue to hold the samples
        task_queue = asyncio.Queue()
        for request in requests:
            await task_queue.put(request)
        failure_counts = {self.cache.hash(request): 0 for request in requests}

        # limit the number of concurrent requests
        semaphore = asyncio.Semaphore(self.max_parallel_size)

        async def after_failure(request: Dict):
            failure_counts[self.cache.hash(request)] += 1
            if self.max_retries is None:
                print(f"Re-queue failed request: {self.cache.hash(request)}, failed {failure_counts[self.cache.hash(request)]} times.", flush=True)
                await task_queue.put(request)
                await asyncio.sleep(random.randint(3, 10))
            else:
                if failure_counts[self.cache.hash(request)] < self.max_retries:
                    print(f"Re-queue failed request: {self.cache.hash(request)}, failed {failure_counts[self.cache.hash(request)]} times.", flush=True)
                    await task_queue.put(request)
                    await asyncio.sleep(random.randint(3, 10))
                else:
                    print(f"Request failed after {self.max_retries} retries. Aborting this request.", flush=True)
                    progress.advance(task)

        async def worker():
            while not task_queue.empty():
                request = await task_queue.get()
                async with semaphore:
                    # Apply rate limiting if configured
                    if self.rate_limiter is not None:
                        async with self.rate_limiter:
                            response = await self.make_request_async(request, **kwargs)
                    else:
                        response = await self.make_request_async(request, **kwargs)
                    
                    if response is not None:
                        await self.save_to_cache_thread_safe(request, response, save_request=save_request)
                        progress.advance(task)
                        print(f"Request succeeded for request: {self.cache.hash(request)}", flush=True)
                    else:
                        await after_failure(request)
                task_queue.task_done()

        with create_progress() as progress:
            task = progress.add_task(f"[cyan]{self.progress_msg}", total=len(requests))
            workers = [asyncio.create_task(worker()) for _ in range(self.max_parallel_size)]

            await task_queue.join()
            for worker_task in workers:
                worker_task.cancel()

    def request(self, requests: List[Dict[str, Any]], save_request: bool = False, **kwargs) -> List[ChatCompletion | None]:
        """
        Make requests for all uncached samples in given list.
        This is a synchronous wrapper around request_async.

        Args:
            requests: List of requests to make
            **kwargs: additional arguments to pass to request_async
        """

        # get the samples that are not cached
        remaining_requests = [request for request in requests if not self.cache.is_cached(request)]
        print(f"Remaining {len(remaining_requests)} requests to generate", flush=True)

        # request the responses
        asyncio.run(self.request_async(requests=remaining_requests, save_request=save_request, **kwargs))

        # collect the responses
        responses = self.collect_responses(requests)

        return responses

    async def make_request_async(self, request: Dict, **kwargs) -> Dict | None:
        """
        Make a single request to the API and cache the response.
        """
        raise NotImplementedError("Implement this in the subclass")

    def collect_responses(self, requests: List[Dict]) -> List[ChatCompletion | None]:
        """
        Collect the responses from the cache.
        """
        raise NotImplementedError("Implement this in the subclass")
