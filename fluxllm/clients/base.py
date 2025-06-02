import asyncio
import os
import random
from typing import Any, Dict, List, Optional

from aiolimiter import AsyncLimiter
from openai.types.chat import ChatCompletion

from fluxllm.client_utils import FluxCache, create_progress


class BaseClient:
    """
    A client that allows for concurrent requests to the same API.
    
    Rate limiting is implemented using both queries-per-second (QPS) and 
    queries-per-minute (QPM) controls. If both are specified, both limits 
    will be enforced simultaneously. Concurrency is automatically calculated 
    based on the more restrictive of the two rate limits.
    """

    def __init__(
        self,
        cache_file: str | None = None,
        max_retries: int | None = None,
        max_qps: Optional[float] = None,  # None means no QPS rate limiting
        max_qpm: float = 100,  # Default to 100 requests per minute
        progress_msg: str = "Requesting...",
    ):
        """
        Initialize the client.

        Args:
            cache_file: the cache file to use.
            max_retries: the maximum number of retries to make, defaults to `None`.
            max_qps: maximum queries per second (rate limit), defaults to `None`.
            max_qpm: maximum queries per minute (rate limit), defaults to `100`.
            progress_msg: the message to display in the progress bar, defaults to `Requesting...`.
        """
        if cache_file is None:
            cache_file = os.getenv("CACHE_FILE", "cache.jsonl")

        # initialize the cache client
        self.cache = FluxCache(cache_file)
        self.lock = asyncio.Lock()
        self.max_retries = max_retries
        self.max_qps = max_qps
        self.max_qpm = max_qpm
        
        # Configure rate limiters for both QPS and QPM if specified
        self.qps_limiter = None
        self.qpm_limiter = None
        
        if max_qps is not None and max_qps > 0:
            self.qps_limiter = AsyncLimiter(max_rate=max_qps, time_period=1.0)
            
        if max_qpm > 0:
            self.qpm_limiter = AsyncLimiter(max_rate=max_qpm, time_period=60.0)
            
        # Calculate optimal concurrency based on rate limits
        if max_qps is not None and max_qps > 0:
            qps_concurrency = max(1, int(max_qps))
        else:
            qps_concurrency = float('inf')
            
        if max_qpm > 0:
            qpm_concurrency = max(1, int(max_qpm / 60))
        else:
            qpm_concurrency = float('inf')
            
        # Use the more restrictive concurrency limit
        self.concurrency = min(qps_concurrency, qpm_concurrency)
        if self.concurrency == float('inf'):
            self.concurrency = 1  # Default if no rate limits are set
            
        self.progress_msg = progress_msg

    async def save_to_cache_thread_safe(self, sample: Dict, response: Dict, save_request: bool = False):
        async with self.lock:
            self.cache.save_to_cache(sample, response, save_request=save_request)

    async def execute_with_rate_limiting(self, request: Dict, **kwargs):
        """
        Execute a request with rate limiting applied.
        
        This method applies both QPS and QPM rate limiting if configured.
        
        Args:
            request: The request to execute
            **kwargs: Additional arguments to pass to make_request_async
            
        Returns:
            The response from make_request_async
        """
        # Apply QPS rate limiting if configured
        if self.qps_limiter is not None:
            async with self.qps_limiter:
                # Apply QPM rate limiting if configured
                if self.qpm_limiter is not None:
                    async with self.qpm_limiter:
                        return await self.make_request_async(request, **kwargs)
                else:
                    return await self.make_request_async(request, **kwargs)
        # Only apply QPM rate limiting if QPS is not configured
        elif self.qpm_limiter is not None:
            async with self.qpm_limiter:
                return await self.make_request_async(request, **kwargs)
        # No rate limiting
        else:
            return await self.make_request_async(request, **kwargs)

    async def request_async(
        self,
        requests: List[Dict[str, Any]],
        save_request: bool = False,
        **kwargs,
    ) -> None:
        """
        Make requests for all uncached samples in given list.
        This function is designed to handle the generation of multiple responses in a batch.
        It uses rate limiters to control the request rate.

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
                response = await self.execute_with_rate_limiting(request, **kwargs)
                
                if response is not None:
                    await self.save_to_cache_thread_safe(request, response, save_request=save_request)
                    progress.advance(task)
                    print(f"Request succeeded for request: {self.cache.hash(request)}", flush=True)
                else:
                    await after_failure(request)
                task_queue.task_done()

        with create_progress() as progress:
            task = progress.add_task(f"[cyan]{self.progress_msg}", total=len(requests))
            workers = [asyncio.create_task(worker()) for _ in range(self.concurrency)]

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
