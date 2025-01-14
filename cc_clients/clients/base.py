from typing import List, Dict, Any, Callable

import os
import random
import asyncio

from cc_clients.client_utils import CacheClient, create_progress


class ConcurrentClient:
    """
    A client that allows for concurrent requests to the same API.
    """

    def __init__(
        self,
        collate_fn: Callable[[Dict, Dict], Dict],
        cache_file: str | None = None,
        max_retries: int | None = None,
    ):
        """
        Initialize the client.

        Args:
            collate_fn: the function to collate the messages for each sample.
            cache_file: the cache file to use.
            max_retries: the maximum number of retries to make, defaults to `None`.
        """
        if cache_file is None:
            cache_file = os.getenv("CACHE_FILE", "cache.jsonl")

        # initialize the cache client
        self.collate_fn = collate_fn
        self.cache = CacheClient(cache_file)
        self.lock = asyncio.Lock()
        self.max_retries = max_retries

    async def save_to_cache_thread_safe(self, sample: Dict, response: Dict):
        async with self.lock:
            self.cache.save_to_cache(sample, response)

    async def _generate_async(
        self,
        samples: List[Dict],
        **kwargs,
    ) -> None:
        """
        Generate responses for all uncached samples in given list.
        This function is designed to handle the generation of multiple responses in a batch.
        It uses a semaphore to control the number of concurrent requests.

        Args:
            samples: list of samples to generate responses for
            **kwargs: additional arguments to pass to request_async
        """

        # create a queue to hold the samples
        task_queue = asyncio.Queue()
        for sample in samples:
            await task_queue.put(sample)
        failure_counts = {self.cache.hash(sample): 0 for sample in samples}

        # limit the number of concurrent requests
        semaphore = asyncio.Semaphore(kwargs.get("batch_size", 1))

        async def after_failure(sample: Dict):
            failure_counts[self.cache.hash(sample)] += 1
            if self.max_retries is None:
                print(f"Re-queue failed task: {self.cache.hash(sample)}, failed {failure_counts[self.cache.hash(sample)]} times.", flush=True)
                await task_queue.put(sample)
                await asyncio.sleep(random.randint(3, 10))
            else:
                if failure_counts[self.cache.hash(sample)] < self.max_retries:
                    print(f"Re-queue failed task: {self.cache.hash(sample)}, failed {failure_counts[self.cache.hash(sample)]} times.", flush=True)
                    await task_queue.put(sample)
                    await asyncio.sleep(random.randint(3, 10))
                else:
                    print(f"Request failed after {self.max_retries} retries. Adding a dummy response to the cache.", flush=True)
                    response = self.create_dummy_response(**kwargs)
                    await self.save_to_cache_thread_safe(sample, response)

        async def worker():
            while not task_queue.empty():
                sample = await task_queue.get()
                async with semaphore:
                    response = await self.request_async(sample, **kwargs)
                    if response is not None:
                        await self.save_to_cache_thread_safe(sample, response)
                        progress.advance(task_id)
                        print(f"Request succeeded for sample: {self.cache.hash(sample)}", flush=True)
                    else:
                        await after_failure(sample)
                task_queue.task_done()

        with create_progress() as progress:
            progress_prompt = kwargs.get("progress_prompt", self.progress_prompt)
            task_id = progress.add_task(f"[cyan]{progress_prompt}", total=len(samples))
            workers = [asyncio.create_task(worker()) for _ in range(kwargs.get("batch_size", 1))]

            await task_queue.join()
            for worker_task in workers:
                worker_task.cancel()

    def generate(
        self,
        samples: List[Dict],
        **kwargs,
    ) -> None:
        """
        Generate responses for all uncached samples in given list.
        This is a synchronous wrapper around generate_async.

        Args:
            samples: List of samples to generate responses for
            **kwargs: additional arguments to pass to request_async
        """

        # get the samples that are not cached
        samples_to_process = [sample for sample in samples if not self.cache.is_cached(sample)]
        print(f"Remaining {len(samples_to_process)} samples to generate", flush=True)

        # request the responses
        asyncio.run(self._generate_async(samples=samples_to_process, **kwargs))

    def request_async(self, sample: Dict, **kwargs) -> Dict | None:
        """
        Make a single request to the API and cache the response.

        Args:
            sample: Dictionary containing the input data to generate a response for
            **kwargs: Additional arguments that will be passed to the API call if they are in SUPPORT_ARGS

        Returns:
            bool: True if the request was successful and cached, False if it failed
        """
        raise NotImplementedError("Implement this in the subclass")

    def collect_responses(self, samples: List[Dict]) -> List[Any]:
        """
        Get the cached responses for a list of samples.
        """
        raise NotImplementedError("Implement this in the subclass")

    def create_dummy_response(self, **kwargs) -> Any:
        """
        Return a dummy response.
        """
        raise NotImplementedError("Implement this in the subclass")
