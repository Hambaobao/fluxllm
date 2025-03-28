import hashlib
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict


class FluxCache:
    _lock = multiprocessing.Lock()
    _manager = multiprocessing.Manager()

    def __init__(self, cache_file: str | None = None):
        """
        Initialize the API cache manager

        Args:
            cache_file: path to the cache file
        """
        if cache_file is None:
            cache_file = os.getenv("CACHE_FILE", "cache.jsonl")
        self.cache_file = Path(cache_file)
        self.cache_data: Dict[str, Any] = self._manager.dict()
        self._load_cache()

    def _load_cache(self):
        """
        Load the cache file
        """
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = {item["id"]: item for item in (json.loads(line) for line in f)}
            print(f"Found {len(cache_data)} cached samples")
        else:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)

        self.cache_data.update(cache_data)

    def hash(self, sample: Dict) -> str:
        """
        Generate a unique hash ID for the input sample

        Args:
            sample: request sample
        Returns:
            generated hash ID
        """
        sorted_str = json.dumps(sample, sort_keys=True)
        return hashlib.md5(sorted_str.encode('utf-8')).hexdigest()

    def save_to_cache(self, sample: Dict, response: Dict, save_request: bool = False):
        """
        Add the API response to the cache
        
        Args:
            sample: original request sample
            response: API response result
            save_request: whether to save the request sample
        """
        sample_id = self.hash(sample)
        if save_request:
            cache_entry = {'id': sample_id, 'request': sample, 'response': response}
        else:
            cache_entry = {'id': sample_id, 'response': response}

        with self._lock:
            with open(self.cache_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(cache_entry, ensure_ascii=False) + '\n')
            self.cache_data[sample_id] = cache_entry

    def is_cached(self, sample: Dict) -> bool:
        """
        Check if the data is already cached

        Args:
            sample: request sample
        Returns:
            True if the sample is cached, False otherwise
        """
        sample_id = self.hash(sample)
        return sample_id in self.cache_data

    def collect_result(self, sample: Dict) -> Dict | None:
        """
        Get the cached response corresponding to the sample.

        Args:
            sample: request sample
        Returns:
            cached response associated with the sample
        """
        sample_id = self.hash(sample)
        if sample_id not in self.cache_data:
            return None
        return self.cache_data[sample_id]['response']
