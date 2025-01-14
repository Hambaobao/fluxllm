import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any


class CacheClient:

    def __init__(self, cache_file: str | None = None):
        """
        Initialize the API cache manager

        Args:
            cache_file: path to the cache file
        """
        if cache_file is None:
            cache_file = os.getenv("CACHE_FILE", "cache.jsonl")
        self.cache_file = Path(cache_file)
        self.cache_data: Dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self):
        """
        Load the cache file
        """
        cache_data = {}
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    cache_data[data['id']] = data
            print(f"Found {len(cache_data)} cached samples")
        else:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
        self.cache_data = cache_data

    def hash(self, sample: Dict) -> str:
        """
        Generate a unique hash ID for the input sample

        Args:
            sample: request sample
        Returns:
            generated hash ID
        """
        # Convert the data to a sorted JSON string to ensure the same data generates the same hash
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

        self.cache_data[sample_id] = cache_entry

        # Append to the cache file
        with open(self.cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(cache_entry, ensure_ascii=False) + '\n')

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

    def collect_result(self, sample: Dict) -> Dict:
        """
        Get the cached response corresponding to the sample.

        Args:
            sample: request sample
        Returns:
            cached response associated with the sample
        """
        sample_id = self.hash(sample)
        assert sample_id in self.cache_data, f"Sample {sample_id} not found in cache"
        return self.cache_data[sample_id]['response']

    def collect_results(self, samples: List[Dict], ignore_missing: bool = False) -> List[Dict]:
        """
        Get the cached responses for a list of samples

        Args:
            samples: list of samples to get responses for
            ignore_missing: if True, ignore samples that are not found in the cache
        Returns:
            list of cached responses
        """
        results = []
        for sample in samples:
            if self.is_cached(sample):
                results.append(self.collect_result(sample))
            elif not ignore_missing:
                raise ValueError(f"Sample {sample} not found in cache")
        return results
