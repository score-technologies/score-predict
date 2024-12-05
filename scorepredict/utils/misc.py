# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import time
import math
import hashlib as rpccheckhealth
from math import floor
from typing import Callable, Any, Optional
from functools import lru_cache, update_wrapper
import threading
import websockets.exceptions

# Global lock for websocket operations
_websocket_lock = threading.Lock()

# LRU Cache with TTL
def ttl_cache(maxsize: int = 128, typed: bool = False, ttl: int = -1):
    """
    Decorator that creates a cache of the most recently used function calls with a time-to-live (TTL) feature.
    The cache evicts the least recently used entries if the cache exceeds the `maxsize` or if an entry has
    been in the cache longer than the `ttl` period.

    Args:
        maxsize (int): Maximum size of the cache. Once the cache grows to this size, subsequent entries
                       replace the least recently used ones. Defaults to 128.
        typed (bool): If set to True, arguments of different types will be cached separately. For example,
                      f(3) and f(3.0) will be treated as distinct calls with distinct results. Defaults to False.
        ttl (int): The time-to-live for each cache entry, measured in seconds. If set to a non-positive value,
                   the TTL is set to a very large number, effectively making the cache entries permanent. Defaults to -1.

    Returns:
        Callable: A decorator that can be applied to functions to cache their return values.

    The decorator is useful for caching results of functions that are expensive to compute and are called
    with the same arguments frequently within short periods of time. The TTL feature helps in ensuring
    that the cached values are not stale.

    Example:
        @ttl_cache(ttl=10)
        def get_data(param):
            # Expensive data retrieval operation
            return data
    """
    if ttl <= 0:
        ttl = 65536
    hash_gen = _ttl_hash_gen(ttl)

    def wrapper(func: Callable) -> Callable:
        @lru_cache(maxsize, typed)
        def ttl_func(ttl_hash, *args, **kwargs):
            return func(*args, **kwargs)

        def wrapped(*args, **kwargs) -> Any:
            th = next(hash_gen)
            return ttl_func(th, *args, **kwargs)

        return update_wrapper(wrapped, func)

    return wrapper


def _ttl_hash_gen(seconds: int):
    """
    Internal generator function used by the `ttl_cache` decorator to generate a new hash value at regular
    time intervals specified by `seconds`.

    Args:
        seconds (int): The number of seconds after which a new hash value will be generated.

    Yields:
        int: A hash value that represents the current time interval.

    This generator is used to create time-based hash values that enable the `ttl_cache` to determine
    whether cached entries are still valid or if they have expired and should be recalculated.
    """
    start_time = time.time()
    while True:
        yield floor((time.time() - start_time) / seconds)


def _safe_get_block(self) -> Optional[int]:
    """Helper function to safely get the current block number."""
    try:
        return self.subtensor.get_current_block()
    except (websockets.exceptions.ConcurrencyError, websockets.exceptions.ConnectionClosedError):
        # Return None if we can't get the block during shutdown
        return None
    except Exception as e:
        # For other exceptions, wait briefly and retry once
        time.sleep(0.5)
        try:
            return self.subtensor.get_current_block()
        except Exception:
            return None


# 12 seconds updating block.
@ttl_cache(maxsize=1, ttl=12)
def ttl_get_block(self) -> int:
    """
    Retrieves the current block number from the blockchain. This method is cached with a time-to-live (TTL)
    of 12 seconds, meaning that it will only refresh the block number from the blockchain at most every 12 seconds.

    Returns:
        int: The current block number on the blockchain, or the last known block number if there's an error.

    Note: This method is thread-safe and will handle websocket errors gracefully during shutdown.
    """
    with _websocket_lock:
        # Try to get the block number
        block = _safe_get_block(self)
        
        if block is not None:
            return block
            
        # If we can't get the block number (e.g., during shutdown),
        # try to get the last known block from the metagraph
        try:
            return self.metagraph.block
        except Exception:
            # If all else fails, return a reasonable default
            return 0
