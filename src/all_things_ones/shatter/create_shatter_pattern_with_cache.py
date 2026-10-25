import hashlib
import pickle
from pathlib import Path

from .create_shatter_pattern import create_shatter_pattern
from .model import ShatterPattern


def create_shatter_pattern_with_cache(
    width: int, height: int, num_pieces: int, seed: int
) -> ShatterPattern:
    cache_key = _get_cache_key(width, height, num_pieces, seed)
    pattern = _load_pattern_from_cache(cache_key)
    if pattern is None:
        pattern = create_shatter_pattern(width, height, num_pieces, seed)
        _save_pattern_to_cache(pattern, cache_key)
    return pattern


def _get_cache_key(width: int, height: int, num_pieces: int, seed: int) -> str:
    key_string = f"{width}_{height}_{num_pieces}_{seed}"
    return hashlib.md5(key_string.encode()).hexdigest()


def _get_cache_dir() -> Path:
    cache_dir = Path.cwd() / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_pattern_from_cache(cache_key: str):
    cache_file = _get_cache_dir() / f"{cache_key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None


def _save_pattern_to_cache(pattern, cache_key: str):
    cache_file = _get_cache_dir() / f"{cache_key}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(pattern, f)
