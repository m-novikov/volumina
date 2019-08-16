import abc
import threading
from collections import OrderedDict
from typing import NamedTuple, Any, Type, TypeVar, Callable


T = TypeVar("T")


class ICache(abc.ABC):
    @abc.abstractmethod
    def get(self, key: str, default=None) -> Any:
        ...

    @abc.abstractmethod
    def set(self, key: str, value: Any) -> Any:
        ...

    @abc.abstractmethod
    def delete(self, key: str) -> None:
        ...

    @abc.abstractmethod
    def touch(self, key: str) -> None:
        ...

    @abc.abstractmethod
    def __contains__(self, key: str) -> bool:
        ...


class KVCache(ICache):
    _MISSING = object()

    class _Entry(NamedTuple):
        obj: Any
        size: int  # bytes

    def __init__(self, mem_limit=256 * 1024 * 1024):
        self._cache = OrderedDict()
        self._mem_limit = mem_limit
        self._mem = 0
        self._lock = threading.RLock()
        self._cachable_types: Dict[T, Callable[[T], int]] = {}

    def get(self, key: str, default=None) -> Any:
        with self._lock:
            entry = self._cache.get(key, self._MISSING)
            self._cache.move_to_end(key)

        if entry is not self._MISSING:
            return entry.obj
        else:
            return default

    def delete(self, key: str) -> None:
        with self._lock:
            self._cache.pop(key, None)

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in list(self._cache.keys()):
                if key.startswith(prefix):
                    self.delete(key)

    def keys(self):
        return list(self._cache.keys())

    def set(self, key, value) -> None:
        with self._lock:
            old_entry = self._cache.get(key)

            if old_entry is not None:
                self._mem -= old_entry.size

            get_size_fn = self._cachable_types.get(type(value))
            if get_size_fn is None:
                raise ValueError(f"Unknown type {type(value)}")

            size = get_size_fn(value)
            self._mem += size
            self._cache[key] = self._Entry(value, size)
            self._cache.move_to_end(key)

    def __contains__(self, key) -> bool:
        return key in self._cache

    def __len__(self):
        return len(self._cache)

    def touch(self, key: str) -> None:
        with self._lock:
            self._cache.move_to_end(key)

    def register_type(self, _type: Type[T], get_size_fn: Callable[[T], int]) -> None:
        self._cachable_types[_type] = get_size_fn

    @property
    def used_memory(self) -> int:
        return self._mem

    def clean(self) -> None:
        with self._lock:
            while self._mem > self._mem_limit:
                key, entry = next(iter(self._cache.items()))
                del self._cache[key]
                self._mem -= entry.size
