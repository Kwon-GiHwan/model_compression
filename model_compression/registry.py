"""범용 Registry 유틸리티. 데코레이터 기반 동적 등록을 지원."""
from typing import Any, Callable


class Registry:
    """키-팩토리 매핑을 관리하는 범용 레지스트리."""

    def __init__(self, name: str):
        self.name = name
        self._entries: dict[str, Callable[..., Any]] = {}

    def register(self, key: str) -> Callable:
        """데코레이터: 팩토리 함수 또는 클래스를 등록."""
        def decorator(cls_or_fn: Callable) -> Callable:
            self._entries[key] = cls_or_fn
            return cls_or_fn
        return decorator

    def get(self, key: str) -> Callable:
        if key not in self._entries:
            available = list(self._entries.keys())
            raise ValueError(f"지원하지 않는 {self.name}: {key}\n사용 가능: {available}")
        return self._entries[key]

    def keys(self) -> list[str]:
        return list(self._entries.keys())
