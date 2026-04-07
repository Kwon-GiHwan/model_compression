from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable


class BaseDataLoader(ABC):
    """
    모든 데이터 로더의 공통 인터페이스.
    이미지/NLP/오디오/tabular 등 태스크에 무관하게 Iterable 데이터를 반환.
    """

    @abstractmethod
    def get_dataloader(self) -> Iterable: ...

    @classmethod
    def from_config(cls, config: Any) -> "BaseDataLoader":
        raise NotImplementedError
