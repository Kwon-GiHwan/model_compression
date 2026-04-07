from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> object:
        """Teacher 모델 반환"""
        ...

    @classmethod
    def from_config(cls, config) -> "BaseLoader":
        raise NotImplementedError
