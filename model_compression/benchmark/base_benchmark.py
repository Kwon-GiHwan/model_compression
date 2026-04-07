from __future__ import annotations

from abc import ABC, abstractmethod


class BaseBenchmark(ABC):
    """
    모든 벤치마크의 공통 인터페이스.
    """

    @abstractmethod
    def run(self, model, config) -> dict:
        """
        벤치마크 실행 후 결과를 dict로 반환.
        """
        ...

    @classmethod
    def from_config(cls, config) -> "BaseBenchmark":
        raise NotImplementedError
