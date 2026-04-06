from __future__ import annotations

from abc import ABC, abstractmethod


class BaseReporter(ABC):
    """
    모든 리포터의 공통 인터페이스.
    """

    @abstractmethod
    def report(self, original_result: dict, compressed_result: dict):
        """
        벤치마크 결과를 출력하거나 저장.
        """
        ...
