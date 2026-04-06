from abc import ABC, abstractmethod


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> object:
        """Teacher 모델 반환"""
        ...
