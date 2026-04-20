from abc import ABC, abstractmethod

from config import Config


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> object:
        """Teacher 모델 반환"""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: Config) -> "BaseLoader": ...
