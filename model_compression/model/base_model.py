from abc import ABC, abstractmethod

import torch.nn as nn


class BaseModel(ABC):
    """
    모든 모델 래퍼의 공통 인터페이스.
    프레임워크(ultralytics, transformers, 순수 PyTorch)에 무관하게 동일하게 동작.
    """

    @abstractmethod
    def load(self, path: str) -> "BaseModel": ...

    @abstractmethod
    def save(self, path: str): ...

    @abstractmethod
    def get_raw(self) -> nn.Module:
        """압축 방법론에 전달할 순수 PyTorch nn.Module 반환"""
        ...

    @abstractmethod
    def get_tokenizer(self):
        """NLP 모델은 토크나이저 반환, 이미지 모델은 None"""
        ...
