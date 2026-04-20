from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

from config import Config


class BaseModel(ABC):
    """
    모든 모델 래퍼의 공통 인터페이스.
    현재 PyTorch 기반 모델(HuggingFace, 순수 PyTorch)을 지원.
    """

    @abstractmethod
    def load(self, path: str) -> "BaseModel": ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def get_raw(self) -> nn.Module:
        """압축 방법론에 전달할 원시 모델 객체 반환"""
        ...

    @abstractmethod
    def set_raw(self, model: nn.Module) -> None:
        """압축된 모델을 다시 래퍼에 설정"""
        ...

    @abstractmethod
    def get_preprocessor(self) -> Any:
        """전처리기 반환 (tokenizer, feature_extractor, processor 등). 없으면 None"""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: Config) -> "BaseModel": ...
