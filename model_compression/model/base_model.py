from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """
    모든 모델 래퍼의 공통 인터페이스.
    프레임워크(PyTorch, TensorFlow, JAX, ONNX 등)에 무관하게 동일하게 동작.
    """

    @abstractmethod
    def load(self, path: str) -> "BaseModel": ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def get_raw(self) -> Any:
        """압축 방법론에 전달할 원시 모델 객체 반환"""
        ...

    def set_raw(self, model: Any) -> None:
        """압축된 모델을 다시 래퍼에 설정"""
        raise NotImplementedError

    def get_preprocessor(self) -> Any:
        """전처리기 반환 (tokenizer, feature_extractor, processor 등). 없으면 None"""
        raise NotImplementedError

    def get_tokenizer(self) -> Any:
        """Deprecated: use get_preprocessor() instead."""
        return self.get_preprocessor()
