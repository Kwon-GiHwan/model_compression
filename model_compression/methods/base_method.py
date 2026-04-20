from abc import ABC, abstractmethod
from typing import Iterable

import torch.nn as nn

from config import Config


class BaseMethod(ABC):
    """
    모든 압축 방법론의 공통 인터페이스.

    student : 압축 대상 모델 객체
    teacher : KD 계열에서 사용, Pruning은 None
    dataloader : 학습이 필요한 방법론에서 사용

    새 방법론 추가:
      1. methods/pruning/ 또는 methods/distillation/ 에 구현체 작성
      2. methods/registry.py 에 분기 추가
      3. .env 에서 METHOD 값 변경
    """

    @abstractmethod
    def apply(
        self, student: nn.Module, teacher: nn.Module | None = None, dataloader: Iterable | None = None
    ) -> nn.Module: ...

    def validate(self, config: Config) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: Config) -> "BaseMethod": ...

    @classmethod
    def requires_teacher(cls) -> bool:
        """이 방법론이 teacher 모델을 필요로 하는지 여부"""
        return False

    @classmethod
    def requires_dataloader(cls) -> bool:
        """이 방법론이 dataloader를 필요로 하는지 여부"""
        return False
