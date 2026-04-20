from abc import ABC, abstractmethod
from typing import Iterable

import torch.nn as nn

from config import Config


class BaseMethod(ABC):
    """
    모든 압축 방법론의 공통 인터페이스.

    apply() 파라미터:
      student    : 압축 대상 모델 (nn.Module)
      teacher    : KD 계열에서 사용. 그 외에는 None
      dataloader : 학습/calibration이 필요한 방법론에서 사용. 그 외에는 None

    외부 의존성 플래그:
      requires_teacher()    : teacher 로딩 필요 여부 (기본 False)
      requires_dataloader() : dataloader 로딩 필요 여부 (기본 False)
      main.py 는 이 플래그를 보고 필요한 자원만 로드한다.

    새 방법론 추가 (3단계):
      1. methods/<category>/ 에 구현체 작성. 기존 카테고리(pruning, quantization,
         distillation)에 속하지 않으면 새 디렉토리 생성 가능 — Registry 는
         카테고리명을 특별 취급하지 않는다.
      2. methods/registry.py 에 `_registry.register("<category>.<name>")(Class)` 추가.
      3. .env 에서 METHOD 값을 `<category>.<name>` 으로 설정.

    구현 가이드:
      - apply() 는 압축된 nn.Module 을 반환한다. 내부에서 student 를 직접 변형하지
        말고 copy.deepcopy 후 처리하는 것이 기존 구현체의 관례.
      - validate(config) 에서 필수 설정 검증. 실패 시 ValueError.
      - from_config(config) 는 필수. Config 의 <CATEGORY>_ 접두사 필드 및
        필요 시 config.data.*, config.train.* 을 조합.
      - 학습 루프가 필요하면 methods/utils.py 의 unpack_batch,
        forward_and_extract_logits 를 재사용.
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
