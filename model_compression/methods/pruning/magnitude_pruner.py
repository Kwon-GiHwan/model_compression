import copy
from typing import Iterable

import torch
import torch.nn as nn
import torch_pruning as tp

from config import Config
from model_compression.methods.base_method import BaseMethod


class MagnitudePruner(BaseMethod):
    """
    Structured Magnitude Pruning.
    이미지 모델 / Transformer 모두 적용 가능.
    dependency graph를 자동 추적하므로 아키텍처 무관.
    """

    def __init__(
        self, pruning_ratio: float, input_size: int = 224, is_nlp: bool = False, max_length: int = 128
    ):
        self.pruning_ratio = pruning_ratio
        self.input_size = input_size
        self.is_nlp = is_nlp
        self.max_length = max_length

    def apply(self, student: nn.Module, teacher: nn.Module | None = None, dataloader: Iterable | None = None) -> nn.Module:
        model = copy.deepcopy(student).cpu().eval()

        # 입력 예시: NLP는 토큰 ID, 이미지는 픽셀
        if self.is_nlp:
            example_input = {"input_ids": torch.zeros(1, self.max_length, dtype=torch.long)}
        else:
            example_input = torch.randn(1, 3, self.input_size, self.input_size)

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_input,
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=self.pruning_ratio,
            ignored_layers=[],  # 필요 시 외부에서 주입 가능
        )

        print(f"[MagnitudePruner] Pruning 시작 (ratio={self.pruning_ratio})")
        pruner.step()
        print("[MagnitudePruner] 완료")
        return model

    @classmethod
    def from_config(cls, config: Config) -> "MagnitudePruner":
        is_nlp = config.data.type == "hf_datasets"
        return cls(
            pruning_ratio=config.PRUNING_RATIO,
            input_size=config.INPUT_SIZE,
            is_nlp=is_nlp,
            max_length=config.data.max_length,
        )

    def validate(self, config: Config) -> None:
        if not 0 < config.PRUNING_RATIO < 1:
            raise ValueError("PRUNING_RATIO는 0~1 사이여야 합니다")
