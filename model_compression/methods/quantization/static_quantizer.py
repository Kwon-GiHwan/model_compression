import copy
from typing import Iterable

import torch
import torch.nn as nn

from config import Config
from model_compression.methods.base_method import BaseMethod


class StaticQuantizer(BaseMethod):
    """
    Static Post-Training Quantization.
    calibration 데이터로 활성화 통계를 수집하여 정적 양자화 수행.
    """

    def __init__(self, backend: str = "x86", calibration_batches: int = 100):
        self.backend = backend
        self.calibration_batches = calibration_batches

    @classmethod
    def requires_dataloader(cls) -> bool:
        return True

    def apply(self, student: nn.Module, teacher: nn.Module | None = None, dataloader: Iterable | None = None) -> nn.Module:
        if dataloader is None:
            raise ValueError("[StaticQuantizer] calibration용 dataloader가 필요합니다")

        model = copy.deepcopy(student).cpu().eval()
        model.qconfig = torch.ao.quantization.get_default_qconfig(self.backend)
        prepared = torch.ao.quantization.prepare(model)

        # Calibration: forward pass로 활성화 통계 수집
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.calibration_batches:
                    break
                if isinstance(batch, (list, tuple)):
                    prepared(batch[0].cpu())
                else:
                    inputs = {k: v.cpu() for k, v in batch.items() if k != "label"}
                    prepared(**inputs)

        quantized = torch.ao.quantization.convert(prepared)
        print(f"[StaticQuantizer] 완료 (calibration batches: {min(i + 1, self.calibration_batches)})")
        return quantized

    @classmethod
    def from_config(cls, config: Config) -> "StaticQuantizer":
        return cls(
            backend=config.QUANT_BACKEND,
            calibration_batches=config.QUANT_CALIBRATION_BATCHES,
        )

    def validate(self, config: Config) -> None:
        valid_backends = ("x86", "fbgemm", "qnnpack")
        if config.QUANT_BACKEND not in valid_backends:
            raise ValueError(f"QUANT_BACKEND는 {valid_backends} 중 하나여야 합니다")
        if config.QUANT_CALIBRATION_BATCHES <= 0:
            raise ValueError("QUANT_CALIBRATION_BATCHES는 0보다 커야 합니다")
