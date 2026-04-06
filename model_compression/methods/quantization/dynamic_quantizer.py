import copy

import torch
import torch.nn as nn

from model_compression.methods.base_method import BaseMethod


class DynamicQuantizer(BaseMethod):
    """
    Dynamic Post-Training Quantization.
    calibration 데이터 없이 Linear/LSTM/GRU 레이어를 동적 양자화.
    모든 모델 아키텍처에 적용 가능.
    """

    def __init__(self, dtype: str = "qint8"):
        self.dtype = getattr(torch, dtype, torch.qint8)

    def apply(self, student, teacher=None, dataloader=None):
        model = copy.deepcopy(student).cpu().eval()
        quantized = torch.ao.quantization.quantize_dynamic(
            model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=self.dtype
        )
        print("[DynamicQuantizer] 완료")
        return quantized

    def validate(self, config):
        valid_dtypes = ("qint8", "float16")
        if config.QUANT_DTYPE not in valid_dtypes:
            raise ValueError(f"QUANT_DTYPE는 {valid_dtypes} 중 하나여야 합니다")
