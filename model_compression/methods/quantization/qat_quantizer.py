import copy

import torch
import torch.nn.functional as F

from config import Config
from model_compression.methods.base_method import BaseMethod
from model_compression.methods.utils import extract_logits, forward_and_extract_logits, unpack_batch


class QATQuantizer(BaseMethod):
    """
    Quantization-Aware Training.
    fake-quantize 노드를 삽입하여 양자화 효과를 시뮬레이션하며 학습.
    학습 후 실제 양자화 모델로 변환.
    """

    def __init__(
        self,
        backend: str = "x86",
        epochs: int = 10,
        device: str = "cpu",
        lr: float = 1e-4,
    ):
        self.backend = backend
        self.epochs = epochs
        self.device = device
        self.lr = lr

    @classmethod
    def requires_dataloader(cls) -> bool:
        return True

    def apply(self, student, teacher=None, dataloader=None):
        if dataloader is None:
            raise ValueError("[QATQuantizer] 학습용 dataloader가 필요합니다")

        model = copy.deepcopy(student).to(self.device).train()
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(self.backend)
        prepared = torch.ao.quantization.prepare_qat(model)

        optimizer = torch.optim.AdamW(prepared.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in dataloader:
                inputs, labels = unpack_batch(batch, self.device)
                logits = forward_and_extract_logits(prepared, inputs)

                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"  Epoch [{epoch + 1}/{self.epochs}] Loss: {total_loss:.4f}")

        prepared.eval()
        quantized = torch.ao.quantization.convert(prepared.cpu())
        print("[QATQuantizer] 완료")
        return quantized

    @classmethod
    def from_config(cls, config: Config):
        return cls(
            backend=config.QUANT_BACKEND,
            epochs=config.train.epochs,
            device=config.train.device,
            lr=config.train.lr,
        )

    def validate(self, config: Config):
        if config.train.epochs <= 0:
            raise ValueError("TRAIN_EPOCHS는 0보다 커야 합니다")
