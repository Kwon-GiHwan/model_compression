import copy

import torch
import torch.nn.functional as F

from model_compression.methods.base_method import BaseMethod


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

    @staticmethod
    def _extract_logits(output):
        """모델 출력에서 logits 추출."""
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

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
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    logits = self._extract_logits(prepared(inputs))
                else:
                    inputs = {
                        k: v.to(self.device) for k, v in batch.items() if k != "label"
                    }
                    labels = batch["label"].to(self.device)
                    logits = self._extract_logits(prepared(**inputs))

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

    def validate(self, config):
        if config.TRAIN_EPOCHS <= 0:
            raise ValueError("TRAIN_EPOCHS는 0보다 커야 합니다")
