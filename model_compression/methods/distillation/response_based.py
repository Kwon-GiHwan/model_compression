import torch
import torch.nn.functional as F

from model_compression.methods.base_method import BaseMethod
from model_compression.methods.utils import extract_logits


class ResponseBasedDistiller(BaseMethod):
    """
    Response-based Knowledge Distillation.
    Teacher output logits → soft label로 student 학습.
    이미지 분류 / NLP 분류 공통 적용 가능.
    """

    def __init__(
        self,
        epochs: int,
        device: str,
        temperature: float = 4.0,
        alpha: float = 0.7,
        lr: float = 1e-4,
    ):
        self.epochs = epochs
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.lr = lr

    @classmethod
    def requires_teacher(cls) -> bool:
        return True

    @classmethod
    def requires_dataloader(cls) -> bool:
        return True

    def apply(self, student, teacher, dataloader=None):
        if teacher is None:
            raise ValueError("[Distiller] teacher 모델이 필요합니다")
        if dataloader is None:
            raise ValueError("[Distiller] dataloader가 필요합니다")

        student = student.to(self.device).train()
        teacher = teacher.to(self.device).eval()
        optimizer = torch.optim.AdamW(student.parameters(), lr=self.lr)
        T = self.temperature

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in dataloader:
                # 이미지: (imgs, labels) / NLP: dict with input_ids 등
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    with torch.no_grad():
                        teacher_logits = extract_logits(teacher(inputs))
                    student_logits = extract_logits(student(inputs))
                else:
                    inputs = {
                        k: v.to(self.device) for k, v in batch.items() if k != "label"
                    }
                    labels = batch["label"].to(self.device)
                    with torch.no_grad():
                        teacher_logits = extract_logits(teacher(**inputs))
                    student_logits = extract_logits(student(**inputs))

                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T**2)

                ce_loss = F.cross_entropy(student_logits, labels)
                loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"  Epoch [{epoch + 1}/{self.epochs}] Loss: {total_loss:.4f}")

        print("[Distiller] 완료")
        return student

    @classmethod
    def from_config(cls, config):
        return cls(
            epochs=config.TRAIN_EPOCHS,
            device=config.TRAIN_DEVICE,
            temperature=config.DISTILL_TEMPERATURE,
            alpha=config.DISTILL_ALPHA,
            lr=config.TRAIN_LR,
        )

    def validate(self, config):
        if not config.TEACHER_LOADER:
            raise ValueError("Distillation은 TEACHER_LOADER 설정이 필요합니다")
