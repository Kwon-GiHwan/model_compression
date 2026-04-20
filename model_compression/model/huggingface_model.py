import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from config import Config
from model_compression.model.base_model import BaseModel


class HuggingFaceModel(BaseModel):
    """
    HuggingFace Transformers 모델 래퍼.
    NLP (BERT, GPT, LLaMA 등) 및 Vision (ViT, DETR 등) 모두 지원.
    로컬 경로 또는 HF Hub repo ID 모두 수용.
    """

    def __init__(self, task: str = "classification"):
        self._model = None
        self._tokenizer = None
        self.task = task

    def load(self, path: str) -> "HuggingFaceModel":
        print(f"[HuggingFaceModel] 로드: {path}")
        self._model = AutoModel.from_pretrained(path)
        self._model.eval()

        # NLP 모델: 토크나이저 로드 시도
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception:  # noqa: BLE001 — tokenizer may not exist for vision models
            # Vision 모델 등 토크나이저 없는 경우
            try:
                self._tokenizer = AutoProcessor.from_pretrained(path)
            except Exception:  # noqa: BLE001 — processor may not exist either
                self._tokenizer = None

        return self

    def save(self, path: str) -> None:
        self._model.save_pretrained(path)
        if self._tokenizer:
            self._tokenizer.save_pretrained(path)
        print(f"[HuggingFaceModel] 저장 완료: {path}")

    def set_raw(self, model: nn.Module) -> None:
        self._model = model

    def get_raw(self) -> nn.Module:
        return self._model

    def get_preprocessor(self):
        return self._tokenizer

    @classmethod
    def from_config(cls, config: Config) -> "HuggingFaceModel":
        instance = cls(task=config.TASK)
        return instance.load(config.MODEL_PATH)
