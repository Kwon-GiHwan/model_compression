import torch

from config import Config
from model_compression.model.base_model import BaseModel


class PyTorchModel(BaseModel):
    """
    순수 PyTorch 모델 래퍼.
    torchvision, timm, 커스텀 모델 등 nn.Module이면 모두 수용.
    """

    def __init__(self):
        self._model = None

    def load(self, path: str) -> "PyTorchModel":
        self._model = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(self._model, dict):
            # state_dict만 저장된 경우 — 아키텍처는 별도 제공 필요
            raise ValueError(
                "state_dict만 저장된 파일입니다. "
                "아키텍처 정의 후 load_state_dict()를 직접 사용하세요."
            )
        self._model.eval()
        print(f"[PyTorchModel] 로드 완료: {path}")
        return self

    def save(self, path: str):
        torch.save(self._model, path)
        print(f"[PyTorchModel] 저장 완료: {path}")

    def set_raw(self, model) -> None:
        self._model = model

    def get_raw(self):
        return self._model

    def get_preprocessor(self):
        return None

    @classmethod
    def from_config(cls, config: Config) -> "PyTorchModel":
        instance = cls()
        return instance.load(config.MODEL_PATH)
