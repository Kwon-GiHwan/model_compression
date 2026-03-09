import torch
from transformers import AutoModel

from model_compression.model.loader.base_loader import BaseLoader


class LocalLoader(BaseLoader):
    def __init__(self, model_path: str):
        self.model_path = model_path

    def load(self):
        print(f"[LocalLoader] 로컬 로드: {self.model_path}")
        try:
            # HuggingFace 로컬 디렉토리 시도
            model = AutoModel.from_pretrained(self.model_path)
        except Exception:
            # 순수 PyTorch .pt 파일
            model = torch.load(self.model_path, map_location="cpu")
        model.eval()
        return model
