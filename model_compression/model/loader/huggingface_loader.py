from huggingface_hub import hf_hub_download
from transformers import AutoModel

from model_compression.model.loader.base_loader import BaseLoader


class HuggingFaceLoader(BaseLoader):
    def __init__(self, repo_id: str, filename: str = None):
        self.repo_id = repo_id
        self.filename = filename

    def load(self):
        print(f"[HuggingFaceLoader] HF Hub 로드: {self.repo_id}")

        if self.filename:
            path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
            import torch

            model = torch.load(path, map_location="cpu")
        else:
            model = AutoModel.from_pretrained(self.repo_id)

        model.eval()
        return model
