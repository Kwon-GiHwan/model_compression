from model_compression.model.loader.huggingface_loader import HuggingFaceLoader
from model_compression.model.loader.local_loader import LocalLoader
from model_compression.registry import Registry
from config import Config

_registry = Registry("TEACHER_LOADER")
_registry._entries["local"] = LocalLoader
_registry._entries["huggingface"] = HuggingFaceLoader


def get_teacher_loader(config: Config):
    cls = _registry.get(config.TEACHER_LOADER)
    if config.TEACHER_LOADER == "huggingface":
        return cls(
            repo_id=config.TEACHER_HF_REPO,
            filename=config.TEACHER_HF_FILENAME,
        )
    return cls(config.TEACHER_MODEL_PATH)
