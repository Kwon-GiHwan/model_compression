from model_compression.model.loader.huggingface_loader import HuggingFaceLoader
from model_compression.model.loader.local_loader import LocalLoader
from model_compression.registry import Registry
from config import Config

_registry = Registry("TEACHER_LOADER")
_registry.register("local")(LocalLoader)
_registry.register("huggingface")(HuggingFaceLoader)


def get_teacher_loader(config: Config):
    cls = _registry.get(config.TEACHER_LOADER)
    return cls.from_config(config)
