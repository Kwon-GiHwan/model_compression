from model_compression.model.loader.huggingface_loader import HuggingFaceLoader
from model_compression.model.loader.local_loader import LocalLoader

from config import Config


def get_teacher_loader(config: Config):
    registry = {
        "local": lambda: LocalLoader(config.TEACHER_MODEL_PATH),
        "huggingface": lambda: HuggingFaceLoader(
            repo_id=config.TEACHER_HF_REPO,
            filename=config.TEACHER_HF_FILENAME,
        ),
    }

    if config.TEACHER_LOADER not in registry:
        raise ValueError(f"지원하지 않는 TEACHER_LOADER: {config.TEACHER_LOADER}")

    return registry[config.TEACHER_LOADER]()
