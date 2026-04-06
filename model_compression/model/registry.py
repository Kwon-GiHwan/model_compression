from model_compression.model.base_model import BaseModel
from model_compression.model.huggingface_model import HuggingFaceModel
from model_compression.model.pytorch_model import PyTorchModel
from model_compression.registry import Registry
from config import Config

_registry = Registry("MODEL_TYPE")
_registry.register("pytorch")(PyTorchModel)
_registry.register("huggingface")(HuggingFaceModel)


def get_model(config: Config) -> BaseModel:
    """
    MODEL_TYPE 환경변수에 따라 모델 래퍼 반환.
    새 모델 타입 추가: 구현체 작성 후 _registry.register()로 등록.
    """
    # Validate key via registry (raises ValueError on unknown type)
    _registry.get(config.MODEL_TYPE)

    if config.MODEL_TYPE == "huggingface":
        model = HuggingFaceModel(task=config.TASK)
    else:
        model = PyTorchModel()
    return model.load(config.MODEL_PATH)
