from model_compression.model.base_model import BaseModel
from model_compression.model.huggingface_model import HuggingFaceModel
from model_compression.model.pytorch_model import PyTorchModel
from config import Config


def get_model(config: Config) -> BaseModel:
    """
    MODEL_TYPE 환경변수에 따라 모델 래퍼 반환.
    새 모델 타입 추가: 구현체 작성 후 여기에 분기 추가.
    """
    registry = {
        "pytorch": PyTorchModel,
        "huggingface": lambda: HuggingFaceModel(task=config.TASK),
    }

    if config.MODEL_TYPE not in registry:
        raise ValueError(f"지원하지 않는 MODEL_TYPE: {config.MODEL_TYPE}")

    model = registry[config.MODEL_TYPE]()
    return model.load(config.MODEL_PATH)
