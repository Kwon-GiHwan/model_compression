from model_compression.data.base_dataloader import BaseDataLoader
from model_compression.data.image_dataloader import ImageDataLoader
from model_compression.data.nlp_dataloader import NLPDataLoader
from model_compression.registry import Registry
from config import Config

_registry = Registry("DATASET_TYPE")
_registry.register("hf_datasets")(NLPDataLoader)
_registry.register("torchvision")(ImageDataLoader)
_registry.register("local_folder")(ImageDataLoader)


def get_dataloader(config: Config) -> BaseDataLoader:
    """DATASET_TYPE 환경변수에 따라 DataLoader 반환."""
    cls = _registry.get(config.DATASET_TYPE)
    return cls.from_config(config)
