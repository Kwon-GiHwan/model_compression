from model_compression.data.base_dataloader import BaseDataLoader
from model_compression.data.image_dataloader import ImageDataLoader
from model_compression.data.nlp_dataloader import NLPDataLoader
from model_compression.registry import Registry
from config import Config

_registry = Registry("DATASET_TYPE")
_registry.register("hf_datasets")(NLPDataLoader)
_registry.register("torchvision")(ImageDataLoader)
_registry.register("local_folder")(ImageDataLoader)

_NLP_KEYS = {"hf_datasets"}
_IMAGE_KEYS = {"torchvision", "local_folder"}


def get_dataloader(config: Config, tokenizer=None) -> BaseDataLoader:
    """
    DATASET_TYPE 환경변수에 따라 DataLoader 반환.
    새 데이터 소스 추가: 구현체 작성 후 decorator로 등록.
    """
    # Validate key via registry (raises ValueError on unknown type)
    _registry.get(config.DATASET_TYPE)

    if config.DATASET_TYPE in _NLP_KEYS:
        return NLPDataLoader(
            dataset_name=config.DATASET_NAME,
            tokenizer_path=config.MODEL_PATH,
            dataset_config=config.DATASET_CONFIG,
            split=config.DATASET_SPLIT,
            batch_size=config.DATASET_BATCH_SIZE,
            max_length=config.DATASET_MAX_LENGTH,
        )
    return ImageDataLoader(
        dataset_path=config.DATASET_PATH,
        task=config.TASK,
        input_size=config.BENCHMARK_INPUT_SIZE,
        batch_size=config.DATASET_BATCH_SIZE,
        split=config.DATASET_SPLIT,
    )
