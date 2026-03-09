from model_compression.methods.base_method import BaseMethod
from model_compression.methods.distillation.response_based import ResponseBasedDistiller
from model_compression.methods.pruning.attention_head_pruner import AttentionHeadPruner
from model_compression.methods.pruning.magnitude_pruner import MagnitudePruner
from config import Config


def get_method(config: Config) -> BaseMethod:
    """
    METHOD 환경변수에 따라 방법론 구현체 반환.

    새 방법론 추가:
      1. methods/pruning/ 또는 methods/distillation/ 에 구현체 작성
      2. 아래 registry에 키-값 추가
      3. .env 에서 METHOD 변경
    """
    is_nlp = config.DATASET_TYPE == "hf_datasets"

    registry = {
        "pruning.magnitude": lambda: MagnitudePruner(
            pruning_ratio=config.PRUNING_RATIO,
            input_size=config.BENCHMARK_INPUT_SIZE,
            is_nlp=is_nlp,
        ),
        "pruning.attention_head": lambda: AttentionHeadPruner(
            pruning_ratio=config.PRUNING_RATIO,
        ),
        "distillation.response_based": lambda: ResponseBasedDistiller(
            epochs=config.TRAIN_EPOCHS,
            device=config.TRAIN_DEVICE,
            temperature=config.DISTILL_TEMPERATURE,
            alpha=config.DISTILL_ALPHA,
            lr=config.TRAIN_LR,
        ),
    }

    if config.METHOD not in registry:
        raise ValueError(
            f"지원하지 않는 METHOD: {config.METHOD}\n사용 가능: {list(registry.keys())}"
        )

    return registry[config.METHOD]()
