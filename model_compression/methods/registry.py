from model_compression.methods.base_method import BaseMethod
from model_compression.methods.distillation.response_based import ResponseBasedDistiller
from model_compression.methods.pruning.attention_head_pruner import AttentionHeadPruner
from model_compression.methods.pruning.magnitude_pruner import MagnitudePruner
from model_compression.registry import Registry
from config import Config

_registry = Registry("METHOD")
_registry.register("pruning.magnitude")(MagnitudePruner)
_registry.register("pruning.attention_head")(AttentionHeadPruner)
_registry.register("distillation.response_based")(ResponseBasedDistiller)


def get_method(config: Config) -> BaseMethod:
    """
    METHOD 환경변수에 따라 방법론 구현체 반환.

    새 방법론 추가:
      1. methods/pruning/ 또는 methods/distillation/ 에 구현체 작성
      2. _registry.register() 데코레이터로 등록
      3. .env 에서 METHOD 변경
    """
    # Validate key via registry (raises ValueError on unknown method)
    _registry.get(config.METHOD)

    is_nlp = config.DATASET_TYPE == "hf_datasets"

    if config.METHOD == "pruning.magnitude":
        return MagnitudePruner(
            pruning_ratio=config.PRUNING_RATIO,
            input_size=config.BENCHMARK_INPUT_SIZE,
            is_nlp=is_nlp,
        )
    if config.METHOD == "pruning.attention_head":
        return AttentionHeadPruner(
            pruning_ratio=config.PRUNING_RATIO,
        )
    return ResponseBasedDistiller(
        epochs=config.TRAIN_EPOCHS,
        device=config.TRAIN_DEVICE,
        temperature=config.DISTILL_TEMPERATURE,
        alpha=config.DISTILL_ALPHA,
        lr=config.TRAIN_LR,
    )
