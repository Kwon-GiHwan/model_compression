from model_compression.methods.base_method import BaseMethod
from model_compression.methods.distillation.response_based import ResponseBasedDistiller
from model_compression.methods.pruning.attention_head_pruner import AttentionHeadPruner
from model_compression.methods.pruning.magnitude_pruner import MagnitudePruner
from model_compression.methods.quantization.dynamic_quantizer import DynamicQuantizer
from model_compression.methods.quantization.static_quantizer import StaticQuantizer
from model_compression.methods.quantization.qat_quantizer import QATQuantizer
from model_compression.registry import Registry
from config import Config

_registry = Registry("METHOD")
_registry.register("pruning.magnitude")(MagnitudePruner)
_registry.register("pruning.attention_head")(AttentionHeadPruner)
_registry.register("distillation.response_based")(ResponseBasedDistiller)
_registry.register("quantization.dynamic")(DynamicQuantizer)
_registry.register("quantization.static")(StaticQuantizer)
_registry.register("quantization.qat")(QATQuantizer)


def get_method(config: Config) -> BaseMethod:
    """METHOD 환경변수에 따라 방법론 구현체 반환."""
    cls = _registry.get(config.METHOD)
    return cls.from_config(config)
