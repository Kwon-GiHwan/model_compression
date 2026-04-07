from model_compression.benchmark.base_benchmark import BaseBenchmark
from model_compression.benchmark.latency_benchmark import LatencyBenchmark
from model_compression.registry import Registry
from config import Config

_registry = Registry("BENCHMARK_TYPE")
_registry.register("latency")(LatencyBenchmark)


def get_benchmark(config: Config) -> BaseBenchmark:
    cls = _registry.get(config.BENCHMARK_TYPE)
    return cls.from_config(config)
