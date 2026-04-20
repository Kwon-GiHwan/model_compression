from model_compression.benchmark.latency_benchmark import LatencyBenchmark
from model_compression.registry import Registry
from config import Config

_registry = Registry("BENCHMARK_TYPE")
_registry.register("latency")(LatencyBenchmark)


def get_benchmark(config: Config) -> LatencyBenchmark:
    cls = _registry.get(config.benchmark.type)
    return cls.from_config(config)
