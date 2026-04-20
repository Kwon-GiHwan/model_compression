from model_compression.reporter.console_reporter import ConsoleReporter
from model_compression.registry import Registry
from config import Config

_registry = Registry("REPORTER_TYPE")
_registry.register("console")(ConsoleReporter)


def get_reporter(config: Config) -> ConsoleReporter:
    cls = _registry.get(config.benchmark.reporter_type)
    return cls.from_config(config)
