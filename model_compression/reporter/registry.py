from model_compression.reporter.base_reporter import BaseReporter
from model_compression.reporter.console_reporter import ConsoleReporter
from model_compression.registry import Registry
from config import Config

_registry = Registry("REPORTER_TYPE")
_registry.register("console")(ConsoleReporter)


def get_reporter(config: Config) -> BaseReporter:
    cls = _registry.get(config.REPORTER_TYPE)
    return cls.from_config(config)
