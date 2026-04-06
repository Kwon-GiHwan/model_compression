from copy import copy

from model_compression.benchmark.latency_benchmark import LatencyBenchmark
from model_compression.data.registry import get_dataloader
from model_compression.methods.registry import get_method
from model_compression.model.loader import get_teacher_loader
from model_compression.model.registry import get_model
from model_compression.reporter.console_reporter import ConsoleReporter
from config import Config


def run_apply(config: Config):
    print(f"[Main] 방법론 적용: METHOD={config.METHOD}")

    student_wrapper = get_model(config)

    method = get_method(config)
    method.validate(config)

    teacher = None
    if method.requires_teacher():
        teacher = get_teacher_loader(config).load()

    dataloader = None
    if method.requires_dataloader():
        dl_wrapper = get_dataloader(config, tokenizer=student_wrapper.get_tokenizer())
        dataloader = dl_wrapper.get_dataloader()

    compressed = method.apply(
        student=student_wrapper.get_raw(),
        teacher=teacher,
        dataloader=dataloader,
    )

    try:
        student_wrapper.set_raw(compressed)
    except NotImplementedError:
        student_wrapper._model = compressed  # temporary fallback
    student_wrapper.save(config.OUTPUT_MODEL_PATH)
    print(f"[Main] 압축 모델 저장: {config.OUTPUT_MODEL_PATH}")


def run_benchmark(config: Config):
    print("[Main] Benchmark 시작")

    original_wrapper = get_model(config)
    original = original_wrapper.get_raw()

    benchmark_config = copy(config)
    benchmark_config.MODEL_PATH = config.OUTPUT_MODEL_PATH
    compressed_wrapper = get_model(benchmark_config)
    compressed = compressed_wrapper.get_raw()

    bench = LatencyBenchmark()
    reporter = ConsoleReporter()

    original_result = bench.run(original, config)
    compressed_result = bench.run(compressed, config)

    reporter.report(original_result, compressed_result)


def main():
    config = Config()
    print(f"[Main] MODE={config.MODE} / METHOD={config.METHOD} / TASK={config.TASK}")

    if config.MODE == "apply":
        run_apply(config)
    elif config.MODE == "benchmark":
        run_benchmark(config)
    elif config.MODE == "full":
        run_apply(config)
        run_benchmark(config)
    else:
        raise ValueError(f"알 수 없는 MODE: {config.MODE}")


if __name__ == "__main__":
    main()
