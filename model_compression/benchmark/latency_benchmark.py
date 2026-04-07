import time

import torch

from model_compression.benchmark.base_benchmark import BaseBenchmark


class LatencyBenchmark(BaseBenchmark):
    """
    추론 속도 측정. 이미지/NLP 모델 공통 사용.
    NLP는 dummy token input, 이미지는 dummy pixel input 사용.
    """

    @classmethod
    def from_config(cls, config) -> "LatencyBenchmark":
        return cls()

    def run(self, model, config) -> dict:
        device = config.BENCHMARK_DEVICE
        runs = config.BENCHMARK_RUNS
        is_nlp = config.DATASET_TYPE == "hf_datasets"

        model = model.to(device).eval()

        if is_nlp:
            dummy = {
                "input_ids": torch.zeros(
                    1, config.DATASET_MAX_LENGTH, dtype=torch.long
                ).to(device)
            }
        else:
            s = config.BENCHMARK_INPUT_SIZE
            dummy = torch.randn(1, 3, s, s).to(device)

        with torch.no_grad():
            for _ in range(10):
                model(**dummy) if isinstance(dummy, dict) else model(dummy)

        latencies = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                model(**dummy) if isinstance(dummy, dict) else model(dummy)
                latencies.append((time.perf_counter() - start) * 1000)

        total_params = sum(p.numel() for p in model.parameters())

        return {
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
            "min_latency_ms": round(min(latencies), 3),
            "max_latency_ms": round(max(latencies), 3),
            "total_params": total_params,
            "param_size_mb": round(
                sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2), 2
            ),
        }
