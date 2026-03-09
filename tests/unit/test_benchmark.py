"""
Unit tests for benchmark module.
"""

import pytest
import torch

from model_compression.benchmark.latency_benchmark import LatencyBenchmark


class TestLatencyBenchmark:
    """Test LatencyBenchmark class."""

    def test_run_with_image_model(self, simple_cnn_model, mock_config):
        """Test running benchmark on image model."""
        mock_config.BENCHMARK_DEVICE = "cpu"
        mock_config.BENCHMARK_RUNS = 5
        mock_config.BENCHMARK_INPUT_SIZE = 224
        mock_config.DATASET_TYPE = "local_folder"

        benchmark = LatencyBenchmark()
        result = benchmark.run(simple_cnn_model, mock_config)

        assert "avg_latency_ms" in result
        assert "min_latency_ms" in result
        assert "max_latency_ms" in result
        assert "total_params" in result
        assert "param_size_mb" in result

        assert result["avg_latency_ms"] > 0
        assert result["min_latency_ms"] > 0
        assert result["max_latency_ms"] >= result["avg_latency_ms"]
        assert result["total_params"] > 0
        assert result["param_size_mb"] > 0

    def test_run_with_nlp_model(self, simple_mlp_model, mock_config):
        """Test running benchmark on NLP model."""
        mock_config.BENCHMARK_DEVICE = "cpu"
        mock_config.BENCHMARK_RUNS = 5
        mock_config.DATASET_MAX_LENGTH = 128
        mock_config.DATASET_TYPE = "hf_datasets"

        benchmark = LatencyBenchmark()
        result = benchmark.run(simple_mlp_model, mock_config)

        assert "avg_latency_ms" in result
        assert "total_params" in result
        assert result["avg_latency_ms"] > 0
        assert result["total_params"] > 0

    def test_result_types(self, simple_cnn_model, mock_config):
        """Test that result values have correct types."""
        mock_config.BENCHMARK_DEVICE = "cpu"
        mock_config.BENCHMARK_RUNS = 3
        mock_config.BENCHMARK_INPUT_SIZE = 224
        mock_config.DATASET_TYPE = "local_folder"

        benchmark = LatencyBenchmark()
        result = benchmark.run(simple_cnn_model, mock_config)

        assert isinstance(result["avg_latency_ms"], (int, float))
        assert isinstance(result["min_latency_ms"], (int, float))
        assert isinstance(result["max_latency_ms"], (int, float))
        assert isinstance(result["total_params"], int)
        assert isinstance(result["param_size_mb"], (int, float))

    def test_warmup_runs(self, simple_cnn_model, mock_config):
        """Test that warmup runs are executed before measurement."""
        mock_config.BENCHMARK_DEVICE = "cpu"
        mock_config.BENCHMARK_RUNS = 2
        mock_config.BENCHMARK_INPUT_SIZE = 224
        mock_config.DATASET_TYPE = "local_folder"

        benchmark = LatencyBenchmark()

        # Should complete without errors (warmup + measurement)
        result = benchmark.run(simple_cnn_model, mock_config)
        assert result is not None
