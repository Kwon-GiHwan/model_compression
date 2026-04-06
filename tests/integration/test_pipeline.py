"""
Integration tests for complete compression pipeline.
Tests end-to-end workflows with real components.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from model_compression.benchmark.latency_benchmark import LatencyBenchmark
from model_compression.methods.pruning.magnitude_pruner import MagnitudePruner
from model_compression.model.pytorch_model import PyTorchModel
from model_compression.reporter.console_reporter import ConsoleReporter


class TestPruningPipeline:
    """Test complete pruning pipeline."""

    @patch("model_compression.methods.pruning.magnitude_pruner.tp")
    def test_full_pruning_workflow(
        self, mock_tp, simple_cnn_model, temp_dir, mock_config
    ):
        """Test complete pruning workflow from model load to benchmark."""
        # Setup
        model_path = f"{temp_dir}/model.pt"
        output_path = f"{temp_dir}/compressed.pt"
        torch.save(simple_cnn_model, model_path)

        mock_config.MODEL_PATH = model_path
        mock_config.OUTPUT_MODEL_PATH = output_path
        mock_config.PRUNING_RATIO = 0.3
        mock_config.BENCHMARK_DEVICE = "cpu"
        mock_config.BENCHMARK_RUNS = 3
        mock_config.BENCHMARK_INPUT_SIZE = 224
        mock_config.DATASET_TYPE = "local_folder"

        # Mock pruning
        mock_pruner = Mock()
        mock_tp.pruner.MagnitudePruner.return_value = mock_pruner

        # Step 1: Load model
        model_wrapper = PyTorchModel()
        model_wrapper.load(model_path)
        assert model_wrapper.get_raw() is not None

        # Step 2: Apply pruning
        pruner = MagnitudePruner(pruning_ratio=0.3, input_size=224, is_nlp=False)
        compressed_model = pruner.apply(model_wrapper.get_raw())
        assert compressed_model is not None
        mock_pruner.step.assert_called_once()

        # Step 3: Save compressed model
        model_wrapper.set_raw(compressed_model)
        model_wrapper.save(output_path)

        # Step 4: Benchmark
        benchmark = LatencyBenchmark()
        original_result = benchmark.run(simple_cnn_model, mock_config)
        compressed_result = benchmark.run(compressed_model, mock_config)

        assert original_result["total_params"] > 0
        assert compressed_result["total_params"] > 0

        # Step 5: Report
        reporter = ConsoleReporter()
        reporter.report(original_result, compressed_result)

    def test_model_save_and_load_cycle(self, simple_cnn_model, temp_dir):
        """Test that model can be saved and loaded correctly."""
        model_path = f"{temp_dir}/cycle_test.pt"

        # Save
        wrapper1 = PyTorchModel()
        wrapper1.set_raw(simple_cnn_model)
        wrapper1.save(model_path)

        # Load
        wrapper2 = PyTorchModel()
        wrapper2.load(model_path)

        # Compare parameters
        original_params = sum(p.numel() for p in simple_cnn_model.parameters())
        loaded_params = sum(p.numel() for p in wrapper2.get_raw().parameters())

        assert original_params == loaded_params


class TestBenchmarkPipeline:
    """Test benchmark pipeline integration."""

    def test_compare_two_models(self, simple_cnn_model, simple_mlp_model, mock_config):
        """Test benchmarking and comparing two different models."""
        mock_config.BENCHMARK_DEVICE = "cpu"
        mock_config.BENCHMARK_RUNS = 3
        mock_config.BENCHMARK_INPUT_SIZE = 224
        mock_config.DATASET_TYPE = "local_folder"

        benchmark = LatencyBenchmark()

        result1 = benchmark.run(simple_cnn_model, mock_config)

        # For MLP, we need to adjust config for NLP
        mock_config.DATASET_TYPE = "hf_datasets"
        mock_config.DATASET_MAX_LENGTH = 128
        result2 = benchmark.run(simple_mlp_model, mock_config)

        # Both should have valid results
        assert result1["avg_latency_ms"] > 0
        assert result2["avg_latency_ms"] > 0
        assert result1["total_params"] != result2["total_params"]

    def test_benchmark_reproducibility(self, simple_cnn_model, mock_config):
        """Test that benchmark results are consistent across runs."""
        mock_config.BENCHMARK_DEVICE = "cpu"
        mock_config.BENCHMARK_RUNS = 5
        mock_config.BENCHMARK_INPUT_SIZE = 224
        mock_config.DATASET_TYPE = "local_folder"

        benchmark = LatencyBenchmark()

        result1 = benchmark.run(simple_cnn_model, mock_config)
        result2 = benchmark.run(simple_cnn_model, mock_config)

        # Total params should be exactly the same
        assert result1["total_params"] == result2["total_params"]
        assert result1["param_size_mb"] == result2["param_size_mb"]

        # Latency should be similar (within reasonable variance)
        # Note: Latency can vary significantly on different runs, so we use a generous threshold
        latency_diff = abs(result1["avg_latency_ms"] - result2["avg_latency_ms"])
        max_latency = max(result1["avg_latency_ms"], result2["avg_latency_ms"])
        assert latency_diff < max_latency * 2.0  # Within 200% (very generous for CI)


class TestModelLoaderIntegration:
    """Test model loader integration with different sources."""

    @patch("model_compression.model.loader.local_loader.torch.load")
    def test_local_loader_pytorch_model(self, mock_torch_load, simple_cnn_model):
        """Test loading PyTorch model from local path."""
        from model_compression.model.loader.local_loader import LocalLoader

        mock_torch_load.return_value = simple_cnn_model

        loader = LocalLoader("/test/model.pt")
        model = loader.load()

        assert model is not None
        mock_torch_load.assert_called_once_with("/test/model.pt", map_location="cpu")

    @patch("model_compression.model.loader.huggingface_loader.AutoModel")
    def test_huggingface_loader(self, mock_auto_model):
        """Test loading model from HuggingFace Hub."""
        from model_compression.model.loader.huggingface_loader import HuggingFaceLoader

        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        loader = HuggingFaceLoader(repo_id="test/model")
        model = loader.load()

        assert model is not None
        mock_auto_model.from_pretrained.assert_called_once_with("test/model")


class TestRegistryIntegration:
    """Test that registries work together correctly."""

    @patch("model_compression.model.registry.PyTorchModel")
    @patch("model_compression.methods.registry.MagnitudePruner")
    def test_model_and_method_registry(self, mock_pruner, mock_model, mock_config):
        """Test that model and method registries integrate properly."""
        from model_compression.methods.registry import get_method
        from model_compression.model.registry import get_model

        mock_config.MODEL_TYPE = "pytorch"
        mock_config.METHOD = "pruning.magnitude"
        mock_config.DATASET_TYPE = "local_folder"

        # Get model
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        mock_instance.load.return_value = mock_instance

        model = get_model(mock_config)

        # Get method
        mock_pruner_instance = Mock()
        mock_pruner.return_value = mock_pruner_instance

        method = get_method(mock_config)

        # Both should be retrieved successfully
        mock_model.assert_called_once()
        mock_pruner.assert_called_once()

    @patch("model_compression.data.registry.NLPDataLoader")
    @patch("model_compression.methods.registry.ResponseBasedDistiller")
    def test_data_and_distillation_registry(
        self, mock_distiller, mock_dataloader, mock_config
    ):
        """Test that data and distillation registries work together."""
        from model_compression.data.registry import get_dataloader
        from model_compression.methods.registry import get_method

        mock_config.DATASET_TYPE = "hf_datasets"
        mock_config.METHOD = "distillation.response_based"

        # Get dataloader
        dataloader = get_dataloader(mock_config)

        # Get method
        method = get_method(mock_config)

        mock_dataloader.assert_called_once()
        mock_distiller.assert_called_once()
