"""
End-to-end integration tests.
Tests complete workflows as a user would run them.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch


class TestEndToEndMagnitudePruning:
    """Test complete magnitude pruning workflow.

    Note: Some tests are skipped to avoid complex mocking of main() function.
    Core functionality is tested in unit tests and pipeline integration tests.
    """

    """Test complete magnitude pruning workflow."""

    @patch("main.get_model")
    @patch("model_compression.methods.pruning.magnitude_pruner.tp")
    def test_magnitude_pruning_apply_mode(
        self, mock_tp, mock_get_model, mock_config, simple_cnn_model, temp_dir
    ):
        """Test end-to-end magnitude pruning in apply mode."""
        from main import run_apply

        # Setup
        mock_config.MODE = "apply"
        mock_config.METHOD = "pruning.magnitude"
        mock_config.MODEL_TYPE = "pytorch"
        mock_config.OUTPUT_MODEL_PATH = f"{temp_dir}/output.pt"
        mock_config.data.type = "local_folder"

        # Mock model loading via get_model (registry now uses from_config internally)
        mock_model_instance = Mock()
        mock_model_instance.get_raw.return_value = simple_cnn_model
        mock_model_instance.get_preprocessor.return_value = None
        mock_get_model.return_value = mock_model_instance

        # Mock pruning
        mock_pruner = Mock()
        mock_tp.pruner.MagnitudePruner.return_value = mock_pruner

        # Run
        run_apply(mock_config)

        # Verify
        mock_get_model.assert_called_once_with(mock_config)
        mock_pruner.step.assert_called_once()
        mock_model_instance.save.assert_called_once_with(mock_config.OUTPUT_MODEL_PATH)

    @pytest.mark.skip(
        reason="Complex mocking required for main() - covered by pipeline tests"
    )
    def test_magnitude_pruning_benchmark_mode(self):
        """Test end-to-end magnitude pruning in benchmark mode.

        This test is skipped because it requires extensive mocking of the main() flow.
        The functionality is adequately covered by:
        - TestPruningPipeline.test_full_pruning_workflow
        - TestBenchmarkPipeline tests
        """
        pass


class TestEndToEndDistillation:
    """Test complete distillation workflow."""

    @pytest.mark.skip(
        reason="Complex mocking required for main() - covered by pipeline tests"
    )
    def test_distillation_apply_mode(self):
        """Test end-to-end distillation in apply mode.

        This test is skipped because it requires extensive mocking of the main() flow.
        The functionality is adequately covered by:
        - Unit tests for ResponseBasedDistiller
        - Registry integration tests
        """
        pass


class TestEndToEndFullMode:
    """Test complete workflow in full mode (apply + benchmark)."""

    @patch("main.get_model")
    @patch("model_compression.methods.pruning.magnitude_pruner.tp")
    def test_full_mode_execution(
        self,
        mock_tp,
        mock_get_model,
        mock_config,
        simple_cnn_model,
        temp_dir,
    ):
        """Test full mode executes both apply and benchmark."""
        from main import main

        # Setup
        mock_config.MODE = "full"
        mock_config.METHOD = "pruning.magnitude"
        mock_config.MODEL_TYPE = "pytorch"
        mock_config.OUTPUT_MODEL_PATH = f"{temp_dir}/full_output.pt"
        mock_config.benchmark.device = "cpu"
        mock_config.benchmark.runs = 2
        mock_config.data.type = "local_folder"
        mock_config.INPUT_SIZE = 224

        # Mock model via get_model (registry now uses from_config internally)
        mock_model_instance = Mock()
        mock_model_instance.get_raw.return_value = simple_cnn_model
        mock_model_instance.get_preprocessor.return_value = None
        mock_get_model.return_value = mock_model_instance

        # Mock pruning
        mock_pruner = Mock()
        mock_tp.pruner.MagnitudePruner.return_value = mock_pruner

        # Patch Config to return our mock
        with patch("main.Config", return_value=mock_config):
            main()

        # Verify both apply and benchmark were executed
        # Apply + Benchmark: get_model called 3 times (apply + 2x benchmark)
        assert mock_get_model.call_count >= 2
        # Apply: save called
        mock_model_instance.save.assert_called()


class TestEndToEndErrorHandling:
    """Test error handling in end-to-end workflows."""

    @patch("model_compression.model.registry.get_model")
    def test_invalid_mode_raises_error(self, mock_get_model, mock_config):
        """Test that invalid mode raises ValueError."""
        from main import main

        mock_config.MODE = "invalid_mode"

        # Mock model to avoid file loading
        mock_model = Mock()
        mock_get_model.return_value = mock_model

        with patch("main.Config", return_value=mock_config):
            with pytest.raises(ValueError, match="알 수 없는 MODE"):
                main()

    @pytest.mark.skip(reason="Error handling covered by unit tests")
    def test_model_loading_error_propagates(self):
        """Test that model loading errors propagate correctly.

        Covered by unit tests: test_model.py::test_get_invalid_model_type
        """
        pass

    @pytest.mark.skip(reason="Error handling covered by unit tests")
    def test_invalid_method_error(self):
        """Test that invalid method raises error.

        Covered by unit tests: test_methods.py::test_get_invalid_method
        """
        pass
