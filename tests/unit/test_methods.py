"""
Unit tests for methods module.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from model_compression.methods.distillation.response_based import ResponseBasedDistiller
from model_compression.methods.pruning.attention_head_pruner import AttentionHeadPruner
from model_compression.methods.pruning.magnitude_pruner import MagnitudePruner
from model_compression.methods.registry import get_method


class TestMagnitudePruner:
    """Test MagnitudePruner class."""

    def test_initialization(self):
        """Test MagnitudePruner initialization."""
        pruner = MagnitudePruner(pruning_ratio=0.3, input_size=224, is_nlp=False)

        assert pruner.pruning_ratio == 0.3
        assert pruner.input_size == 224
        assert pruner.is_nlp is False

    @patch("model_compression.methods.pruning.magnitude_pruner.tp")
    def test_apply_image_model(self, mock_tp, simple_cnn_model):
        """Test applying magnitude pruning to image model."""
        mock_pruner = Mock()
        mock_tp.pruner.MagnitudePruner.return_value = mock_pruner

        pruner = MagnitudePruner(pruning_ratio=0.3, input_size=224, is_nlp=False)
        result = pruner.apply(simple_cnn_model)

        mock_pruner.step.assert_called_once()
        assert isinstance(result, nn.Module)

    @patch("model_compression.methods.pruning.magnitude_pruner.tp")
    def test_apply_nlp_model(self, mock_tp, simple_mlp_model):
        """Test applying magnitude pruning to NLP model."""
        mock_pruner = Mock()
        mock_tp.pruner.MagnitudePruner.return_value = mock_pruner

        pruner = MagnitudePruner(pruning_ratio=0.3, input_size=224, is_nlp=True)
        result = pruner.apply(simple_mlp_model)

        mock_pruner.step.assert_called_once()
        assert isinstance(result, nn.Module)

    def test_validate_invalid_ratio(self, mock_config):
        """Test validation fails with invalid pruning ratio."""
        mock_config.PRUNING_RATIO = 1.5

        pruner = MagnitudePruner(pruning_ratio=1.5, input_size=224, is_nlp=False)

        with pytest.raises(ValueError, match="PRUNING_RATIO는 0~1 사이여야 합니다"):
            pruner.validate(mock_config)


class TestAttentionHeadPruner:
    """Test AttentionHeadPruner class."""

    def test_initialization(self):
        """Test AttentionHeadPruner initialization."""
        pruner = AttentionHeadPruner(pruning_ratio=0.3)
        assert pruner.pruning_ratio == 0.3

    def test_validate_invalid_model_type(self, mock_config):
        """Test validation fails with non-HuggingFace model."""
        mock_config.MODEL_TYPE = "pytorch"

        pruner = AttentionHeadPruner(pruning_ratio=0.3)

        with pytest.raises(ValueError, match="HuggingFace Transformer 모델만 지원"):
            pruner.validate(mock_config)


class TestResponseBasedDistiller:
    """Test ResponseBasedDistiller class."""

    def test_initialization(self):
        """Test ResponseBasedDistiller initialization."""
        distiller = ResponseBasedDistiller(
            epochs=10, device="cpu", temperature=4.0, alpha=0.7, lr=1e-4
        )

        assert distiller.epochs == 10
        assert distiller.device == "cpu"
        assert distiller.temperature == 4.0
        assert distiller.alpha == 0.7
        assert distiller.lr == 1e-4

    def test_apply_without_teacher_raises_error(self, simple_mlp_model):
        """Test that applying without teacher raises error."""
        distiller = ResponseBasedDistiller(
            epochs=1, device="cpu", temperature=4.0, alpha=0.7, lr=1e-4
        )

        with pytest.raises(ValueError, match="teacher 모델이 필요합니다"):
            distiller.apply(simple_mlp_model, teacher=None)

    def test_apply_without_dataloader_raises_error(self, simple_mlp_model):
        """Test that applying without dataloader raises error."""
        teacher = Mock()
        distiller = ResponseBasedDistiller(
            epochs=1, device="cpu", temperature=4.0, alpha=0.7, lr=1e-4
        )

        with pytest.raises(ValueError, match="dataloader가 필요합니다"):
            distiller.apply(simple_mlp_model, teacher=teacher, dataloader=None)

    def test_validate_without_teacher_loader(self, mock_config):
        """Test validation fails without teacher loader."""
        mock_config.TEACHER_LOADER = ""

        distiller = ResponseBasedDistiller(
            epochs=1, device="cpu", temperature=4.0, alpha=0.7, lr=1e-4
        )

        with pytest.raises(ValueError, match="TEACHER_LOADER 설정이 필요합니다"):
            distiller.validate(mock_config)


class TestMethodsRegistry:
    """Test methods registry."""

    @patch("model_compression.methods.registry.MagnitudePruner")
    def test_get_magnitude_pruner(self, mock_pruner, mock_config):
        """Test getting magnitude pruner from registry."""
        mock_config.METHOD = "pruning.magnitude"
        mock_config.PRUNING_RATIO = 0.3
        mock_config.BENCHMARK_INPUT_SIZE = 224
        mock_config.DATASET_TYPE = "hf_datasets"

        result = get_method(mock_config)

        mock_pruner.assert_called_once_with(
            pruning_ratio=0.3, input_size=224, is_nlp=True
        )

    @patch("model_compression.methods.registry.AttentionHeadPruner")
    def test_get_attention_head_pruner(self, mock_pruner, mock_config):
        """Test getting attention head pruner from registry."""
        mock_config.METHOD = "pruning.attention_head"
        mock_config.PRUNING_RATIO = 0.3

        result = get_method(mock_config)

        mock_pruner.assert_called_once_with(pruning_ratio=0.3)

    @patch("model_compression.methods.registry.ResponseBasedDistiller")
    def test_get_response_based_distiller(self, mock_distiller, mock_config):
        """Test getting response based distiller from registry."""
        mock_config.METHOD = "distillation.response_based"
        mock_config.TRAIN_EPOCHS = 10
        mock_config.TRAIN_DEVICE = "cpu"
        mock_config.DISTILL_TEMPERATURE = 4.0
        mock_config.DISTILL_ALPHA = 0.7
        mock_config.TRAIN_LR = 1e-4

        result = get_method(mock_config)

        mock_distiller.assert_called_once_with(
            epochs=10, device="cpu", temperature=4.0, alpha=0.7, lr=1e-4
        )

    def test_get_invalid_method(self, mock_config):
        """Test that invalid method raises error."""
        mock_config.METHOD = "invalid.method"

        with pytest.raises(ValueError, match="지원하지 않는 METHOD"):
            get_method(mock_config)
