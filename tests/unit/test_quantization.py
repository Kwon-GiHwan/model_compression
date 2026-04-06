"""
Unit tests for quantization methods.
"""

from unittest.mock import Mock, patch

import pytest
import torch

from model_compression.methods.quantization.dynamic_quantizer import DynamicQuantizer
from model_compression.methods.quantization.static_quantizer import StaticQuantizer
from model_compression.methods.quantization.qat_quantizer import QATQuantizer
from model_compression.methods.registry import get_method


class TestDynamicQuantizer:
    """Test DynamicQuantizer class."""

    def test_initialization(self):
        quantizer = DynamicQuantizer(dtype="qint8")
        assert quantizer.dtype == torch.qint8

    def test_initialization_float16(self):
        quantizer = DynamicQuantizer(dtype="float16")
        assert quantizer.dtype == torch.float16

    def test_requires_teacher_false(self):
        assert DynamicQuantizer.requires_teacher() is False

    def test_requires_dataloader_false(self):
        assert DynamicQuantizer.requires_dataloader() is False

    @patch("model_compression.methods.quantization.dynamic_quantizer.torch.ao.quantization.quantize_dynamic")
    def test_apply(self, mock_quantize, simple_cnn_model):
        mock_quantize.return_value = simple_cnn_model
        quantizer = DynamicQuantizer(dtype="qint8")
        result = quantizer.apply(simple_cnn_model)
        mock_quantize.assert_called_once()
        assert result is not None

    def test_validate_valid_dtype(self, mock_config):
        mock_config.QUANT_DTYPE = "qint8"
        DynamicQuantizer().validate(mock_config)

    def test_validate_invalid_dtype(self, mock_config):
        mock_config.QUANT_DTYPE = "invalid"
        with pytest.raises(ValueError, match="QUANT_DTYPE"):
            DynamicQuantizer().validate(mock_config)


class TestStaticQuantizer:
    """Test StaticQuantizer class."""

    def test_initialization(self):
        quantizer = StaticQuantizer(backend="x86", calibration_batches=50)
        assert quantizer.backend == "x86"
        assert quantizer.calibration_batches == 50

    def test_requires_teacher_false(self):
        assert StaticQuantizer.requires_teacher() is False

    def test_requires_dataloader_true(self):
        assert StaticQuantizer.requires_dataloader() is True

    def test_apply_without_dataloader_raises_error(self, simple_cnn_model):
        with pytest.raises(ValueError, match="calibration용 dataloader가 필요합니다"):
            StaticQuantizer().apply(simple_cnn_model, dataloader=None)

    def test_validate_valid_backend(self, mock_config):
        mock_config.QUANT_BACKEND = "fbgemm"
        mock_config.QUANT_CALIBRATION_BATCHES = 100
        StaticQuantizer().validate(mock_config)

    def test_validate_invalid_backend(self, mock_config):
        mock_config.QUANT_BACKEND = "invalid"
        mock_config.QUANT_CALIBRATION_BATCHES = 100
        with pytest.raises(ValueError, match="QUANT_BACKEND"):
            StaticQuantizer().validate(mock_config)

    def test_validate_invalid_calibration_batches(self, mock_config):
        mock_config.QUANT_BACKEND = "x86"
        mock_config.QUANT_CALIBRATION_BATCHES = 0
        with pytest.raises(ValueError, match="QUANT_CALIBRATION_BATCHES"):
            StaticQuantizer().validate(mock_config)


class TestQATQuantizer:
    """Test QATQuantizer class."""

    def test_initialization(self):
        quantizer = QATQuantizer(backend="x86", epochs=10, device="cpu", lr=1e-4)
        assert quantizer.backend == "x86"
        assert quantizer.epochs == 10
        assert quantizer.device == "cpu"
        assert quantizer.lr == 1e-4

    def test_requires_teacher_false(self):
        assert QATQuantizer.requires_teacher() is False

    def test_requires_dataloader_true(self):
        assert QATQuantizer.requires_dataloader() is True

    def test_apply_without_dataloader_raises_error(self, simple_cnn_model):
        with pytest.raises(ValueError, match="학습용 dataloader가 필요합니다"):
            QATQuantizer(epochs=1, device="cpu").apply(simple_cnn_model, dataloader=None)

    def test_validate_valid_config(self, mock_config):
        mock_config.TRAIN_EPOCHS = 10
        QATQuantizer(epochs=10, device="cpu").validate(mock_config)

    def test_validate_invalid_epochs(self, mock_config):
        mock_config.TRAIN_EPOCHS = 0
        with pytest.raises(ValueError, match="TRAIN_EPOCHS"):
            QATQuantizer(epochs=1, device="cpu").validate(mock_config)

    def test_extract_logits_with_logits_attr(self):
        output = Mock()
        output.logits = torch.randn(2, 10)
        assert torch.equal(QATQuantizer._extract_logits(output), output.logits)

    def test_extract_logits_with_tuple(self):
        logits = torch.randn(2, 10)
        assert torch.equal(QATQuantizer._extract_logits((logits, None)), logits)

    def test_extract_logits_with_tensor(self):
        logits = torch.randn(2, 10)
        assert torch.equal(QATQuantizer._extract_logits(logits), logits)


class TestQuantizationRegistry:
    """Test quantization methods in registry."""

    @patch("model_compression.methods.registry.DynamicQuantizer")
    def test_get_dynamic_quantizer(self, mock_quantizer, mock_config):
        mock_config.METHOD = "quantization.dynamic"
        mock_config.QUANT_DTYPE = "qint8"
        get_method(mock_config)
        mock_quantizer.assert_called_once_with(dtype="qint8")

    @patch("model_compression.methods.registry.StaticQuantizer")
    def test_get_static_quantizer(self, mock_quantizer, mock_config):
        mock_config.METHOD = "quantization.static"
        mock_config.QUANT_BACKEND = "x86"
        mock_config.QUANT_CALIBRATION_BATCHES = 50
        get_method(mock_config)
        mock_quantizer.assert_called_once_with(backend="x86", calibration_batches=50)

    @patch("model_compression.methods.registry.QATQuantizer")
    def test_get_qat_quantizer(self, mock_quantizer, mock_config):
        mock_config.METHOD = "quantization.qat"
        mock_config.QUANT_BACKEND = "x86"
        mock_config.TRAIN_EPOCHS = 10
        mock_config.TRAIN_DEVICE = "cpu"
        mock_config.TRAIN_LR = 1e-4
        get_method(mock_config)
        mock_quantizer.assert_called_once_with(
            backend="x86", epochs=10, device="cpu", lr=1e-4
        )
