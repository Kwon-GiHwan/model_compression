"""
Unit tests for model module.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from model_compression.model.huggingface_model import HuggingFaceModel
from model_compression.model.pytorch_model import PyTorchModel
from model_compression.model.registry import get_model


class TestPyTorchModel:
    """Test PyTorchModel class."""

    def test_load_model(self, simple_cnn_model, temp_dir):
        """Test loading a PyTorch model."""
        model_path = f"{temp_dir}/test_model.pt"
        torch.save(simple_cnn_model, model_path)

        pytorch_model = PyTorchModel()
        loaded = pytorch_model.load(model_path)

        assert loaded is not None
        assert isinstance(loaded.get_raw(), nn.Module)
        assert loaded.get_preprocessor() is None

    def test_save_model(self, simple_cnn_model, temp_dir):
        """Test saving a PyTorch model."""
        model_path = f"{temp_dir}/saved_model.pt"

        pytorch_model = PyTorchModel()
        pytorch_model.set_raw(simple_cnn_model)
        pytorch_model.save(model_path)

        assert (
            torch.load(model_path, map_location="cpu", weights_only=False) is not None
        )

    def test_load_state_dict_only_raises_error(self, temp_dir):
        """Test that loading state_dict only raises error."""
        model_path = f"{temp_dir}/state_dict.pt"
        torch.save({"state": "dict"}, model_path)

        pytorch_model = PyTorchModel()
        with pytest.raises(ValueError, match="state_dict만 저장된 파일입니다"):
            pytorch_model.load(model_path)

    def test_get_raw_returns_model(self, simple_cnn_model):
        """Test that get_raw returns the underlying model."""
        pytorch_model = PyTorchModel()
        pytorch_model.set_raw(simple_cnn_model)

        raw = pytorch_model.get_raw()
        assert raw is simple_cnn_model


class TestHuggingFaceModel:
    """Test HuggingFaceModel class."""

    @patch("model_compression.model.huggingface_model.AutoModel")
    @patch("model_compression.model.huggingface_model.AutoTokenizer")
    def test_load_model_with_tokenizer(self, mock_tokenizer, mock_auto_model):
        """Test loading HuggingFace model with tokenizer."""
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_tok = Mock()
        mock_tokenizer.from_pretrained.return_value = mock_tok

        hf_model = HuggingFaceModel(task="classification")
        loaded = hf_model.load("test/model")

        assert loaded is not None
        mock_auto_model.from_pretrained.assert_called_once_with("test/model")
        mock_tokenizer.from_pretrained.assert_called_once_with("test/model")
        assert hf_model.get_preprocessor() is mock_tok

    @patch("model_compression.model.huggingface_model.AutoModel")
    @patch("model_compression.model.huggingface_model.AutoTokenizer")
    def test_load_model_without_tokenizer(self, mock_tokenizer, mock_auto_model):
        """Test loading HuggingFace model without tokenizer (vision model)."""
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_tokenizer.from_pretrained.side_effect = Exception("No tokenizer")

        with patch(
            "model_compression.model.huggingface_model.AutoProcessor"
        ) as mock_extractor:
            mock_extractor.from_pretrained.side_effect = Exception("No extractor")

            hf_model = HuggingFaceModel(task="classification")
            loaded = hf_model.load("test/model")

            assert loaded is not None
            assert hf_model.get_preprocessor() is None

    @patch("model_compression.model.huggingface_model.AutoModel")
    def test_save_model(self, mock_auto_model, temp_dir):
        """Test saving HuggingFace model."""
        mock_model = Mock()
        mock_tokenizer = Mock()

        hf_model = HuggingFaceModel()
        hf_model.set_raw(mock_model)
        hf_model._tokenizer = mock_tokenizer

        save_path = f"{temp_dir}/saved_model"
        hf_model.save(save_path)

        mock_model.save_pretrained.assert_called_once_with(save_path)
        mock_tokenizer.save_pretrained.assert_called_once_with(save_path)


class TestModelRegistry:
    """Test model registry."""

    @patch("torch.load")
    def test_get_pytorch_model(self, mock_torch_load, mock_config, tmp_path):
        """Test getting PyTorch model from registry."""
        import torch.nn as nn
        mock_config.MODEL_TYPE = "pytorch"
        mock_config.MODEL_PATH = str(tmp_path / "test.pt")

        mock_nn_model = MagicMock(spec=nn.Module)
        mock_torch_load.return_value = mock_nn_model

        result = get_model(mock_config)

        assert isinstance(result, PyTorchModel)

    @patch("model_compression.model.huggingface_model.AutoTokenizer")
    @patch("model_compression.model.huggingface_model.AutoModel")
    def test_get_huggingface_model(self, mock_auto_model, mock_tokenizer, mock_config):
        """Test getting HuggingFace model from registry."""
        mock_config.MODEL_TYPE = "huggingface"
        mock_config.MODEL_PATH = "bert-base"
        mock_config.TASK = "classification"

        mock_auto_model.from_pretrained.return_value = MagicMock()
        mock_tokenizer.from_pretrained.return_value = MagicMock()

        result = get_model(mock_config)

        assert isinstance(result, HuggingFaceModel)

    def test_get_invalid_model_type(self, mock_config):
        """Test that invalid model type raises error."""
        mock_config.MODEL_TYPE = "invalid_type"

        with pytest.raises(ValueError, match="지원하지 않는 MODEL_TYPE"):
            get_model(mock_config)
