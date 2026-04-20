"""
Unit tests for data module.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest
from torch.utils.data import DataLoader

from model_compression.data.image_dataloader import ImageDataLoader
from model_compression.data.nlp_dataloader import NLPDataLoader
from model_compression.data.registry import get_dataloader


class TestImageDataLoader:
    """Test ImageDataLoader class."""

    @patch("model_compression.data.image_dataloader.ImageFolder")
    def test_get_dataloader(self, mock_image_folder):
        """Test getting image dataloader."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_image_folder.return_value = mock_dataset

        loader = ImageDataLoader(
            dataset_path="/test/path",
            task="classification",
            input_size=224,
            batch_size=16,
        )

        dataloader = loader.get_dataloader()

        assert isinstance(dataloader, DataLoader)
        mock_image_folder.assert_called_once()

    def test_initialization_parameters(self):
        """Test ImageDataLoader initialization with custom parameters."""
        loader = ImageDataLoader(
            dataset_path="/custom/path",
            task="detection",
            input_size=512,
            batch_size=32,
            split="val",
        )

        assert loader.dataset_path == "/custom/path"
        assert loader.task == "detection"
        assert loader.input_size == 512
        assert loader.batch_size == 32
        assert loader.split == "val"


class TestNLPDataLoader:
    """Test NLPDataLoader class."""

    @patch("model_compression.data.nlp_dataloader.load_dataset")
    @patch("model_compression.data.nlp_dataloader.AutoTokenizer")
    def test_get_dataloader(self, mock_tokenizer, mock_load_dataset):
        """Test getting NLP dataloader."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.map.return_value = mock_dataset
        mock_dataset.set_format = Mock()
        mock_dataset.__len__ = Mock(return_value=100)
        mock_load_dataset.return_value = mock_dataset

        # Mock tokenizer
        mock_tok = Mock()
        mock_tok.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        mock_tokenizer.from_pretrained.return_value = mock_tok

        loader = NLPDataLoader(
            dataset_name="test_dataset",
            tokenizer_path="test/tokenizer",
            dataset_config="config",
            split="train",
            batch_size=16,
            max_length=128,
        )

        dataloader = loader.get_dataloader()

        assert isinstance(dataloader, DataLoader)
        mock_load_dataset.assert_called_once_with(
            "test_dataset", "config", split="train"
        )
        mock_tokenizer.from_pretrained.assert_called_once_with("test/tokenizer")

    def test_initialization_parameters(self):
        """Test NLPDataLoader initialization with custom parameters."""
        loader = NLPDataLoader(
            dataset_name="custom_dataset",
            tokenizer_path="custom/tokenizer",
            dataset_config="custom_config",
            split="validation",
            batch_size=8,
            max_length=256,
        )

        assert loader.dataset_name == "custom_dataset"
        assert loader.tokenizer_path == "custom/tokenizer"
        assert loader.dataset_config == "custom_config"
        assert loader.split == "validation"
        assert loader.batch_size == 8
        assert loader.max_length == 256


class TestDataRegistry:
    """Test data registry."""

    def test_get_nlp_dataloader(self, mock_config):
        """Test getting NLP dataloader from registry."""
        mock_config.data.type = "hf_datasets"
        mock_config.data.name = "test_dataset"
        mock_config.data.config = "config"
        mock_config.data.split = "train"
        mock_config.data.batch_size = 16
        mock_config.data.max_length = 128
        mock_config.MODEL_PATH = "test/model"

        result = get_dataloader(mock_config)

        assert isinstance(result, NLPDataLoader)
        assert result.dataset_name == "test_dataset"
        assert result.tokenizer_path == "test/model"
        assert result.dataset_config == "config"
        assert result.split == "train"
        assert result.batch_size == 16
        assert result.max_length == 128

    def test_get_image_dataloader(self, mock_config):
        """Test getting image dataloader from registry."""
        mock_config.data.type = "local_folder"
        mock_config.data.path = "/test/images"
        mock_config.TASK = "classification"
        mock_config.INPUT_SIZE = 224
        mock_config.data.batch_size = 32
        mock_config.data.split = "train"

        result = get_dataloader(mock_config)

        assert isinstance(result, ImageDataLoader)
        assert result.dataset_path == "/test/images"
        assert result.task == "classification"
        assert result.input_size == 224
        assert result.batch_size == 32
        assert result.split == "train"

    def test_get_invalid_dataset_type(self, mock_config):
        """Test that invalid dataset type raises error."""
        mock_config.data.type = "invalid_type"

        with pytest.raises(ValueError, match="지원하지 않는 DATASET_TYPE"):
            get_dataloader(mock_config)
