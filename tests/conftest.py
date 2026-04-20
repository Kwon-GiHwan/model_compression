"""
Pytest fixtures and configuration for all tests.
"""

import os
import tempfile
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from tests.conftest_models import SimpleCNN, SimpleMLP


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_config():
    """Mock Config object with default values."""
    config = MagicMock()
    config.MODE = "full"
    config.METHOD = "pruning.magnitude"
    config.MODEL_TYPE = "pytorch"
    config.MODEL_PATH = "test_model.pt"
    config.TASK = "classification"
    config.OUTPUT_MODEL_PATH = "test_output.pt"

    config.TEACHER_LOADER = "local"
    config.TEACHER_MODEL_PATH = "teacher.pt"
    config.TEACHER_HF_REPO = ""
    config.TEACHER_HF_FILENAME = None

    config.data.type = "hf_datasets"
    config.data.name = "test_dataset"
    config.data.config = None
    config.data.split = "train"
    config.data.path = ""
    config.data.batch_size = 16
    config.data.max_length = 128

    config.PRUNING_RATIO = 0.3
    config.PRUNING_DEVICE = "cpu"

    config.QUANT_DTYPE = "qint8"
    config.QUANT_BACKEND = "x86"
    config.QUANT_CALIBRATION_BATCHES = 100

    config.DISTILL_TEMPERATURE = 4.0
    config.DISTILL_ALPHA = 0.7

    config.train.epochs = 2
    config.train.device = "cpu"
    config.train.lr = 1e-4

    config.benchmark.device = "cpu"
    config.benchmark.runs = 10
    config.INPUT_SIZE = 224
    config.benchmark.type = "latency"
    config.benchmark.reporter_type = "console"

    return config


@pytest.fixture
def simple_cnn_model():
    """Create a simple CNN model for testing."""
    return SimpleCNN()


@pytest.fixture
def simple_mlp_model():
    """Create a simple MLP model for testing."""
    return SimpleMLP()


@pytest.fixture
def mock_preprocessor():
    """Mock tokenizer for NLP tests."""
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.randint(0, 1000, (1, 128)),
        "attention_mask": torch.ones(1, 128),
    }
    return tokenizer


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_nlp_batch():
    """Create a sample NLP batch."""
    return {
        "input_ids": torch.randint(0, 1000, (4, 128)),
        "attention_mask": torch.ones(4, 128),
        "label": torch.randint(0, 10, (4,)),
    }


@pytest.fixture
def env_file(temp_dir):
    """Create a temporary .env file."""
    env_path = os.path.join(temp_dir, ".env")
    with open(env_path, "w") as f:
        f.write("MODE=apply\n")
        f.write("METHOD=pruning.magnitude\n")
        f.write("MODEL_TYPE=pytorch\n")
        f.write("PRUNING_RATIO=0.3\n")
    return env_path
