"""
Unit tests for config module.
"""

import os
from unittest.mock import patch

import pytest

from config import Config


class TestConfig:
    """Test Config class."""

    def test_config_default_values(self):
        """Test that Config has correct default values."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear environment and reload
            config = Config()
            assert config.MODE == "full"
            assert config.BENCHMARK_RUNS == 100
            assert config.TRAIN_EPOCHS == 20

    def test_config_env_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(
            os.environ,
            {
                "MODE": "apply",
                "METHOD": "distillation.response_based",
                "PRUNING_RATIO": "0.5",
                "BENCHMARK_RUNS": "50",
            },
        ):
            # Reload module to pick up env vars
            import importlib

            import config as config_module

            importlib.reload(config_module)

            config = config_module.Config()
            assert config.MODE == "apply"
            assert config.METHOD == "distillation.response_based"
            assert config.PRUNING_RATIO == 0.5
            assert config.BENCHMARK_RUNS == 50

    def test_config_type_conversion(self):
        """Test that Config converts types correctly."""
        with patch.dict(
            os.environ,
            {"DATASET_BATCH_SIZE": "32", "PRUNING_RATIO": "0.4", "TRAIN_EPOCHS": "10"},
        ):
            import importlib

            import config as config_module

            importlib.reload(config_module)

            config = config_module.Config()
            assert isinstance(config.DATASET_BATCH_SIZE, int)
            assert isinstance(config.PRUNING_RATIO, float)
            assert isinstance(config.TRAIN_EPOCHS, int)
            assert config.DATASET_BATCH_SIZE == 32
            assert config.PRUNING_RATIO == 0.4

    def test_config_none_handling(self):
        """Test that Config handles None values correctly."""
        with patch.dict(os.environ, {"DATASET_CONFIG": "", "TEACHER_HF_FILENAME": ""}):
            import importlib

            import config as config_module

            importlib.reload(config_module)

            config = config_module.Config()
            # Empty strings should map to None for optional fields
            assert config.TEACHER_HF_FILENAME is None
