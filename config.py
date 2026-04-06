import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    MODE: str = field(default_factory=lambda: os.getenv("MODE", "full"))
    METHOD: str = field(default_factory=lambda: os.getenv("METHOD", "pruning.magnitude"))
    MODEL_TYPE: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "huggingface"))
    MODEL_PATH: str = field(default_factory=lambda: os.getenv("MODEL_PATH", ""))
    TASK: str = field(default_factory=lambda: os.getenv("TASK", "classification"))
    OUTPUT_MODEL_PATH: str = field(default_factory=lambda: os.getenv("OUTPUT_MODEL_PATH", "compressed_model.pt"))

    TEACHER_LOADER: str = field(default_factory=lambda: os.getenv("TEACHER_LOADER", "huggingface"))
    TEACHER_MODEL_PATH: str = field(default_factory=lambda: os.getenv("TEACHER_MODEL_PATH", ""))
    TEACHER_HF_REPO: str = field(default_factory=lambda: os.getenv("TEACHER_HF_REPO", ""))
    TEACHER_HF_FILENAME: str | None = field(default_factory=lambda: os.getenv("TEACHER_HF_FILENAME", "") or None)

    DATASET_TYPE: str = field(default_factory=lambda: os.getenv("DATASET_TYPE", "hf_datasets"))
    DATASET_NAME: str = field(default_factory=lambda: os.getenv("DATASET_NAME", ""))
    DATASET_CONFIG: str | None = field(default_factory=lambda: os.getenv("DATASET_CONFIG", "") or None)
    DATASET_SPLIT: str = field(default_factory=lambda: os.getenv("DATASET_SPLIT", "train"))
    DATASET_PATH: str = field(default_factory=lambda: os.getenv("DATASET_PATH", ""))
    DATASET_BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("DATASET_BATCH_SIZE", "16")))
    DATASET_MAX_LENGTH: int = field(default_factory=lambda: int(os.getenv("DATASET_MAX_LENGTH", "128")))

    PRUNING_RATIO: float = field(default_factory=lambda: float(os.getenv("PRUNING_RATIO", "0.3")))
    PRUNING_DEVICE: str = field(default_factory=lambda: os.getenv("PRUNING_DEVICE", "cpu"))

    QUANT_DTYPE: str = field(default_factory=lambda: os.getenv("QUANT_DTYPE", "qint8"))
    QUANT_BACKEND: str = field(default_factory=lambda: os.getenv("QUANT_BACKEND", "x86"))
    QUANT_CALIBRATION_BATCHES: int = field(default_factory=lambda: int(os.getenv("QUANT_CALIBRATION_BATCHES", "100")))

    DISTILL_TEMPERATURE: float = field(default_factory=lambda: float(os.getenv("DISTILL_TEMPERATURE", "4.0")))
    DISTILL_ALPHA: float = field(default_factory=lambda: float(os.getenv("DISTILL_ALPHA", "0.7")))

    TRAIN_EPOCHS: int = field(default_factory=lambda: int(os.getenv("TRAIN_EPOCHS", "20")))
    TRAIN_DEVICE: str = field(default_factory=lambda: os.getenv("TRAIN_DEVICE", "mps"))
    TRAIN_LR: float = field(default_factory=lambda: float(os.getenv("TRAIN_LR", "1e-4")))

    BENCHMARK_DEVICE: str = field(default_factory=lambda: os.getenv("BENCHMARK_DEVICE", "mps"))
    BENCHMARK_RUNS: int = field(default_factory=lambda: int(os.getenv("BENCHMARK_RUNS", "100")))
    BENCHMARK_INPUT_SIZE: int = field(default_factory=lambda: int(os.getenv("BENCHMARK_INPUT_SIZE", "224")))
