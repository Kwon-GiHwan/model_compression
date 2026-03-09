import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    MODE = os.getenv("MODE", "full")
    METHOD = os.getenv("METHOD", "pruning.magnitude")

    MODEL_TYPE = os.getenv("MODEL_TYPE", "huggingface")
    MODEL_PATH = os.getenv("MODEL_PATH", "")
    TASK = os.getenv("TASK", "classification")
    OUTPUT_MODEL_PATH = os.getenv("OUTPUT_MODEL_PATH", "compressed_model.pt")

    TEACHER_LOADER = os.getenv("TEACHER_LOADER", "huggingface")
    TEACHER_MODEL_PATH = os.getenv("TEACHER_MODEL_PATH", "")
    TEACHER_HF_REPO = os.getenv("TEACHER_HF_REPO", "")
    TEACHER_HF_FILENAME = os.getenv("TEACHER_HF_FILENAME", "") or None

    DATASET_TYPE = os.getenv("DATASET_TYPE", "hf_datasets")
    DATASET_NAME = os.getenv("DATASET_NAME", "")
    DATASET_CONFIG = os.getenv("DATASET_CONFIG", "") or None
    DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")
    DATASET_PATH = os.getenv("DATASET_PATH", "")
    DATASET_BATCH_SIZE = int(os.getenv("DATASET_BATCH_SIZE", "16"))
    DATASET_MAX_LENGTH = int(os.getenv("DATASET_MAX_LENGTH", "128"))

    PRUNING_RATIO = float(os.getenv("PRUNING_RATIO", "0.3"))
    PRUNING_DEVICE = os.getenv("PRUNING_DEVICE", "cpu")

    DISTILL_TEMPERATURE = float(os.getenv("DISTILL_TEMPERATURE", "4.0"))
    DISTILL_ALPHA = float(os.getenv("DISTILL_ALPHA", "0.7"))

    TRAIN_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "20"))
    TRAIN_DEVICE = os.getenv("TRAIN_DEVICE", "mps")
    TRAIN_LR = float(os.getenv("TRAIN_LR", "1e-4"))

    BENCHMARK_DEVICE = os.getenv("BENCHMARK_DEVICE", "mps")
    BENCHMARK_RUNS = int(os.getenv("BENCHMARK_RUNS", "100"))
    BENCHMARK_INPUT_SIZE = int(os.getenv("BENCHMARK_INPUT_SIZE", "224"))
