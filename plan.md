# Model Compression Pipeline - Claude Code 실행 계획서 v3
# 이미지 / NLP 모델 모두 지원, 특정 프레임워크에 종속되지 않는 범용 구조

---

## 설계 원칙

1. **모든 레이어는 추상 인터페이스로 정의** — 구현체는 언제든 교체 가능
2. **프레임워크 비종속** — Ultralytics, HuggingFace, 순수 PyTorch 모두 수용
3. **환경변수만으로 전체 파이프라인 제어** — 코드 수정 없이 모델/방법론/태스크 전환
4. **새 모델/방법론 추가 = 파일 1개 + 분기 1줄**

---

## 프로젝트 구조

```
compression/
├── .env
├── main.py
├── config.py
│
├── model/                          # 모델 로딩 추상화
│   ├── base_model.py               # 추상 인터페이스
│   ├── registry.py                 # MODEL_TYPE → 구현체 매핑
│   ├── pytorch_model.py            # 순수 PyTorch (.pt, .pth)
│   ├── huggingface_model.py        # HuggingFace Transformers / timm
│   └── loader/
│       ├── base_loader.py
│       ├── local_loader.py         # 로컬 파일
│       └── huggingface_loader.py   # HF Hub
│
├── data/                           # 데이터 추상화
│   ├── base_dataloader.py          # 추상 인터페이스
│   ├── registry.py                 # TASK → DataLoader 매핑
│   ├── image_dataloader.py         # 이미지 분류 / 검출 (torchvision)
│   └── nlp_dataloader.py           # HuggingFace datasets
│
├── methods/                        # 압축 방법론
│   ├── base_method.py              # 추상 인터페이스
│   ├── registry.py                 # METHOD → 구현체 매핑
│   ├── pruning/
│   │   ├── magnitude_pruner.py     # Structured Magnitude (이미지/NLP 공통)
│   │   └── attention_head_pruner.py # Transformer Attention Head
│   └── distillation/
│       ├── response_based.py       # Soft label KD (공통)
│       └── feature_based.py        # 중간 레이어 KD (Transformer 특화)
│
├── trainer/                        # Fine-tuning / Distillation 학습 루프
│   ├── base_trainer.py             # 추상 인터페이스
│   ├── registry.py                 # TASK → Trainer 매핑
│   ├── classification_trainer.py   # 분류 태스크 (이미지/NLP 공통)
│   ├── detection_trainer.py        # 객체 검출
│   └── seq2seq_trainer.py          # 번역, 요약 등
│
├── benchmark/                      # 성능 측정
│   ├── base_benchmark.py
│   ├── latency_benchmark.py        # 추론 속도 (공통)
│   └── accuracy_benchmark.py       # 정확도 측정 (태스크별 분기)
│
└── reporter/
    ├── base_reporter.py
    └── console_reporter.py
```

---

## 환경변수 (.env)

```env
# ── 모드 ──────────────────────────────────────────────
# apply     : 방법론 적용 후 저장
# benchmark : 원본 vs 압축 모델 비교
# full      : apply + benchmark
MODE=full

# ── 모델 설정 ─────────────────────────────────────────
# pytorch / huggingface
MODEL_TYPE=huggingface

# 로컬 경로 또는 HF repo ID
MODEL_PATH=klue/bert-base

# 태스크: classification / detection / seq2seq
TASK=classification

# 저장 경로
OUTPUT_MODEL_PATH=compressed_model.pt

# ── Teacher 설정 ───────────────────────────────────────
# local / huggingface
TEACHER_LOADER=huggingface
TEACHER_MODEL_PATH=./teacher_model.pt       # TEACHER_LOADER=local 일 때
TEACHER_HF_REPO=klue/roberta-large         # TEACHER_LOADER=huggingface 일 때
TEACHER_HF_FILENAME=                        # 특정 파일 지정 (선택)

# ── 데이터셋 설정 ──────────────────────────────────────
# hf_datasets / torchvision / local_folder
DATASET_TYPE=hf_datasets

DATASET_NAME=klue                           # DATASET_TYPE=hf_datasets 일 때
DATASET_CONFIG=ynat
DATASET_SPLIT=train

DATASET_PATH=./data/images                  # DATASET_TYPE=local_folder 일 때
DATASET_BATCH_SIZE=16
DATASET_MAX_LENGTH=128                      # NLP 토크나이징 최대 길이

# ── 방법론 선택 ────────────────────────────────────────
# pruning.magnitude / pruning.attention_head
# distillation.response_based / distillation.feature_based
METHOD=pruning.magnitude

# ── Pruning 설정 ───────────────────────────────────────
PRUNING_RATIO=0.3
PRUNING_DEVICE=cpu

# ── Distillation 설정 ──────────────────────────────────
DISTILL_TEMPERATURE=4.0
DISTILL_ALPHA=0.7

# ── 학습 설정 (Pruning fine-tuning / Distillation 공통) ─
TRAIN_EPOCHS=20
TRAIN_DEVICE=mps
TRAIN_LR=1e-4

# ── Benchmark 설정 ─────────────────────────────────────
BENCHMARK_DEVICE=mps
BENCHMARK_RUNS=100
BENCHMARK_INPUT_SIZE=224                    # 이미지: HxW / NLP: 무시됨
```

---

## 전체 코드

### config.py
```python
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
    TEACHER_HF_FILENAME = os.getenv("TEACHER_HF_FILENAME", None)

    DATASET_TYPE = os.getenv("DATASET_TYPE", "hf_datasets")
    DATASET_NAME = os.getenv("DATASET_NAME", "")
    DATASET_CONFIG = os.getenv("DATASET_CONFIG", None)
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
```

---

### model/base_model.py
```python
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(ABC):
    """
    모든 모델 래퍼의 공통 인터페이스.
    프레임워크(ultralytics, transformers, 순수 PyTorch)에 무관하게 동일하게 동작.
    """

    @abstractmethod
    def load(self, path: str) -> "BaseModel": ...

    @abstractmethod
    def save(self, path: str): ...

    @abstractmethod
    def get_raw(self) -> nn.Module:
        """압축 방법론에 전달할 순수 PyTorch nn.Module 반환"""
        ...

    @abstractmethod
    def get_tokenizer(self):
        """NLP 모델은 토크나이저 반환, 이미지 모델은 None"""
        ...
```

---

### model/pytorch_model.py
```python
import torch
import torch.nn as nn
from model.base_model import BaseModel


class PyTorchModel(BaseModel):
    """
    순수 PyTorch 모델 래퍼.
    torchvision, timm, 커스텀 모델 등 nn.Module이면 모두 수용.
    """

    def __init__(self):
        self._model: nn.Module = None

    def load(self, path: str) -> "PyTorchModel":
        self._model = torch.load(path, map_location="cpu")
        if isinstance(self._model, dict):
            # state_dict만 저장된 경우 — 아키텍처는 별도 제공 필요
            raise ValueError(
                "state_dict만 저장된 파일입니다. "
                "아키텍처 정의 후 load_state_dict()를 직접 사용하세요."
            )
        self._model.eval()
        print(f"[PyTorchModel] 로드 완료: {path}")
        return self

    def save(self, path: str):
        torch.save(self._model, path)
        print(f"[PyTorchModel] 저장 완료: {path}")

    def get_raw(self) -> nn.Module:
        return self._model

    def get_tokenizer(self):
        return None
```

---

### model/huggingface_model.py
```python
import torch
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from model.base_model import BaseModel


class HuggingFaceModel(BaseModel):
    """
    HuggingFace Transformers 모델 래퍼.
    NLP (BERT, GPT, LLaMA 등) 및 Vision (ViT, DETR 등) 모두 지원.
    로컬 경로 또는 HF Hub repo ID 모두 수용.
    """

    def __init__(self, task: str = "classification"):
        self._model = None
        self._tokenizer = None
        self.task = task

    def load(self, path: str) -> "HuggingFaceModel":
        print(f"[HuggingFaceModel] 로드: {path}")
        self._model = AutoModel.from_pretrained(path)
        self._model.eval()

        # NLP 모델: 토크나이저 로드 시도
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(path)
        except Exception:
            # Vision 모델 등 토크나이저 없는 경우
            try:
                self._tokenizer = AutoFeatureExtractor.from_pretrained(path)
            except Exception:
                self._tokenizer = None

        return self

    def save(self, path: str):
        self._model.save_pretrained(path)
        if self._tokenizer:
            self._tokenizer.save_pretrained(path)
        print(f"[HuggingFaceModel] 저장 완료: {path}")

    def get_raw(self):
        return self._model

    def get_tokenizer(self):
        return self._tokenizer
```

---

### model/registry.py
```python
from config import Config
from model.base_model import BaseModel
from model.pytorch_model import PyTorchModel
from model.huggingface_model import HuggingFaceModel


def get_model(config: Config) -> BaseModel:
    """
    MODEL_TYPE 환경변수에 따라 모델 래퍼 반환.
    새 모델 타입 추가: 구현체 작성 후 여기에 분기 추가.
    """
    registry = {
        "pytorch": PyTorchModel,
        "huggingface": lambda: HuggingFaceModel(task=config.TASK),
    }

    if config.MODEL_TYPE not in registry:
        raise ValueError(f"지원하지 않는 MODEL_TYPE: {config.MODEL_TYPE}")

    model = registry[config.MODEL_TYPE]()
    return model.load(config.MODEL_PATH)
```

---

### model/loader/base_loader.py
```python
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> object:
        """Teacher 모델(nn.Module) 반환"""
        ...
```

---

### model/loader/local_loader.py
```python
import torch
from transformers import AutoModel
from model.loader.base_loader import BaseLoader


class LocalLoader(BaseLoader):
    def __init__(self, model_path: str):
        self.model_path = model_path

    def load(self):
        print(f"[LocalLoader] 로컬 로드: {self.model_path}")
        try:
            # HuggingFace 로컬 디렉토리 시도
            model = AutoModel.from_pretrained(self.model_path)
        except Exception:
            # 순수 PyTorch .pt 파일
            model = torch.load(self.model_path, map_location="cpu")
        model.eval()
        return model
```

---

### model/loader/huggingface_loader.py
```python
from huggingface_hub import hf_hub_download
from transformers import AutoModel
from model.loader.base_loader import BaseLoader


class HuggingFaceLoader(BaseLoader):
    def __init__(self, repo_id: str, filename: str = None):
        self.repo_id = repo_id
        self.filename = filename

    def load(self):
        print(f"[HuggingFaceLoader] HF Hub 로드: {self.repo_id}")

        if self.filename:
            path = hf_hub_download(repo_id=self.repo_id, filename=self.filename)
            import torch
            model = torch.load(path, map_location="cpu")
        else:
            model = AutoModel.from_pretrained(self.repo_id)

        model.eval()
        return model
```

---

### model/loader/__init__.py
```python
from config import Config
from model.loader.local_loader import LocalLoader
from model.loader.huggingface_loader import HuggingFaceLoader


def get_teacher_loader(config: Config):
    registry = {
        "local": lambda: LocalLoader(config.TEACHER_MODEL_PATH),
        "huggingface": lambda: HuggingFaceLoader(
            repo_id=config.TEACHER_HF_REPO,
            filename=config.TEACHER_HF_FILENAME,
        ),
    }

    if config.TEACHER_LOADER not in registry:
        raise ValueError(f"지원하지 않는 TEACHER_LOADER: {config.TEACHER_LOADER}")

    return registry[config.TEACHER_LOADER]()
```

---

### data/base_dataloader.py
```python
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader


class BaseDataLoader(ABC):
    """
    모든 데이터 로더의 공통 인터페이스.
    이미지/NLP/오디오 등 태스크에 무관하게 DataLoader를 반환.
    """

    @abstractmethod
    def get_dataloader(self) -> DataLoader:
        ...
```

---

### data/image_dataloader.py
```python
import torchvision.transforms as T
from torchvision.datasets import ImageFolder, CocoDetection
from torch.utils.data import DataLoader
from data.base_dataloader import BaseDataLoader


class ImageDataLoader(BaseDataLoader):
    """
    torchvision 기반 이미지 데이터 로더.
    로컬 ImageFolder 구조 또는 COCO 형식 지원.
    """

    def __init__(
        self,
        dataset_path: str,
        task: str = "classification",
        input_size: int = 224,
        batch_size: int = 16,
        split: str = "train",
    ):
        self.dataset_path = dataset_path
        self.task = task
        self.input_size = input_size
        self.batch_size = batch_size
        self.split = split

    def get_dataloader(self) -> DataLoader:
        transform = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        dataset = ImageFolder(root=self.dataset_path, transform=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
```

---

### data/nlp_dataloader.py
```python
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from data.base_dataloader import BaseDataLoader


class NLPDataLoader(BaseDataLoader):
    """
    HuggingFace datasets 기반 NLP 데이터 로더.
    dataset name과 config만 있으면 자동 로드.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer_path: str,
        dataset_config: str = None,
        split: str = "train",
        batch_size: int = 16,
        max_length: int = 128,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.tokenizer_path = tokenizer_path
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length

    def get_dataloader(self) -> DataLoader:
        dataset = load_dataset(self.dataset_name, self.dataset_config, split=self.split)
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        def tokenize(batch):
            # 텍스트 컬럼 자동 탐지
            text_col = next(
                (c for c in ["text", "sentence", "document", "question"] if c in batch),
                list(batch.keys())[0]
            )
            return tokenizer(
                batch[text_col],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        dataset = dataset.map(tokenize, batched=True)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
```

---

### data/registry.py
```python
from config import Config
from data.base_dataloader import BaseDataLoader
from data.image_dataloader import ImageDataLoader
from data.nlp_dataloader import NLPDataLoader


def get_dataloader(config: Config, tokenizer=None) -> BaseDataLoader:
    """
    DATASET_TYPE 환경변수에 따라 DataLoader 반환.
    새 데이터 소스 추가: 구현체 작성 후 여기에 분기 추가.
    """
    if config.DATASET_TYPE == "hf_datasets":
        return NLPDataLoader(
            dataset_name=config.DATASET_NAME,
            tokenizer_path=config.MODEL_PATH,
            dataset_config=config.DATASET_CONFIG,
            split=config.DATASET_SPLIT,
            batch_size=config.DATASET_BATCH_SIZE,
            max_length=config.DATASET_MAX_LENGTH,
        )
    elif config.DATASET_TYPE in ("torchvision", "local_folder"):
        return ImageDataLoader(
            dataset_path=config.DATASET_PATH,
            task=config.TASK,
            input_size=config.BENCHMARK_INPUT_SIZE,
            batch_size=config.DATASET_BATCH_SIZE,
            split=config.DATASET_SPLIT,
        )
    else:
        raise ValueError(f"지원하지 않는 DATASET_TYPE: {config.DATASET_TYPE}")
```

---

### methods/base_method.py
```python
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseMethod(ABC):
    """
    모든 압축 방법론의 공통 인터페이스.

    student : 압축 대상 nn.Module
    teacher : KD 계열에서 사용, Pruning은 None
    dataloader : 학습이 필요한 방법론에서 사용

    새 방법론 추가:
      1. methods/pruning/ 또는 methods/distillation/ 에 구현체 작성
      2. methods/registry.py 에 분기 추가
      3. .env 에서 METHOD 값 변경
    """

    @abstractmethod
    def apply(self, student: nn.Module, teacher: nn.Module = None, dataloader=None) -> nn.Module:
        ...

    def validate(self, config) -> None:
        pass
```

---

### methods/pruning/magnitude_pruner.py
```python
import copy
import torch
import torch_pruning as tp
from methods.base_method import BaseMethod


class MagnitudePruner(BaseMethod):
    """
    Structured Magnitude Pruning.
    이미지 모델 / Transformer 모두 적용 가능.
    dependency graph를 자동 추적하므로 아키텍처 무관.
    """

    def __init__(self, pruning_ratio: float, input_size: int = 224, is_nlp: bool = False):
        self.pruning_ratio = pruning_ratio
        self.input_size = input_size
        self.is_nlp = is_nlp

    def apply(self, student, teacher=None, dataloader=None):
        model = copy.deepcopy(student).to("cpu").eval()

        # 입력 예시: NLP는 토큰 ID, 이미지는 픽셀
        if self.is_nlp:
            example_input = {"input_ids": torch.zeros(1, 128, dtype=torch.long)}
        else:
            example_input = torch.randn(1, 3, self.input_size, self.input_size)

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs=example_input,
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=self.pruning_ratio,
            ignored_layers=[],  # 필요 시 외부에서 주입 가능
        )

        print(f"[MagnitudePruner] Pruning 시작 (ratio={self.pruning_ratio})")
        pruner.step()
        print("[MagnitudePruner] 완료")
        return model

    def validate(self, config):
        if not 0 < config.PRUNING_RATIO < 1:
            raise ValueError("PRUNING_RATIO는 0~1 사이여야 합니다")
```

---

### methods/pruning/attention_head_pruner.py
```python
import copy
import torch
from methods.base_method import BaseMethod


class AttentionHeadPruner(BaseMethod):
    """
    Transformer Attention Head Pruning.
    중요도가 낮은 attention head를 제거.
    NLP / Vision Transformer 모두 적용 가능.
    """

    def __init__(self, pruning_ratio: float):
        self.pruning_ratio = pruning_ratio

    def apply(self, student, teacher=None, dataloader=None):
        model = copy.deepcopy(student).to("cpu").eval()

        pruned_heads = {}
        for layer_idx, layer in enumerate(model.encoder.layer):
            attn = layer.attention.self
            num_heads = attn.num_attention_heads
            num_to_prune = max(1, int(num_heads * self.pruning_ratio))

            # 각 head의 중요도 = weight norm 합산
            head_importance = torch.zeros(num_heads)
            for i in range(num_heads):
                head_size = attn.attention_head_size
                start = i * head_size
                end = (i + 1) * head_size
                head_importance[i] = (
                    attn.query.weight[start:end].norm()
                    + attn.key.weight[start:end].norm()
                    + attn.value.weight[start:end].norm()
                )

            # 중요도 낮은 head 선택
            heads_to_prune = head_importance.argsort()[:num_to_prune].tolist()
            pruned_heads[layer_idx] = heads_to_prune

        model.prune_heads(pruned_heads)
        print(f"[AttentionHeadPruner] 완료: {pruned_heads}")
        return model

    def validate(self, config):
        if config.MODEL_TYPE != "huggingface":
            raise ValueError("AttentionHeadPruner는 HuggingFace Transformer 모델만 지원합니다")
```

---

### methods/distillation/response_based.py
```python
import torch
import torch.nn.functional as F
from methods.base_method import BaseMethod


class ResponseBasedDistiller(BaseMethod):
    """
    Response-based Knowledge Distillation.
    Teacher output logits → soft label로 student 학습.
    이미지 분류 / NLP 분류 공통 적용 가능.
    """

    def __init__(self, epochs: int, device: str, temperature: float = 4.0, alpha: float = 0.7, lr: float = 1e-4):
        self.epochs = epochs
        self.device = device
        self.temperature = temperature
        self.alpha = alpha
        self.lr = lr

    def apply(self, student, teacher, dataloader=None):
        if teacher is None:
            raise ValueError("[Distiller] teacher 모델이 필요합니다")
        if dataloader is None:
            raise ValueError("[Distiller] dataloader가 필요합니다")

        student = student.to(self.device).train()
        teacher = teacher.to(self.device).eval()
        optimizer = torch.optim.AdamW(student.parameters(), lr=self.lr)
        T = self.temperature

        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in dataloader:
                # 이미지: (imgs, labels) / NLP: dict with input_ids 등
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
                    with torch.no_grad():
                        teacher_logits = teacher(inputs).logits
                    student_logits = student(inputs).logits
                else:
                    inputs = {k: v.to(self.device) for k, v in batch.items() if k != "label"}
                    labels = batch["label"].to(self.device)
                    with torch.no_grad():
                        teacher_logits = teacher(**inputs).logits
                    student_logits = student(**inputs).logits

                kd_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=-1),
                    F.softmax(teacher_logits / T, dim=-1),
                    reduction="batchmean",
                ) * (T ** 2)

                ce_loss = F.cross_entropy(student_logits, labels)
                loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"  Epoch [{epoch+1}/{self.epochs}] Loss: {total_loss:.4f}")

        print("[Distiller] 완료")
        return student

    def validate(self, config):
        if not config.TEACHER_LOADER:
            raise ValueError("Distillation은 TEACHER_LOADER 설정이 필요합니다")
```

---

### methods/registry.py
```python
from config import Config
from methods.base_method import BaseMethod
from methods.pruning.magnitude_pruner import MagnitudePruner
from methods.pruning.attention_head_pruner import AttentionHeadPruner
from methods.distillation.response_based import ResponseBasedDistiller


def get_method(config: Config) -> BaseMethod:
    """
    METHOD 환경변수에 따라 방법론 구현체 반환.

    새 방법론 추가:
      1. methods/pruning/ 또는 methods/distillation/ 에 구현체 작성
      2. 아래 registry에 키-값 추가
      3. .env 에서 METHOD 변경
    """
    is_nlp = config.DATASET_TYPE == "hf_datasets"

    registry = {
        "pruning.magnitude": lambda: MagnitudePruner(
            pruning_ratio=config.PRUNING_RATIO,
            input_size=config.BENCHMARK_INPUT_SIZE,
            is_nlp=is_nlp,
        ),
        "pruning.attention_head": lambda: AttentionHeadPruner(
            pruning_ratio=config.PRUNING_RATIO,
        ),
        "distillation.response_based": lambda: ResponseBasedDistiller(
            epochs=config.TRAIN_EPOCHS,
            device=config.TRAIN_DEVICE,
            temperature=config.DISTILL_TEMPERATURE,
            alpha=config.DISTILL_ALPHA,
            lr=config.TRAIN_LR,
        ),
    }

    if config.METHOD not in registry:
        raise ValueError(
            f"지원하지 않는 METHOD: {config.METHOD}\n"
            f"사용 가능: {list(registry.keys())}"
        )

    return registry[config.METHOD]()
```

---

### benchmark/latency_benchmark.py
```python
import time
import torch
from benchmark.base_benchmark import BaseBenchmark


class LatencyBenchmark(BaseBenchmark):
    """
    추론 속도 측정. 이미지/NLP 모델 공통 사용.
    NLP는 dummy token input, 이미지는 dummy pixel input 사용.
    """

    def run(self, model, config) -> dict:
        device = config.BENCHMARK_DEVICE
        runs = config.BENCHMARK_RUNS
        is_nlp = config.DATASET_TYPE == "hf_datasets"

        model = model.to(device).eval()

        if is_nlp:
            dummy = {"input_ids": torch.zeros(1, config.DATASET_MAX_LENGTH, dtype=torch.long).to(device)}
        else:
            s = config.BENCHMARK_INPUT_SIZE
            dummy = torch.randn(1, 3, s, s).to(device)

        with torch.no_grad():
            for _ in range(10):
                model(**dummy) if isinstance(dummy, dict) else model(dummy)

        latencies = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                model(**dummy) if isinstance(dummy, dict) else model(dummy)
                latencies.append((time.perf_counter() - start) * 1000)

        total_params = sum(p.numel() for p in model.parameters())

        return {
            "avg_latency_ms": round(sum(latencies) / len(latencies), 3),
            "min_latency_ms": round(min(latencies), 3),
            "max_latency_ms": round(max(latencies), 3),
            "total_params": total_params,
            "param_size_mb": round(total_params * 4 / (1024 ** 2), 2),
        }
```

---

### main.py
```python
from config import Config
from model.registry import get_model
from model.loader import get_teacher_loader
from methods.registry import get_method
from data.registry import get_dataloader
from benchmark.latency_benchmark import LatencyBenchmark
from reporter.console_reporter import ConsoleReporter


def run_apply(config: Config):
    print(f"[Main] 방법론 적용: METHOD={config.METHOD}")

    student_wrapper = get_model(config)

    teacher = None
    if config.METHOD.startswith("distillation"):
        teacher = get_teacher_loader(config).load()

    dataloader = None
    if config.METHOD.startswith("distillation"):
        dl_wrapper = get_dataloader(config, tokenizer=student_wrapper.get_tokenizer())
        dataloader = dl_wrapper.get_dataloader()

    method = get_method(config)
    method.validate(config)

    compressed = method.apply(
        student=student_wrapper.get_raw(),
        teacher=teacher,
        dataloader=dataloader,
    )

    student_wrapper._model = compressed
    student_wrapper.save(config.OUTPUT_MODEL_PATH)
    print(f"[Main] 압축 모델 저장: {config.OUTPUT_MODEL_PATH}")


def run_benchmark(config: Config):
    print("[Main] Benchmark 시작")

    original_wrapper = get_model(config)
    original = original_wrapper.get_raw()

    # 압축 모델 로드 (저장 타입에 따라 분기)
    import torch
    from transformers import AutoModel
    try:
        compressed = AutoModel.from_pretrained(config.OUTPUT_MODEL_PATH)
    except Exception:
        compressed = torch.load(config.OUTPUT_MODEL_PATH, map_location="cpu")

    bench = LatencyBenchmark()
    reporter = ConsoleReporter()

    original_result = bench.run(original, config)
    compressed_result = bench.run(compressed, config)

    reporter.report(original_result, compressed_result)


def main():
    config = Config()
    print(f"[Main] MODE={config.MODE} / METHOD={config.METHOD} / TASK={config.TASK}")

    if config.MODE == "apply":
        run_apply(config)
    elif config.MODE == "benchmark":
        run_benchmark(config)
    elif config.MODE == "full":
        run_apply(config)
        run_benchmark(config)
    else:
        raise ValueError(f"알 수 없는 MODE: {config.MODE}")


if __name__ == "__main__":
    main()
```

---

## 사용 예시

### YOLO11 Pruning (이미지)
```env
MODEL_TYPE=pytorch
MODEL_PATH=yolo11n.pt
TASK=detection
DATASET_TYPE=local_folder
DATASET_PATH=./coco128
METHOD=pruning.magnitude
```

### KLUE BERT Distillation (NLP)
```env
MODEL_TYPE=huggingface
MODEL_PATH=klue/bert-base
TASK=classification
DATASET_TYPE=hf_datasets
DATASET_NAME=klue
DATASET_CONFIG=ynat
TEACHER_LOADER=huggingface
TEACHER_HF_REPO=klue/roberta-large
METHOD=distillation.response_based
```

### ViT Attention Head Pruning (이미지 Transformer)
```env
MODEL_TYPE=huggingface
MODEL_PATH=google/vit-base-patch16-224
TASK=classification
DATASET_TYPE=local_folder
DATASET_PATH=./imagenet_subset
METHOD=pruning.attention_head
```

---

## 새 방법론/모델 추가 방법

```
방법론 추가:
  1. methods/pruning/my_method.py → BaseMethod 상속, apply() 구현
  2. methods/registry.py → registry dict에 키-값 추가
  3. .env → METHOD=pruning.my_method

모델 타입 추가:
  1. model/my_model.py → BaseModel 상속, load/save/get_raw() 구현
  2. model/registry.py → registry dict에 추가

데이터 소스 추가:
  1. data/my_dataloader.py → BaseDataLoader 상속, get_dataloader() 구현
  2. data/registry.py → 분기 추가
```