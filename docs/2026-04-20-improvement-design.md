# Model Compression Framework — 개선 설계서

> 범용성 7→9, OOP 효율성 7→9 목표  
> 기반 문서: `docs/2026-04-20-architecture-review.md`  
> 작성일: 2026-04-20

---

## 목차

1. [범용성 개선 설계](#1-범용성-개선-설계)
2. [OOP 효율성 개선 설계](#2-oop-효율성-개선-설계)
3. [변경 영향도 분석](#3-변경-영향도-분석)
4. [구현 우선순위와 순서](#4-구현-우선순위와-순서)

---

## 1. 범용성 개선 설계

### 1.1 Config 분리: flat → structured

**문제**: `config.py`의 32개 필드가 하나의 flat dataclass에 혼재. 방법론 추가 시마다 필드가 무한 증가하고, 어떤 필드가 어떤 방법론용인지 불명확하다.

**분리 기준** (coding-style 준수: "구현체 1개뿐인 interface 금지"):

| 그룹 | 필드 수 | 참조처 | 분리 여부 | 근거 |
|------|:---:|---|:---:|---|
| Data | 7 | NLPDataLoader, ImageDataLoader, MagnitudePruner, LatencyBenchmark | ✅ 분리 | 4곳에서 참조 |
| Train | 3 | QATQuantizer, ResponseBasedDistiller | ✅ 분리 | 2곳에서 동일 조합 참조 |
| Benchmark | 5 | LatencyBenchmark, main.run_benchmark() | ✅ 분리 | 5필드, 2곳 참조 |
| Pruning | 2 | MagnitudePruner, AttentionHeadPruner | ❌ 유지 | 2필드뿐, 별도 dataclass는 과잉 |
| Quant | 3 | Dynamic/Static/QAT — 각각 다른 조합 | ❌ 유지 | 공통 구조가 아님 |
| Distill | 2 | ResponseBasedDistiller만 | ❌ 유지 | 참조처 1개, 분리하면 원칙 위반 |

**Before** (`config.py`):
```python
@dataclass
class Config:
    # 30+ 필드가 같은 레벨에 혼재
    PRUNING_RATIO: float = field(default_factory=lambda: float(os.getenv("PRUNING_RATIO", "0.3")))
    QUANT_DTYPE: str = field(default_factory=lambda: os.getenv("QUANT_DTYPE", "qint8"))
    TRAIN_EPOCHS: int = field(default_factory=lambda: int(os.getenv("TRAIN_EPOCHS", "20")))
    TRAIN_DEVICE: str = field(default_factory=lambda: os.getenv("TRAIN_DEVICE", "mps"))
    DATASET_TYPE: str = field(default_factory=lambda: os.getenv("DATASET_TYPE", "hf_datasets"))
    BENCHMARK_DEVICE: str = field(default_factory=lambda: os.getenv("BENCHMARK_DEVICE", "mps"))
    # ...
```

**After**:
```python
@dataclass
class DataConfig:
    """데이터 관련 설정. NLPDataLoader, ImageDataLoader 등 4곳에서 참조."""
    type: str = field(default_factory=lambda: os.getenv("DATASET_TYPE", "hf_datasets"))
    name: str = field(default_factory=lambda: os.getenv("DATASET_NAME", ""))
    config: str | None = field(default_factory=lambda: os.getenv("DATASET_CONFIG", "") or None)
    split: str = field(default_factory=lambda: os.getenv("DATASET_SPLIT", "train"))
    path: str = field(default_factory=lambda: os.getenv("DATASET_PATH", ""))
    batch_size: int = field(default_factory=lambda: int(os.getenv("DATASET_BATCH_SIZE", "16")))
    max_length: int = field(default_factory=lambda: int(os.getenv("DATASET_MAX_LENGTH", "128")))


@dataclass
class TrainConfig:
    """학습 관련 설정. QATQuantizer, ResponseBasedDistiller에서 참조."""
    epochs: int = field(default_factory=lambda: int(os.getenv("TRAIN_EPOCHS", "20")))
    device: str = field(default_factory=lambda: os.getenv("TRAIN_DEVICE", "mps"))
    lr: float = field(default_factory=lambda: float(os.getenv("TRAIN_LR", "1e-4")))


@dataclass
class BenchmarkConfig:
    """벤치마크 관련 설정. LatencyBenchmark, main.run_benchmark()에서 참조."""
    device: str = field(default_factory=lambda: os.getenv("BENCHMARK_DEVICE", "mps"))
    runs: int = field(default_factory=lambda: int(os.getenv("BENCHMARK_RUNS", "100")))
    input_size: int = field(default_factory=lambda: int(os.getenv("BENCHMARK_INPUT_SIZE", "224")))
    type: str = field(default_factory=lambda: os.getenv("BENCHMARK_TYPE", "latency"))
    reporter_type: str = field(default_factory=lambda: os.getenv("REPORTER_TYPE", "console"))


@dataclass
class Config:
    # 공통 (main.py에서 직접 참조)
    MODE: str = field(default_factory=lambda: os.getenv("MODE", "full"))
    METHOD: str = field(default_factory=lambda: os.getenv("METHOD", "pruning.magnitude"))
    MODEL_TYPE: str = field(default_factory=lambda: os.getenv("MODEL_TYPE", "huggingface"))
    MODEL_PATH: str = field(default_factory=lambda: os.getenv("MODEL_PATH", ""))
    TASK: str = field(default_factory=lambda: os.getenv("TASK", "classification"))
    OUTPUT_MODEL_PATH: str = field(default_factory=lambda: os.getenv("OUTPUT_MODEL_PATH", "compressed_model.pt"))

    # Teacher (main.py에서도 참조하므로 Config에 유지)
    TEACHER_LOADER: str = field(default_factory=lambda: os.getenv("TEACHER_LOADER", "huggingface"))
    TEACHER_MODEL_PATH: str = field(default_factory=lambda: os.getenv("TEACHER_MODEL_PATH", ""))
    TEACHER_HF_REPO: str = field(default_factory=lambda: os.getenv("TEACHER_HF_REPO", ""))
    TEACHER_HF_FILENAME: str | None = field(default_factory=lambda: os.getenv("TEACHER_HF_FILENAME", "") or None)

    # 방법론별 (2~3필드씩, 별도 dataclass는 과잉)
    PRUNING_RATIO: float = field(default_factory=lambda: float(os.getenv("PRUNING_RATIO", "0.3")))
    PRUNING_DEVICE: str = field(default_factory=lambda: os.getenv("PRUNING_DEVICE", "cpu"))
    QUANT_DTYPE: str = field(default_factory=lambda: os.getenv("QUANT_DTYPE", "qint8"))
    QUANT_BACKEND: str = field(default_factory=lambda: os.getenv("QUANT_BACKEND", "x86"))
    QUANT_CALIBRATION_BATCHES: int = field(default_factory=lambda: int(os.getenv("QUANT_CALIBRATION_BATCHES", "100")))
    DISTILL_TEMPERATURE: float = field(default_factory=lambda: float(os.getenv("DISTILL_TEMPERATURE", "4.0")))
    DISTILL_ALPHA: float = field(default_factory=lambda: float(os.getenv("DISTILL_ALPHA", "0.7")))

    # 도메인별 그룹 (2곳 이상 참조)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
```

**구현체 참조 변경 예시** — `qat_quantizer.py`:
```python
# Before
return cls(backend=config.QUANT_BACKEND, epochs=config.TRAIN_EPOCHS,
           device=config.TRAIN_DEVICE, lr=config.TRAIN_LR)

# After
return cls(backend=config.QUANT_BACKEND, epochs=config.train.epochs,
           device=config.train.device, lr=config.train.lr)
```

---

### 1.2 학습 루프 공통화: `unpack_batch()` + `forward_model()` 추출

**문제**: 3곳에서 동일한 batch 분기 패턴이 반복된다.

| 파일 | 라인 | 패턴 |
|------|------|------|
| `qat_quantizer.py` | 46-54 | `isinstance(batch, (list, tuple))` 분기 + `extract_logits` |
| `response_based.py` | 52-64 | 동일 |
| `static_quantizer.py` | 35-39 | `isinstance(batch, (list, tuple))` 분기 (logits 없이) |

**설계 판단**: `train_loop()` 전체를 공통화하지 않는다. 이유:

| 선택지 | 사용처 수 | 판단 |
|--------|:---------:|------|
| `unpack_batch()` + `forward_model()` | 3곳 | ✅ 추출 |
| `train_loop()` 전체 | 실질 1곳 (QAT만 깔끔히 맞음, Distiller는 teacher forward 때문에 불일치) | ❌ 추출하지 않음 |

**After** — `methods/utils.py`에 추가:
```python
import torch
import torch.nn as nn


def unpack_batch(
    batch: dict | list | tuple, device: str, label_key: str = "label"
) -> tuple[dict | torch.Tensor, torch.Tensor]:
    """batch에서 inputs/labels를 분리하고 device로 이동.

    이미지: (tensor, labels) → tensor, labels
    NLP: {"input_ids": ..., "label": ...} → {k: v}, labels
    """
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device), batch[1].to(device)
    inputs = {k: v.to(device) for k, v in batch.items() if k != label_key}
    labels = batch[label_key].to(device)
    return inputs, labels


def forward_model(model: nn.Module, inputs: dict | torch.Tensor) -> torch.Tensor:
    """inputs 타입에 따라 model(inputs) 또는 model(**inputs) 호출 후 logits 추출."""
    if isinstance(inputs, dict):
        return extract_logits(model(**inputs))
    return extract_logits(model(inputs))
```

**After** — `qat_quantizer.py:43-62` 간소화:
```python
# Before (12줄)
for batch in dataloader:
    if isinstance(batch, (list, tuple)):
        inputs, labels = batch[0].to(self.device), batch[1].to(self.device)
        logits = extract_logits(prepared(inputs))
    else:
        inputs = {k: v.to(self.device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(self.device)
        logits = extract_logits(prepared(**inputs))
    loss = F.cross_entropy(logits, labels)
    ...

# After (4줄)
for batch in dataloader:
    inputs, labels = unpack_batch(batch, self.device)
    logits = forward_model(prepared, inputs)
    loss = F.cross_entropy(logits, labels)
    ...
```

`label_key` 파라미터화는 NLP 데이터셋마다 키가 다를 수 있어 (`"label"`, `"labels"`, `"target"`) 실용적 범용성 개선이다.

---

### 1.3 태스크 확장 가능성

**현재 상태**: classification 전용 (`F.cross_entropy` 하드코딩).

**설계 판단**: coding-style 원칙 — "지금 당장 두 번째 사용처가 없으면 추상화하지 않는다." generation/detection을 지금 구현하지 않으므로 구조적 변경 없음. `TODO` 주석으로 확장 지점만 표시:

```python
# qat_quantizer.py
# TODO: generation/detection 등 지원 시 loss 함수를 외부에서 주입하도록 변경
loss = F.cross_entropy(logits, labels)
```

---

## 2. OOP 효율성 개선 설계

### 2.1 `Any` 타입 → 구체적 타입 어노테이션

타입 안전성 점수 4/10 → 8/10 목표. 핵심 인터페이스 5곳의 `Any`를 제거한다.

#### `BaseMethod.apply()` — `base_method.py:22-24`

```python
# Before
@abstractmethod
def apply(self, student: Any, teacher: Any = None, dataloader: Any = None) -> Any: ...

# After
import torch.nn as nn
from typing import Iterable

@abstractmethod
def apply(
    self, student: nn.Module, teacher: nn.Module | None = None,
    dataloader: Iterable | None = None,
) -> nn.Module: ...
```

#### `BaseMethod.validate()` / `from_config()` — `base_method.py:26, 30`

```python
# Before
def validate(self, config: Any) -> None: ...
def from_config(cls, config: Any) -> "BaseMethod": ...

# After
from config import Config

def validate(self, config: Config) -> None: ...
def from_config(cls, config: Config) -> "BaseMethod": ...
```

#### `BaseModel` — `base_model.py:20-35`

```python
# Before
def get_raw(self) -> Any: ...
def set_raw(self, model: Any) -> None: ...
def get_preprocessor(self) -> Any: ...
def from_config(cls, config: Any) -> "BaseModel": ...

# After
def get_raw(self) -> nn.Module: ...
def set_raw(self, model: nn.Module) -> None: ...
def get_preprocessor(self) -> Any: ...  # 유지 — 반환 타입이 Tokenizer | Processor | None, 공통 부모 없음
def from_config(cls, config: Config) -> "BaseModel": ...
```

> `get_preprocessor()`는 `Any` 유지. `PreTrainedTokenizerBase | BaseImageProcessor | None` 유니온을 쓸 수 있지만, 호출처에서 타입별 분기를 하지 않으므로 `Any`가 더 정직하다.

#### `extract_logits()` — `methods/utils.py:6`

```python
# Before
def extract_logits(output: Any) -> Any:

# After
from transformers.modeling_outputs import ModelOutput

def extract_logits(output: ModelOutput | torch.Tensor | tuple | list) -> torch.Tensor:
```

#### `BaseDataLoader.from_config()` — `base_dataloader.py:17`

```python
# Before
def from_config(cls, config: Any) -> "BaseDataLoader":

# After
from config import Config
def from_config(cls, config: Config) -> "BaseDataLoader":
```

---

### 2.2 `from_config()` → `@abstractmethod`

**문제**: 모든 ABC의 `from_config()`이 `raise NotImplementedError`로 구현. 구현 누락이 런타임에서야 발견됨.

**대상**: `BaseMethod`, `BaseModel`, `BaseDataLoader` (3개 파일)

> `BaseBenchmark`/`BaseReporter`는 2.3에서 ABC 자체를 제거하므로 제외.

```python
# Before (base_method.py:29-32)
@classmethod
def from_config(cls, config: Any) -> "BaseMethod":
    """Config 기반으로 인스턴스 생성. 서브클래스에서 반드시 구현."""
    raise NotImplementedError(f"{cls.__name__}.from_config()이 구현되지 않았습니다")

# After
@classmethod
@abstractmethod
def from_config(cls, config: Config) -> "BaseMethod": ...
```

동일 패턴을 `base_model.py:34-36`, `base_dataloader.py:16-18`에 적용.

---

### 2.3 `BaseBenchmark` / `BaseReporter` ABC 제거

**근거**: 각각 구현체 1개뿐. coding-style 원칙: "구현체 1개뿐인 ABC/abstractmethod 금지 — 실제 mock/교체 시점에 추출."

#### `BaseBenchmark` 제거

```python
# Before — base_benchmark.py (전체 파일 삭제)
class BaseBenchmark(ABC):
    @abstractmethod
    def run(self, model, config) -> dict: ...
    @classmethod
    def from_config(cls, config) -> "BaseBenchmark": raise NotImplementedError

# Before — latency_benchmark.py
from model_compression.benchmark.base_benchmark import BaseBenchmark
class LatencyBenchmark(BaseBenchmark):

# After — latency_benchmark.py (base_benchmark.py 삭제)
class LatencyBenchmark:
    @classmethod
    def from_config(cls, config: Config) -> "LatencyBenchmark":
        return cls()
    def run(self, model: nn.Module, config: Config) -> dict:
        # 기존 구현 그대로
```

#### `BaseReporter` 제거

```python
# Before — base_reporter.py (전체 파일 삭제)
class BaseReporter(ABC):
    @abstractmethod
    def report(self, original_result: dict, compressed_result: dict): ...

# Before — console_reporter.py
from model_compression.reporter.base_reporter import BaseReporter
class ConsoleReporter(BaseReporter):

# After — console_reporter.py (base_reporter.py 삭제)
class ConsoleReporter:
    @classmethod
    def from_config(cls, config: Config) -> "ConsoleReporter":
        return cls()
    def report(self, original_result: dict, compressed_result: dict) -> None:
        # 기존 구현 그대로
```

#### Registry 반환 타입 수정

```python
# benchmark/registry.py — Before
def get_benchmark(config: Config) -> BaseBenchmark:

# After
def get_benchmark(config: Config) -> LatencyBenchmark:

# reporter/registry.py — Before
def get_reporter(config: Config) -> BaseReporter:

# After
def get_reporter(config: Config) -> ConsoleReporter:
```

> 두 번째 구현체가 추가되는 시점에 ABC를 다시 추출하면 된다.

---

### 2.4 `BaseModel` 주석 정리

실제와 불일치하는 주석을 수정:

```python
# Before (base_model.py:8-10)
"""
모든 모델 래퍼의 공통 인터페이스.
프레임워크(PyTorch, TensorFlow, JAX, ONNX 등)에 무관하게 동일하게 동작.
"""

# After
"""
모든 모델 래퍼의 공통 인터페이스.
현재 PyTorch 기반 모델(HuggingFace, 순수 PyTorch)을 지원.
"""
```

---

## 3. 변경 영향도 분석

### 3.1 파일별 영향 매트릭스

| 변경 항목 | 수정 파일 | 삭제 파일 | 변경 성격 |
|-----------|:---------:|:---------:|-----------|
| `Any` → 구체 타입 | ~12 | 0 | import 추가 + 시그니처. 기능 변경 없음 |
| `from_config()` abstractmethod | 3 | 0 | 데코레이터 1줄. 기능 변경 없음 |
| ABC 제거 | 4 | 2 | import 경로 변경 |
| `unpack_batch`/`forward_model` 추출 | 4 | 0 | 함수 추출 + 호출부 교체 |
| Config 분리 | ~15 | 0 | 필드 접근 경로 변경 (breaking) |
| 주석 정리 | 1 | 0 | 주석만 |

### 3.2 테스트 수정 필요 여부

| 변경 항목 | 테스트 수정 | 상세 |
|-----------|:-----------:|------|
| `Any` → 타입 | 불필요 | 런타임 동작 변경 없음 |
| `from_config` abstractmethod | 불필요 | 모든 구현체가 이미 구현 완료 |
| ABC 제거 | **필요** | `test_benchmark.py`, `test_reporter.py`의 `BaseBenchmark`/`BaseReporter` import 제거 |
| `unpack_batch` 추출 | **권장** | 새 유틸리티 함수 unit test 추가 |
| Config 분리 | **필요** | `test_config.py`, `conftest.py`의 mock config 필드 경로 변경 |

### 3.3 Breaking Change

| 변경 항목 | Breaking? | 이유 |
|-----------|:---------:|------|
| `Any` → 타입 | No | 시그니처만 변경 |
| `from_config` abstractmethod | No | 모든 구현체 이미 구현 완료 |
| ABC 제거 | **Yes** | `BaseBenchmark`/`BaseReporter` import하는 코드 깨짐 (개인 프로젝트이므로 영향 미미) |
| `unpack_batch` 추출 | No | 내부 리팩토링 |
| Config 분리 | **Yes** | `config.TRAIN_EPOCHS` → `config.train.epochs` |

---

## 4. 구현 우선순위와 순서

의존 관계와 리스크 기준 정렬. **각 Phase를 완료하고 테스트 통과를 확인한 뒤 다음 Phase로 진행.**

### Phase 1: 안전한 변경 (기능 변경 없음, breaking 없음)

| Step | 내용 | 대상 파일 | 노력 | 검증 |
|:----:|------|-----------|:----:|------|
| 1.1 | `Any` → 구체 타입 | ABC 3개 + 구현체 10개 + utils.py | 낮음 | `pytest` 전체 통과 |
| 1.2 | `from_config()` → `@abstractmethod` | ABC 3개 (각 2줄 변경) | 최소 | `pytest` 전체 통과 |
| 1.3 | `BaseModel` 주석 정리 | `base_model.py` 1줄 | 최소 | 없음 |

### Phase 2: 리팩토링 (기능 변경 없음, 내부 구조 변경)

| Step | 내용 | 대상 파일 | 노력 | 검증 |
|:----:|------|-----------|:----:|------|
| 2.1 | `unpack_batch()` + `forward_model()` 추출 | `utils.py` + 3개 구현체 | 낮음 | 기존 테스트 통과 + 새 unit test |
| 2.2 | `BaseBenchmark`/`BaseReporter` ABC 제거 | 4개 수정 + 2개 삭제 | 낮음 | 테스트 수정 후 통과 |

### Phase 3: 구조적 변경 (breaking change)

| Step | 내용 | 대상 파일 | 노력 | 검증 |
|:----:|------|-----------|:----:|------|
| 3.1 | Config 도메인별 분리 | ~15파일 | 중간 | 전체 테스트 + `.env` 기반 실행 확인 |

### Phase별 점수 기여 예상

| Phase | 범용성 기여 | OOP 기여 | 누적 예상 |
|:-----:|:----------:|:--------:|:---------:|
| Phase 1 완료 | — | +1.5 | 범용성 7, OOP 8.5 |
| Phase 2 완료 | +0.5 | +0.5 | 범용성 7.5, OOP 9 |
| Phase 3 완료 | +1.5 | — | **범용성 9, OOP 9** |

---

## 부록: Trade-off 요약

| 결정 사항 | 채택 방향 | 이유 |
|-----------|-----------|------|
| Config 분리 범위 | Data/Train/Benchmark만 (Pruning/Quant/Distill은 유지) | 2~3필드 단독 dataclass는 과잉 |
| 학습 루프 공통화 범위 | `unpack_batch`+`forward_model`만 (`train_loop` 미도입) | Distiller의 teacher forward 때문에 `train_loop` 사용처 실질 1곳 |
| `get_preprocessor()` 타입 | `Any` 유지 | 공통 부모 없음, 호출처에서 타입 분기 안 함 |
| ABC 제거 후 확장 시 | 2번째 구현체 추가 시점에 ABC 재추출 | "필요해질 때 도입" 원칙 |
| 태스크 확장 | TODO 주석만 (`F.cross_entropy` 유지) | 현재 2번째 태스크 없음 |
