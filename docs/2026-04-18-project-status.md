# Model Compression Pipeline — 프로젝트 현황 (2026-04-18)

## 1. 프로젝트 개요

PyTorch 기반 ML 모델 압축 파이프라인. Pruning, Distillation, Quantization 3가지 압축 방법론을 Strategy + Registry 패턴으로 모듈화하여, 모델 타입(HuggingFace, PyTorch)과 데이터셋 타입(NLP, Image)에 무관하게 범용적으로 동작한다.

---

## 2. 완료된 작업 이력

### Phase 1: 구조 분석 & 기반 리팩토링

| 커밋 | 내용 |
|------|------|
| `9f0ce4e` | Base 인터페이스에서 PyTorch 의존성 제거 (`nn.Module` → `Any`, `DataLoader` → `Iterable`) |
| `1311a9a` | 구현체(PyTorchModel, HuggingFaceModel 등)에 새 인터페이스 반영 |
| `06e0ec9` | 공유 `Registry` 유틸리티 도입 — 모든 모듈에서 동일한 등록/조회 패턴 사용 |
| `b66ad36` | `main.py`에서 하드코딩된 메서드 분기 제거 — Registry 기반 디스패치로 전환 |
| `f263b76` | 멀티 아키텍처 지원 — AttentionHeadPruner에 BERT/GPT2/ViT/LLaMA 자동 감지 |
| `31cc72e` | `Config`를 `@dataclass`로 전환, DataLoader 파라미터화 (text_column, normalize_mean 등) |
| `24b6dbd` | Benchmark edge case 안전장치 (division-by-zero, p.element_size()) |
| `b345478` | 레거시 패턴 전면 제거 (get_tokenizer→get_preprocessor, AutoFeatureExtractor→AutoProcessor) |

### Phase 2: Quantization 추가

| 커밋 | 내용 |
|------|------|
| `29003ed` | Dynamic PTQ, Static PTQ, QAT 3종 구현 + 26개 테스트 |

- **DynamicQuantizer**: teacher/dataloader 불필요, 가장 간단한 양자화
- **StaticQuantizer**: calibration dataloader 필요 (`requires_dataloader=True`)
- **QATQuantizer**: training dataloader 필요 (`requires_dataloader=True`), `extract_logits` 유틸 사용

### Phase 3: OCP (Open-Closed Principle) 완성

| 커밋 | 내용 |
|------|------|
| `89d17c5` | "Decorative Registry" 안티패턴 제거 — 모든 모듈에 `from_config()` classmethod 적용 |

**Before (안티패턴):**
```python
def get_method(config):
    _registry.get(config.METHOD)          # 유효성만 검증
    if config.METHOD == "pruning.magnitude":  # 실제 분기는 if-elif
        return MagnitudePruner(...)
```

**After (OCP 준수):**
```python
def get_method(config):
    cls = _registry.get(config.METHOD)    # 클래스 조회
    return cls.from_config(config)        # 클래스가 스스로 생성
```

이 패턴이 **methods, model, data, loader, benchmark, reporter** 6개 레지스트리 전체에 일관 적용됨.

---

## 3. 현재 프로젝트 구조

```
compression_tool/
├── config.py                          # @dataclass Config (48개 필드, 환경변수 기반)
├── main.py                            # run_apply() / run_benchmark() / main()
├── model_compression/
│   ├── registry.py                    # 공유 Registry 유틸리티
│   ├── methods/
│   │   ├── base_method.py             # BaseMethod ABC (apply, validate, from_config, requires_*)
│   │   ├── registry.py                # METHOD 레지스트리 (6개 방법론)
│   │   ├── utils.py                   # extract_logits 유틸
│   │   ├── pruning/
│   │   │   ├── magnitude_pruner.py
│   │   │   └── attention_head_pruner.py
│   │   ├── distillation/
│   │   │   └── response_based.py
│   │   └── quantization/
│   │       ├── dynamic_quantizer.py
│   │       ├── static_quantizer.py
│   │       └── qat_quantizer.py
│   ├── model/
│   │   ├── base_model.py              # BaseModel ABC (load, save, get_raw, set_raw, get_preprocessor, from_config)
│   │   ├── registry.py                # MODEL_TYPE 레지스트리
│   │   ├── pytorch_model.py
│   │   ├── huggingface_model.py
│   │   └── loader/
│   │       ├── __init__.py            # TEACHER_LOADER 레지스트리
│   │       ├── base_loader.py
│   │       ├── local_loader.py
│   │       └── huggingface_loader.py
│   ├── data/
│   │   ├── base_dataloader.py         # BaseDataLoader ABC (get_dataloader, from_config)
│   │   ├── registry.py                # DATASET_TYPE 레지스트리
│   │   ├── nlp_dataloader.py
│   │   └── image_dataloader.py
│   ├── benchmark/
│   │   ├── base_benchmark.py
│   │   ├── registry.py                # BENCHMARK_TYPE 레지스트리
│   │   └── latency_benchmark.py
│   └── reporter/
│       ├── base_reporter.py
│       ├── registry.py                # REPORTER_TYPE 레지스트리
│       └── console_reporter.py
├── tests/
│   ├── conftest.py                    # 공유 fixtures (mock_preprocessor, config 등)
│   ├── conftest_models.py
│   ├── unit/
│   │   ├── test_methods.py
│   │   ├── test_quantization.py       # 26개 테스트
│   │   ├── test_model.py
│   │   ├── test_data.py
│   │   ├── test_config.py
│   │   ├── test_benchmark.py
│   │   └── test_reporter.py
│   └── integration/
│       ├── test_pipeline.py
│       └── test_end_to_end.py
└── docs/
    └── compression-algorithms.md
```

---

## 4. 핵심 설계 패턴

### 4.1 Strategy + Registry + from_config()

모든 확장 포인트가 동일한 3단계 패턴을 따른다:

1. **Base ABC 정의** — `apply()`, `from_config()` 등 인터페이스
2. **Registry 등록** — `_registry.register("key")(ConcreteClass)`
3. **from_config() 디스패치** — `cls = _registry.get(key); return cls.from_config(config)`

새 구현체 추가 시 기존 코드 수정이 불필요 (OCP 준수).

### 4.2 프레임워크 디커플링

Base 인터페이스는 `Any`, `Iterable` 등 제네릭 타입만 사용. PyTorch 임포트는 구현체 레벨에서만 발생.

### 4.3 Capability 메서드

```python
@classmethod
def requires_teacher(cls) -> bool: ...
@classmethod
def requires_dataloader(cls) -> bool: ...
```

`main.py`가 방법론의 요구사항을 동적으로 조회하여, 필요한 경우에만 teacher/dataloader를 로드.

---

## 5. 테스트 현황 (2026-04-18)

```
80 passed, 4 skipped (4.47s)
```

- Unit: methods, quantization, model, data, config, benchmark, reporter
- Integration: pipeline, end-to-end
- 테스트 패턴: `isinstance` + 속성 검증 (from_config 패턴과 호환)

---

## 6. 향후 고려사항

### 6.1 Config 구조 개선 (우선도: 중)

현재 `Config`는 48개 필드가 **평면(flat)** 구조로 되어 있다.

```python
@dataclass
class Config:
    METHOD: str
    PRUNING_RATIO: float      # pruning 전용
    QUANT_DTYPE: str           # quantization 전용
    DISTILL_TEMPERATURE: float # distillation 전용
    ...
```

**문제점:**
- 방법론별 필드가 뒤섞여 있어 어떤 필드가 어떤 방법론에 속하는지 불명확
- 새 방법론 추가 시 Config에 필드를 계속 추가해야 함 (Config이 모든 방법론을 알게 됨)

**개선 방향:**
- Nested Config 또는 method-specific config dict 분리
- 예: `Config.method_params: dict[str, Any]` → 각 `from_config()`에서 필요한 키만 추출
- 또는 `PruningConfig`, `QuantConfig` 등 서브 dataclass 도입

**주의:** 기존 환경변수 기반 설정과의 하위 호환성 유지 필요.

### 6.2 새 압축 방법론 확장 (우선도: 낮)

현재 지원:
- Pruning: Magnitude, Attention Head
- Distillation: Response-Based
- Quantization: Dynamic PTQ, Static PTQ, QAT

추가 고려 대상:
- **Structured Pruning** (채널/필터 단위)
- **Feature-Based Distillation** (중간 레이어 지식 전달)
- **GPTQ/AWQ** (LLM 특화 양자화)
- **Mixed-Precision Quantization**
- **Neural Architecture Search (NAS) 기반 압축**

추가 절차: 구현체 작성 → `from_config()` 구현 → registry 등록 → 테스트 작성 (기존 코드 수정 불필요).

### 6.3 Benchmark/Reporter 확장 (우선도: 낮)

현재 LatencyBenchmark + ConsoleReporter만 존재. Registry 패턴은 이미 적용됨.

추가 고려:
- **AccuracyBenchmark** — 정확도 비교 (데이터셋 필요)
- **MemoryBenchmark** — 메모리 사용량 비교
- **ThroughputBenchmark** — 초당 추론 횟수
- **JSONReporter** / **CSVReporter** — 파일 출력
- **WandBReporter** / **MLflowReporter** — 실험 트래킹 연동

### 6.4 모델 타입 확장 (우선도: 낮)

현재: PyTorchModel, HuggingFaceModel

추가 고려:
- **ONNXModel** — ONNX Runtime 기반 추론/압축
- **TensorFlowModel** — TF/Keras 모델 지원
- **SafeTensorsModel** — safetensors 포맷 지원

Base 인터페이스가 프레임워크에 무관하게 설계되어 있으므로, 구현체만 추가하면 됨.

### 6.5 테스트 강화 (우선도: 중)

- **실제 모델 E2E 테스트**: 현재 통합 테스트는 mock 기반. 소형 모델(distilbert-base 등)을 활용한 실제 압축 E2E 테스트 추가 고려
- **성능 회귀 테스트**: 압축률/정확도 변화를 추적하는 벤치마크 테스트
- **Config 유효성 검증 테스트**: 잘못된 Config 조합에 대한 에러 핸들링 검증

### 6.6 CI/CD 파이프라인 (우선도: 중)

- GitHub Actions 워크플로우 미설정
- pytest + coverage 자동 실행
- pre-commit hooks (black, isort, mypy)

### 6.7 문서화 (우선도: 낮)

- API 문서 자동 생성 (sphinx / mkdocs)
- 사용 예제 (README에 Quick Start)
- 아키텍처 다이어그램

---

## 7. 지원 방법론 요약표

| METHOD 키 | 클래스 | requires_teacher | requires_dataloader |
|-----------|--------|:---:|:---:|
| `pruning.magnitude` | MagnitudePruner | ✗ | ✗ |
| `pruning.attention_head` | AttentionHeadPruner | ✗ | ✗ |
| `distillation.response_based` | ResponseBasedDistiller | ✓ | ✓ |
| `quantization.dynamic` | DynamicQuantizer | ✗ | ✗ |
| `quantization.static` | StaticQuantizer | ✗ | ✓ |
| `quantization.qat` | QATQuantizer | ✗ | ✓ |

---

## 8. 레지스트리 요약표

| 레지스트리 | Config 키 | 등록된 구현체 | 파일 |
|-----------|-----------|-------------|------|
| METHOD | `config.METHOD` | 6개 (위 표 참조) | `methods/registry.py` |
| MODEL_TYPE | `config.MODEL_TYPE` | pytorch, huggingface | `model/registry.py` |
| DATASET_TYPE | `config.DATASET_TYPE` | hf_datasets, torchvision, local_folder | `data/registry.py` |
| TEACHER_LOADER | `config.TEACHER_LOADER` | local, huggingface | `model/loader/__init__.py` |
| BENCHMARK_TYPE | `config.BENCHMARK_TYPE` | latency | `benchmark/registry.py` |
| REPORTER_TYPE | `config.REPORTER_TYPE` | console | `reporter/registry.py` |

---

## 9. 새 구현체 추가 가이드 (예: Structured Pruning)

```python
# 1. 구현체 작성: model_compression/methods/pruning/structured_pruner.py
class StructuredPruner(BaseMethod):
    def __init__(self, pruning_ratio: float, target: str = "channels"):
        self.pruning_ratio = pruning_ratio
        self.target = target

    def apply(self, student, teacher=None, dataloader=None):
        # 구현
        return compressed_model

    @classmethod
    def from_config(cls, config):
        return cls(pruning_ratio=config.PRUNING_RATIO, target="channels")

# 2. 등록: model_compression/methods/registry.py 에 2줄 추가
from model_compression.methods.pruning.structured_pruner import StructuredPruner
_registry.register("pruning.structured")(StructuredPruner)

# 3. 사용: .env에서 METHOD=pruning.structured 설정
# 기존 코드 수정 없이 동작
```
