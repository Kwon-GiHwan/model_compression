# Model Compression Framework — 아키텍처 리뷰

> 범용성(Generality)과 객체지향적 효율성(OOP Efficiency) 분석  
> 작성일: 2026-04-20

---

## 1. 프로젝트 개요

5개 핵심 도메인(methods, model, data, benchmark, reporter)이 동일한 구조 패턴을 따르는 모델 압축 프레임워크.

```
compression_tool/
├── main.py                    # 진입점: run_apply(), run_benchmark()
├── config.py                  # @dataclass Config (30+ 환경변수 필드)
└── model_compression/
    ├── registry.py            # 범용 Registry (데코레이터 기반)
    ├── methods/               # 6개 구현체 (pruning 2, quantization 3, distillation 1)
    ├── model/                 # 2개 구현체 (PyTorch, HuggingFace) + loader/
    ├── data/                  # 2개 구현체 (NLP, Image)
    ├── benchmark/             # 1개 구현체 (Latency)
    └── reporter/              # 1개 구현체 (Console)
```

**공통 패턴**: 모든 모듈이 `base_xxx.py` (ABC) → 구현체 → `registry.py` (`Registry` 인스턴스 + `get_xxx(config)` 팩토리) 구조를 따른다.

**의존 흐름**: `main.py` → 각 모듈 `registry.py` → 구현체 → `base_xxx.py` (ABC)

---

## 2. 범용성 분석

### 2.1 강점: 높은 확장 용이성

Registry 패턴(`model_compression/registry.py`)이 범용적으로 설계되어, 새로운 압축 방법론 추가가 **3단계**로 완료된다:

1. `BaseMethod` 상속 구현체 작성
2. `methods/registry.py`에 `_registry.register("key")(ClassName)` 한 줄 추가
3. `.env`에서 `METHOD=key` 설정

이 패턴이 model, data, benchmark, reporter 모두에 일관 적용되어 학습 비용이 낮다.

### 2.2 강점: 멀티 도메인 지원

- `MagnitudePruner` (`magnitude_pruner.py:28-31`): `is_nlp` 플래그로 NLP/Vision 입력 분기
- `LatencyBenchmark` (`latency_benchmark.py:21-33`): `isinstance(batch, dict)` 체크로 NLP/Vision 공통 처리
- `extract_logits()` (`methods/utils.py`): HuggingFace 출력과 순수 텐서 모두 대응

### 2.3 한계: PyTorch 프레임워크 종속

`BaseModel` 주석에 "TensorFlow, JAX, ONNX 등"을 언급하지만, 실제 구현은 PyTorch에 강하게 결합되어 있다.

| 파일 | PyTorch 직접 의존 |
|------|-------------------|
| `dynamic_quantizer.py:21-22` | `torch.ao.quantization.quantize_dynamic` |
| `static_quantizer.py:27-41` | `torch.ao.quantization.prepare/convert` |
| `magnitude_pruner.py:33-39` | `torch_pruning` 라이브러리 |

현재 PyTorch만 지원하는 것 자체가 문제는 아니다. 다만 BaseModel 주석이 실제와 불일치하므로 정리가 필요하다.

### 2.4 한계: Classification 전용 설계

- `ResponseBasedDistiller` (`response_based.py:72`): `F.cross_entropy()` — 분류 전용
- `NLPDataLoader`: `label_column = "label"` 기본값
- generation, detection, segmentation 등은 현재 구조에서 적용 불가

### 2.5 한계: Config의 확장성 병목

단일 flat `@dataclass`에 모든 방법론의 파라미터가 혼재:

```python
# config.py — 방법론별 파라미터가 한 클래스에 혼재
PRUNING_RATIO: float       # pruning 전용
QUANT_DTYPE: str            # quantization 전용
DISTILL_TEMPERATURE: float  # distillation 전용
```

새 방법론 추가 시마다 Config 필드가 계속 증가하므로, 방법론 수가 늘어나면 비대해진다.

---

## 3. OOP 효율성 분석

### 3.1 SOLID 준수 평가

| 원칙 | 평가 | 근거 |
|------|------|------|
| **SRP** | ✅ 양호 | 각 클래스가 단일 책임 담당 |
| **OCP** | ✅ 양호 | Registry + ABC로 기존 코드 수정 없이 확장 가능 |
| **LSP** | ✅ 양호 | 모든 구현체가 `apply()` 계약 준수 |
| **ISP** | ⚠️ 미흡 | `apply(student, teacher, dataloader)` — 6개 중 3개가 teacher/dataloader 미사용 |
| **DIP** | ✅ 양호 | `main.py`가 registry를 통해 추상에 의존 |

### 3.2 ISP 위반의 실질적 영향

`BaseMethod.apply()` 시그니처:

```python
def apply(self, student: Any, teacher: Any = None, dataloader: Any = None) -> Any:
```

| 구현체 | teacher 사용 | dataloader 사용 |
|--------|:---:|:---:|
| DynamicQuantizer | ✗ | ✗ |
| MagnitudePruner | ✗ | ✗ |
| AttentionHeadPruner | ✗ | ✗ |
| StaticQuantizer | ✗ | ✓ |
| QATQuantizer | ✗ | ✓ |
| ResponseBasedDistiller | ✓ | ✓ |

다만 `requires_teacher()`/`requires_dataloader()`가 런타임 가드 역할을 하고, `main.py:21-27`에서 조건부 로딩하므로 **실용적 trade-off로 수용 가능**하다. 현재 규모에서 인터페이스 분리는 over-engineering이다.

### 3.3 ABC 정당성 평가

> 기준: "구현체 1개뿐인 interface 금지 — 실제 mock/교체 시점에 추출"

| ABC | 구현체 수 | 정당성 |
|-----|:---------:|--------|
| `BaseMethod` | 6 | ✅ 정당함 |
| `BaseModel` | 2 | ✅ 정당함 |
| `BaseDataLoader` | 2 | ✅ 정당함 |
| `BaseLoader` | 2 | ✅ 정당함 |
| **`BaseBenchmark`** | **1** | ⚠️ 의문 — `LatencyBenchmark`만 존재 |
| **`BaseReporter`** | **1** | ⚠️ 의문 — `ConsoleReporter`만 존재 |

accuracy benchmark, JSON reporter 등 확장 계획이 구체적이라면 유지, 아니라면 제거가 원칙에 부합한다.

### 3.4 Registry 패턴 정당성

`model_compression/registry.py`: 28줄, 5개 모듈에서 재사용. **충분히 정당하다.**

에러 메시지에 사용 가능한 키 목록을 포함(`registry.py:23`)하여 디버깅이 쉬운 점도 좋다.

### 3.5 `from_config()` — `@abstractmethod` 누락

모든 ABC에서 `from_config()`이 `raise NotImplementedError`로 구현되어 있다:

```python
# base_method.py:29-32
@classmethod
def from_config(cls, config: Any) -> "BaseMethod":
    raise NotImplementedError(...)
```

`@abstractmethod`가 아니므로 구현 누락이 런타임까지 발견되지 않는다. 해당 파일 5개:
- `base_method.py:29-32`
- `base_model.py:34-36`
- `base_dataloader.py:16-18`
- `base_benchmark.py:18-20`
- `base_reporter.py:18-20`

---

## 4. 강점 정리

| # | 항목 | 근거 |
|---|------|------|
| 1 | **일관된 아키텍처 패턴** | 모든 모듈이 ABC → 구현체 → Registry 구조. 한 모듈을 이해하면 나머지도 즉시 파악 가능 |
| 2 | **간결한 Registry** | 28줄로 필요한 기능만 제공. 에러 메시지에 사용 가능 키 목록 포함 |
| 3 | **NLP/Vision 공통 처리** | `extract_logits()`, `is_nlp` 플래그, `isinstance(batch, ...)` 분기로 실용적 처리 |
| 4 | **조건부 의존성 로딩** | `requires_teacher()`/`requires_dataloader()` 결과에 따라 불필요한 리소스 로딩 방지 |
| 5 | **테스트 구조** | unit + integration 분리, mock 활용, fixture 적절 |

---

## 5. 개선 제안

### 5.1 [높음] `Any` 타입 → 구체적 타입 어노테이션

**현재**: 핵심 인터페이스가 전부 `Any`로 선언.

```python
# base_method.py — 현재
def apply(self, student: Any, teacher: Any = None, dataloader: Any = None) -> Any:
```

**제안**: PyTorch 종속을 인정하고 구체적 타입 사용.

```python
# 개선안
def apply(self, student: nn.Module, teacher: nn.Module | None = None,
          dataloader: Iterable | None = None) -> nn.Module:
```

- **노력**: 낮음
- **영향**: IDE 자동완성, 타입 검사 도구 활용, 버그 조기 발견

### 5.2 [높음] `from_config()` → `@abstractmethod`

**현재**: `raise NotImplementedError` — 구현 누락이 런타임까지 발견 안 됨  
**제안**: `@abstractmethod`로 변경 — 인스턴스화 시점에 즉시 에러

- **노력**: 낮음 (5개 파일, 각 1줄 변경)
- **영향**: 새 구현체 작성 시 구현 누락 방지

### 5.3 [중간] Config 도메인별 분리

**현재**: 단일 flat Config에 30+ 필드 혼재  
**제안**: 방법론별 nested dataclass

```
Config.pruning: PruningConfig   (PRUNING_RATIO, PRUNING_DEVICE)
Config.quant: QuantConfig       (QUANT_DTYPE, QUANT_BACKEND, ...)
Config.distill: DistillConfig   (DISTILL_TEMPERATURE, DISTILL_ALPHA)
Config.train: TrainConfig       (TRAIN_EPOCHS, TRAIN_DEVICE, TRAIN_LR)
```

- **노력**: 중간 (Config 참조하는 모든 코드 수정 필요)
- **영향**: 관심사 분리, 방법론 10개 이상 시 필수

### 5.4 [중간] 학습 루프 코드 중복 제거

`ResponseBasedDistiller.apply()`와 `QATQuantizer.apply()`에서 학습 루프 패턴이 거의 동일하다:
- epoch 반복, `isinstance(batch, ...)` 분기, `.to(device)`, optimizer step, loss 출력

**제안**: 공통 학습 루프를 유틸리티 함수로 추출, loss 계산만 콜백 분리.

- **노력**: 낮음
- **영향**: ~20줄 중복 제거, 새 학습 기반 방법론 추가 시 보일러플레이트 감소

### 5.5 [낮음] `BaseBenchmark`/`BaseReporter` ABC 재검토

구현체가 각 1개. coding-style 원칙("구현체 1개뿐인 interface 금지")에 위반. 확장 계획이 구체적이지 않으면 제거 검토.

### 5.6 [낮음] `print()` → `logging` 전환

프레임워크로서 로그 레벨 제어가 가능해야 한다. `logging` 모듈로 전환 권장.

---

## 6. 종합 평가

| 평가 항목 | 점수 | 설명 |
|-----------|:----:|------|
| 아키텍처 일관성 | 9/10 | 모든 모듈이 동일 패턴. 매우 예측 가능한 구조 |
| 범용성 (확장 용이성) | 7/10 | Registry 기반 확장 우수. Config flat 구조와 classification 전용 한계 |
| OOP 효율성 | 7/10 | 패턴 사용이 대체로 적절. `Any` 남용과 불필요 ABC 2개 존재 |
| 타입 안전성 | 4/10 | 핵심 인터페이스가 전부 `Any`. 타입 체커 무용지물 |
| 코드 중복 | 6/10 | 학습 루프 중복, `isinstance(batch, ...)` 분기 패턴 반복 |
| 테스트 품질 | 7/10 | 단위 + 통합 테스트 존재. mock 활용 적절 |
| 실용성 | 8/10 | 즉시 사용 가능한 수준. 환경변수 기반 설정이 직관적 |

### 종합 등급: **B+ (양호)**

개인 프로젝트로서 구조적 완성도가 높다. Registry + ABC 패턴의 일관된 적용이 가장 큰 강점이다.

**가장 시급한 개선 2가지**:
1. `Any` 타입을 구체 타입으로 교체 (낮은 노력, 높은 영향)
2. `from_config()`을 `@abstractmethod`로 변경 (낮은 노력, 중간 영향)

이 두 가지만으로도 코드 품질이 의미 있게 향상된다. Config 분리는 방법론이 10개 이상으로 늘어나는 시점에 진행해도 무방하다.

---

## 부록: Trade-off 매트릭스

| 결정 사항 | 현재 유지 시 | 변경 시 |
|-----------|-------------|---------|
| Config flat 구조 | 단순, 환경변수 1:1 매핑 | 관심사 분리, OCP 준수 (환경변수 매핑 복잡도 증가) |
| `Any` 타입 | 프레임워크 교체 유연성 (이론적) | 타입 안전성, IDE 지원 (PyTorch 종속 명시화) |
| 단일 `apply()` 시그니처 | 일관된 인터페이스, `main.py` 단순 | ISP 준수 (`main.py` 복잡화, 현재 규모에선 과잉) |
| `BaseBenchmark`/`BaseReporter` ABC | 미래 확장 대비 | 간접층 제거, 규칙 준수 |
