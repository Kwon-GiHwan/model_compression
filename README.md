# Model Compression Pipeline

PyTorch 기반 모델 압축 프레임워크. **Pruning, Quantization, Knowledge Distillation** 6가지 방법론을 지원하며 이미지/NLP 모델 모두 사용 가능합니다.

## 주요 기능

- **Pruning**: Magnitude-based, Attention head pruning
- **Quantization**: Dynamic, Static (PTQ), Quantization-Aware Training (QAT)
- **Knowledge Distillation**: Response-based (logits matching)
- **프레임워크 지원**: 순수 PyTorch, HuggingFace Transformers
- **환경변수 기반 설정**: 코드 수정 없이 `.env` 파일만으로 전체 제어
- **벤치마킹**: 원본 vs 압축 모델 latency/parameter 비교
- **확장 가능 구조**: Registry 패턴으로 새 방법론 추가 시 코드 한 줄

## 설치

```bash
pip install -r requirements.txt
```

## 빠른 시작

```bash
# 1) .env 편집 (이미 존재) — 모드/모델/방법론 선택
vi .env

# 2) 실행
python main.py
```

## 모드

`MODE=` 환경변수로 선택:

| 값 | 동작 |
|-----|------|
| `apply` | 압축만 수행 → `OUTPUT_MODEL_PATH`에 저장 |
| `benchmark` | 원본 vs 압축본 성능 비교만 |
| `full` | apply → benchmark 연속 실행 (기본값) |

## 지원 방법론

`METHOD=` 환경변수에 다음 키 중 하나:

| 카테고리 | METHOD 값 | teacher 필요 | dataloader 필요 | 핵심 파라미터 |
|----------|-----------|:----:|:----:|---------------|
| Pruning | `pruning.magnitude` | ✗ | ✗ | `PRUNING_RATIO` |
| Pruning | `pruning.attention_head` | ✗ | ✗ | `PRUNING_RATIO` (HuggingFace 전용) |
| Quantization | `quantization.dynamic` | ✗ | ✗ | `QUANT_DTYPE` (qint8/float16) |
| Quantization | `quantization.static` | ✗ | ✓ | `QUANT_BACKEND`, `QUANT_CALIBRATION_BATCHES` |
| Quantization | `quantization.qat` | ✗ | ✓ | `QUANT_BACKEND`, `TRAIN_*` |
| Distillation | `distillation.response_based` | ✓ | ✓ | `DISTILL_TEMPERATURE`, `DISTILL_ALPHA`, `TRAIN_*` |

## 환경변수 전체

| 그룹 | 변수 |
|------|------|
| **공통** | `MODE`, `METHOD`, `MODEL_TYPE`, `MODEL_PATH`, `TASK`, `OUTPUT_MODEL_PATH`, `INPUT_SIZE` |
| **Teacher** (distillation) | `TEACHER_LOADER` (huggingface/local), `TEACHER_HF_REPO`, `TEACHER_HF_FILENAME`, `TEACHER_MODEL_PATH` |
| **Pruning** | `PRUNING_RATIO`, `PRUNING_DEVICE` |
| **Quantization** | `QUANT_DTYPE`, `QUANT_BACKEND`, `QUANT_CALIBRATION_BATCHES` |
| **Distillation** | `DISTILL_TEMPERATURE`, `DISTILL_ALPHA` |
| **학습** (QAT/Distill) | `TRAIN_EPOCHS`, `TRAIN_DEVICE` (cpu/cuda/mps), `TRAIN_LR` |
| **데이터** | `DATASET_TYPE` (hf_datasets/torchvision/local_folder), `DATASET_NAME`, `DATASET_CONFIG`, `DATASET_SPLIT`, `DATASET_PATH`, `DATASET_BATCH_SIZE`, `DATASET_MAX_LENGTH` |
| **벤치마크** | `BENCHMARK_DEVICE`, `BENCHMARK_RUNS`, `BENCHMARK_TYPE` (latency), `REPORTER_TYPE` (console) |

> 호환성 노트: `BENCHMARK_INPUT_SIZE`는 `INPUT_SIZE`로 rename됨. 기존 환경변수명도 fallback으로 동작.

## 시나리오별 예시

### A. BERT Magnitude Pruning

```bash
MODE=full
MODEL_TYPE=huggingface
MODEL_PATH=klue/bert-base
TASK=classification
METHOD=pruning.magnitude
PRUNING_RATIO=0.3
DATASET_TYPE=hf_datasets
DATASET_NAME=klue
DATASET_CONFIG=ynat
```

### B. Dynamic Quantization (가장 가벼움 — 데이터 불필요)

```bash
MODE=apply
MODEL_TYPE=huggingface
MODEL_PATH=klue/bert-base
METHOD=quantization.dynamic
QUANT_DTYPE=qint8
```

### C. Knowledge Distillation

```bash
MODE=full
MODEL_TYPE=huggingface
MODEL_PATH=klue/bert-base                # student
METHOD=distillation.response_based
TEACHER_LOADER=huggingface
TEACHER_HF_REPO=klue/roberta-large       # teacher
DISTILL_TEMPERATURE=4.0
DISTILL_ALPHA=0.7
TRAIN_EPOCHS=5
TRAIN_DEVICE=mps
TRAIN_LR=1e-4
DATASET_TYPE=hf_datasets
DATASET_NAME=klue
DATASET_CONFIG=ynat
DATASET_BATCH_SIZE=16
```

### D. ImageNet 모델 (torchvision)

```bash
MODEL_TYPE=pytorch
MODEL_PATH=./resnet50.pt
METHOD=pruning.magnitude
PRUNING_RATIO=0.5
DATASET_TYPE=local_folder
DATASET_PATH=./data/imagenet/train
INPUT_SIZE=224
```

### E. Static Quantization (calibration 데이터 필요)

```bash
MODE=apply
MODEL_TYPE=pytorch
MODEL_PATH=./model.pt
METHOD=quantization.static
QUANT_BACKEND=x86
QUANT_CALIBRATION_BATCHES=100
DATASET_TYPE=local_folder
DATASET_PATH=./data/calibration
```

## 출력

- `compressed_model.pt` (또는 `OUTPUT_MODEL_PATH` 지정 경로) — 압축된 모델
- `benchmark` 모드 결과는 **콘솔 출력**

콘솔 출력 예시:
```
============================================================
 📊 Benchmark Results
============================================================

🔷 Original Model
  • Avg Latency: 45.234 ms
  • Total Params: 110,000,000
  • Model Size: 420.50 MB

🔶 Compressed Model
  • Avg Latency: 31.812 ms
  • Total Params: 77,000,000
  • Model Size: 294.30 MB

📈 Improvement
  • Speedup: 1.42x
  • Size Reduction: 30.01%
  • Parameter Reduction: 30.00%
```

## 테스트

```bash
./run_tests.sh                    # 전체
./run_tests.sh unit              # unit만
./run_tests.sh integration       # integration만
./run_tests.sh all coverage      # coverage 포함 → htmlcov/index.html
```

직접 pytest 호출도 가능:
```bash
pytest tests/ -q
pytest tests/unit/test_methods.py -v
```

현재 베이스라인: **80 passed, 4 skipped**.

## 새 방법론 추가 (3단계)

1. `model_compression/methods/<category>/<name>.py` 작성
   - `BaseMethod` 상속, `apply()`, `from_config()`, `validate()` 구현
   - 외부 의존성이 필요하면 `requires_teacher()` / `requires_dataloader()` 오버라이드
2. `model_compression/methods/registry.py`에 등록:
   ```python
   _registry.register("<category>.<name>")(YourClass)
   ```
3. `.env`에서 `METHOD=<category>.<name>` 설정

상세 가이드: [`model_compression/methods/__init__.py`](model_compression/methods/__init__.py), [`model_compression/methods/base_method.py`](model_compression/methods/base_method.py) docstring

## CI/CD 자동화 (Self-hosted Runner)

이 저장소는 내부망 서버에 self-hosted runner를 두어 commit-driven 학습 자동화를 지원합니다.

- 트리거: `model_compression/**` 변경 시 자동 학습
- 산출물: 서버의 `${MODELS_DIR}/model_<commit>.tflite`
- 후속: `Kwon-GiHwan/NPU-Sim-Repo`로 `repository_dispatch` 발신

설정 가이드: [`docs/2026-04-26-ci-runner-setup.md`](docs/2026-04-26-ci-runner-setup.md)

## 프로젝트 구조

```
.
├── main.py                       # 진입점
├── config.py                     # Config (DataConfig/TrainConfig/BenchmarkConfig + flat)
├── model_compression/
│   ├── methods/                  # 압축 방법론
│   │   ├── pruning/              # magnitude, attention_head
│   │   ├── quantization/         # dynamic, static, qat
│   │   ├── distillation/         # response_based
│   │   ├── base_method.py        # BaseMethod ABC
│   │   ├── registry.py           # 등록/팩토리
│   │   └── utils.py              # unpack_batch, forward_and_extract_logits, extract_logits
│   ├── model/                    # 모델 래퍼 (PyTorch, HuggingFace) + teacher loader
│   ├── data/                     # 데이터 로더 (NLP, Image)
│   ├── benchmark/                # latency benchmark
│   ├── reporter/                 # console reporter
│   └── registry.py               # 범용 Registry 클래스
├── tests/
│   ├── unit/
│   └── integration/
├── scripts/
│   └── train.sh                  # CI 자동화용 서버 측 학습 실행 스크립트
├── docs/                         # 아키텍처 리뷰, 개선 설계서, runner 설정 가이드
└── .github/workflows/
    ├── test.yml                  # PR/푸시 자동 테스트
    └── train.yml                 # self-hosted runner 학습 자동화
```

## 아키텍처 / 설계 문서

- [`docs/2026-04-20-architecture-review.md`](docs/2026-04-20-architecture-review.md) — 아키텍처 리뷰
- [`docs/2026-04-20-improvement-design.md`](docs/2026-04-20-improvement-design.md) — 개선 설계서
- [`docs/compression-algorithms.md`](docs/compression-algorithms.md) — 압축 알고리즘 노트

## License

MIT
