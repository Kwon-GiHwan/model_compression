# Model Compression Pipeline

PyTorch 기반 모델 압축 파이프라인으로, Pruning과 Knowledge Distillation을 지원하며 이미지/NLP 모델 모두 사용 가능합니다.

## 주요 기능

- **Pruning**: Magnitude-based pruning, Attention head pruning
- **Knowledge Distillation**: Response-based, Feature-based distillation
- **프레임워크 비종속**: PyTorch, HuggingFace Transformers 모두 지원
- **벤치마킹**: 원본 vs 압축 모델 성능/속도 비교
- **환경변수 기반 설정**: 코드 수정 없이 `.env` 파일만으로 전체 제어

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 환경 설정

`.env` 파일을 수정하여 모델, 데이터셋, 압축 방법을 설정합니다:

```bash
# 모드 선택
MODE=full              # apply: 압축만 / benchmark: 벤치마크만 / full: 둘 다

# 모델 설정
MODEL_TYPE=huggingface
MODEL_PATH=klue/bert-base
TASK=classification

# 압축 방법 선택
METHOD=pruning.magnitude
PRUNING_RATIO=0.3
```

### 2. 실행

```bash
python main.py
```

## 지원 압축 방법

### Pruning

**Magnitude Pruning** (`pruning.magnitude`)
- 가중치 크기 기반 구조적 프루닝
- `PRUNING_RATIO`: 제거할 가중치 비율 (0.0 ~ 1.0)

**Attention Head Pruning** (`pruning.attention_head`)
- Transformer의 어텐션 헤드 제거
- `PRUNING_RATIO`: 제거할 헤드 비율

### Knowledge Distillation

**Response-based** (`distillation.response_based`)
- Teacher 모델의 출력(logits)을 학습
- `DISTILL_TEMPERATURE`: 소프트맥스 온도
- `DISTILL_ALPHA`: Hard/Soft loss 가중치

**Feature-based** (`distillation.feature_based`)
- 중간 레이어 특징(feature)을 학습
- Transformer 모델에 특화

## 주요 설정 옵션

### 모델 설정

```bash
MODEL_TYPE=pytorch           # pytorch / huggingface
MODEL_PATH=./model.pt        # 로컬 경로 또는 HuggingFace repo ID
TASK=classification          # classification / detection / seq2seq
```

### 데이터셋 설정

```bash
DATASET_TYPE=hf_datasets     # hf_datasets / torchvision / local_folder
DATASET_NAME=klue            # HuggingFace dataset 이름
DATASET_CONFIG=ynat          # Dataset configuration
DATASET_SPLIT=train
DATASET_BATCH_SIZE=16
```

### Teacher 모델 (Distillation 시)

```bash
TEACHER_LOADER=huggingface
TEACHER_HF_REPO=klue/roberta-large
```

### 학습 설정

```bash
TRAIN_EPOCHS=20
TRAIN_DEVICE=mps             # cpu / cuda / mps
TRAIN_LR=1e-4
```

### 벤치마크 설정

```bash
BENCHMARK_DEVICE=mps
BENCHMARK_RUNS=100           # 평균을 위한 추론 횟수
```

## 예제

### 1. BERT 모델 Magnitude Pruning

```bash
# .env 설정
MODEL_TYPE=huggingface
MODEL_PATH=klue/bert-base
METHOD=pruning.magnitude
PRUNING_RATIO=0.3
MODE=full
```

```bash
python main.py
```

### 2. Knowledge Distillation (Teacher → Student)

```bash
# .env 설정
MODEL_TYPE=huggingface
MODEL_PATH=klue/bert-base                    # Student
METHOD=distillation.response_based
TEACHER_LOADER=huggingface
TEACHER_HF_REPO=klue/roberta-large          # Teacher
DISTILL_TEMPERATURE=4.0
DISTILL_ALPHA=0.7
MODE=full
```

```bash
python main.py
```

## 출력

압축 완료 후 생성되는 파일:
- `compressed_model.pt`: 압축된 모델
- `benchmark_report.json`: 성능 비교 결과 (JSON)
- `benchmark_report.txt`: 성능 비교 결과 (TXT)

벤치마크 리포트 예시:
```
=== Model Compression Benchmark ===
Original Model Size: 420.5 MB
Compressed Model Size: 294.3 MB
Compression Ratio: 70.0%

Accuracy: 88.5% → 87.2% (-1.3%)
Latency: 45.2ms → 31.8ms (-29.6%)
Throughput: 22.1 samples/s → 31.4 samples/s (+42.1%)
```

## 테스트

```bash
# 전체 테스트 실행
pytest

# Unit 테스트만
pytest tests/unit/

# Integration 테스트만
pytest tests/integration/
```

## 프로젝트 구조

```
.
├── model_compression/      # 압축 메서드
│   ├── methods/           # Pruning, Distillation 구현
│   ├── model/             # 모델 로더
│   ├── data/              # 데이터 로더
│   ├── trainer/           # 학습 루프
│   └── benchmark/         # 벤치마킹
├── tests/                 # 테스트 코드
├── main.py               # 실행 파일
├── config.py             # 환경 설정
└── .env                  # 환경 변수
```

## License

MIT
