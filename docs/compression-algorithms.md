# 모델 압축 알고리즘 문서

## 1. 요약 (Summary)

`compression_tool`은 PyTorch 기반 모델 압축 프레임워크로, 추론 속도 향상과 모델 크기 감소를 목적으로 한다. 이미지 분류 모델과 HuggingFace Transformer 기반 NLP 모델 모두에 적용 가능하도록 설계되었다.

### 지원하는 압축 방법

| 방법 | 식별자 (`METHOD`) | 적용 대상 |
|------|-------------------|-----------|
| Magnitude Pruning | `pruning.magnitude` | 이미지 모델, Transformer |
| Attention Head Pruning | `pruning.attention_head` | HuggingFace Transformer |
| Response-Based Knowledge Distillation | `distillation.response` | 이미지 분류, NLP 분류 |

---

## 2. 전체 파이프라인 흐름도

```
[환경 변수 / .env]
        |
        v
   [Config 로드]
        |
        v
[모델 로딩 (student / teacher)]
        |
        v
[압축 방법 선택 (METHOD)]
        |
   +---------+----------+
   |                    |
[Pruning]         [Distillation]
   |                    |
   +----+----+          +--- ResponseBasedDistiller
        |    |
MagnitudePruner  AttentionHeadPruner
        |
        v
[압축된 모델 저장 (OUTPUT_MODEL_PATH)]
        |
        v
  [Benchmark 실행]
        |
        v
[결과 출력: latency, params, size]
```

---

## 3. 알고리즘 1: Magnitude Pruning

### 원리

Magnitude Pruning은 뉴런(필터, 채널)의 가중치 크기(magnitude)를 중요도의 대리 지표로 사용한다. 직관은 단순하다: 가중치 norm이 작은 뉴런은 출력에 미치는 영향이 적으므로 제거해도 모델 성능에 미치는 손실이 최소화된다.

이 구현은 **Structured Pruning** 방식을 사용한다. Unstructured Pruning이 개별 가중치를 0으로 만드는 것(sparse matrix)과 달리, Structured Pruning은 필터/채널 단위로 통째로 제거하여 실제 추론 속도 향상으로 이어진다.

`torch_pruning` 라이브러리의 dependency graph 추적을 활용하므로 아키텍처에 무관하게 적용된다. 즉, Conv 레이어를 제거하면 연결된 BatchNorm, 다음 Conv의 입력 채널 수가 자동으로 정합성을 유지하며 조정된다.

### 중요도 수식

각 뉴런(채널, 필터)의 중요도는 L2 norm으로 계산된다:

```
Importance(i) = ||W_i||_2
```

여기서 `W_i`는 i번째 필터(또는 채널)에 해당하는 가중치 텐서다. `p=2`는 L2 norm을 의미한다.

전체 pruning_ratio `r`에 따라 중요도 하위 `r * 100%`에 해당하는 구조를 제거한다.

### 구현 세부사항

```python
pruner = tp.pruner.MagnitudePruner(
    model,
    example_inputs=example_input,
    importance=tp.importance.MagnitudeImportance(p=2),
    pruning_ratio=self.pruning_ratio,
    ignored_layers=[],
)
pruner.step()
```

- `example_inputs`: dependency graph를 추적하기 위한 예시 입력. NLP 모델은 `{"input_ids": torch.zeros(1, 128, dtype=torch.long)}`, 이미지 모델은 `torch.randn(1, 3, 224, 224)` 형태.
- `MagnitudeImportance(p=2)`: L2 norm 기반 중요도 계산기.
- `pruner.step()`: 단일 호출로 중요도 계산 + 구조 제거 + 의존성 재정합 수행.
- `ignored_layers=[]`: 기본적으로 제외 레이어 없음. 필요 시 외부에서 주입 가능.
- 모델은 `copy.deepcopy`로 복사 후 `cpu`에서 `eval()` 상태로 pruning 수행 — 원본 모델 보존, 학습 관련 연산(BatchNorm running mean 갱신 등) 방지.

### 파라미터

| 파라미터 | 환경 변수 | 기본값 | 설명 |
|---------|-----------|--------|------|
| `pruning_ratio` | `PRUNING_RATIO` | `0.3` | 제거할 구조 비율 (0 < ratio < 1) |
| `input_size` | `BENCHMARK_INPUT_SIZE` | `224` | 이미지 입력 크기 (이미지 모델에만 해당) |
| `is_nlp` | `DATASET_TYPE == "hf_datasets"` 여부 | `False` | NLP 모드 여부 |

---

## 4. 알고리즘 2: Attention Head Pruning

### 원리

Transformer 아키텍처에서 Multi-Head Attention은 각 head가 입력 시퀀스의 서로 다른 부분에 집중하도록 설계된다. 그러나 실제로 모든 head가 동등하게 중요하지는 않다 — 일부 head는 중복된 패턴을 학습하거나 거의 활성화되지 않는다.

Attention Head Pruning은 각 head의 가중치 norm을 중요도 지표로 삼아 중요도가 낮은 head를 통째로 제거한다. head를 제거하면 `num_attention_heads`가 줄고, 이에 따라 Q/K/V projection 행렬의 크기도 감소하여 연산량이 직접적으로 줄어든다.

### 중요도 수식

각 레이어의 head `i`에 대한 중요도는 Query, Key, Value weight의 해당 슬라이스의 norm 합산으로 계산된다:

```
Importance(i) = ||W_Q[i*d_h : (i+1)*d_h]||_F
              + ||W_K[i*d_h : (i+1)*d_h]||_F
              + ||W_V[i*d_h : (i+1)*d_h]||_F
```

여기서:
- `d_h = attention_head_size`: 각 head의 차원 수 (`hidden_size / num_heads`)
- `W_Q, W_K, W_V`: 각각 Query, Key, Value projection 가중치 행렬
- `|| · ||_F`: Frobenius norm (`.norm()` 기본값)

슬라이싱 범위 `[i * d_h : (i+1) * d_h]`는 행(row) 방향으로 각 head에 해당하는 가중치 구간을 추출한다.

### 구현 세부사항

```python
for layer_idx, layer in enumerate(model.encoder.layer):
    attn = layer.attention.self
    num_heads = attn.num_attention_heads
    num_to_prune = max(1, int(num_heads * self.pruning_ratio))

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

    heads_to_prune = head_importance.argsort()[:num_to_prune].tolist()
    pruned_heads[layer_idx] = heads_to_prune

model.prune_heads(pruned_heads)
```

- `max(1, int(...))`: 최소 1개의 head는 반드시 제거 — pruning_ratio가 매우 낮더라도 동작 보장.
- `head_importance.argsort()[:num_to_prune]`: 중요도 오름차순 정렬 후 하위 `num_to_prune`개 선택 (가장 덜 중요한 head).
- `model.prune_heads(pruned_heads)`: HuggingFace `PreTrainedModel`의 내장 메서드. 레이어별로 지정된 head를 제거하고 내부 projection 행렬 크기를 자동 조정.
- 각 레이어를 독립적으로 처리 — 레이어마다 다른 수의 head를 제거할 수 있음.

### 적용 대상 제약

```python
def validate(self, config):
    if config.MODEL_TYPE != "huggingface":
        raise ValueError(
            "AttentionHeadPruner는 HuggingFace Transformer 모델만 지원합니다"
        )
```

`model.encoder.layer` 구조와 `model.prune_heads()` API에 의존하므로 HuggingFace `PreTrainedModel` 인터페이스를 따르는 모델에만 적용 가능하다.

### 파라미터

| 파라미터 | 환경 변수 | 기본값 | 설명 |
|---------|-----------|--------|------|
| `pruning_ratio` | `PRUNING_RATIO` | `0.3` | 각 레이어에서 제거할 head 비율 |

---

## 5. 알고리즘 3: Response-Based Knowledge Distillation

### 원리

Knowledge Distillation은 크고 정확한 Teacher 모델의 지식을 작은 Student 모델에 전이하는 기법이다. Response-Based Distillation은 Teacher의 최종 출력(logits)만을 활용한다.

핵심 아이디어는 **soft label**에 있다. 일반 cross-entropy 학습에서 레이블은 one-hot 벡터(`[0, 0, 1, 0, ...]`)다. 반면 Teacher의 softmax 출력은 확률 분포로, 예를 들어 "이 이미지는 고양이(70%)이지만 개(25%)와도 유사하다"는 클래스 간 관계 정보를 담는다. Student는 이 soft label을 학습함으로써 단순 레이블보다 풍부한 지식을 습득한다.

### Loss 수식

최종 loss는 KD loss와 CE loss의 가중 합산이다:

```
L_total = alpha * L_KD + (1 - alpha) * L_CE
```

**KD Loss (Kullback-Leibler Divergence):**

```
L_KD = T^2 * KL( softmax(z_s / T) || softmax(z_t / T) )
     = T^2 * sum[ softmax(z_t/T) * (log softmax(z_t/T) - log softmax(z_s/T)) ]
```

**CE Loss (Cross-Entropy):**

```
L_CE = CrossEntropy(z_s, y_true)
```

여기서:
- `z_s`: Student logits
- `z_t`: Teacher logits
- `T`: Temperature (소프트닝 강도)
- `alpha`: KD loss 가중치
- `y_true`: 실제 정답 레이블

**Temperature `T`의 역할**: `T > 1`로 나누면 softmax 분포가 더 완만(soft)해진다. T가 클수록 작은 클래스들의 확률도 상대적으로 크게 반영되어 Teacher가 가진 클래스 간 유사도 정보가 더 풍부하게 전달된다.

**`T^2` 스케일링**: KL divergence를 `T`로 나눈 logit으로 계산하면 gradient magnitude가 `1/T^2`로 줄어든다. `T^2`를 곱해 gradient 스케일을 원래대로 복원한다 (Hinton et al., 2015).

### 학습 과정

```python
optimizer = torch.optim.AdamW(student.parameters(), lr=self.lr)
T = self.temperature

for epoch in range(self.epochs):
    for batch in dataloader:
        # Teacher는 no_grad로 추론 (학습 없음)
        with torch.no_grad():
            teacher_logits = teacher(**inputs).logits

        student_logits = student(**inputs).logits

        kd_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T**2)

        ce_loss = F.cross_entropy(student_logits, labels)
        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

- Teacher는 `eval()` + `torch.no_grad()` 상태 고정 — Teacher 가중치는 변하지 않음.
- `F.kl_div`는 첫 번째 인자로 **log-확률**, 두 번째 인자로 **확률**을 받는다. 따라서 Student에는 `log_softmax`, Teacher에는 `softmax`를 적용.
- `reduction="batchmean"`: 배치 내 평균으로 정규화. PyTorch KL divergence의 수학적으로 올바른 reduction 방식.
- 배치 형식은 이미지(`(imgs, labels)` 튜플)와 NLP(dict 형태) 모두 처리 가능.

### 파라미터

| 파라미터 | 환경 변수 | 기본값 | 설명 |
|---------|-----------|--------|------|
| `temperature` | `DISTILL_TEMPERATURE` | `4.0` | Softmax 온도. 높을수록 soft label이 완만해짐 |
| `alpha` | `DISTILL_ALPHA` | `0.7` | KD loss 가중치. `1 - alpha`가 CE loss 가중치 |
| `epochs` | `TRAIN_EPOCHS` | `20` | 학습 epoch 수 |
| `lr` | `TRAIN_LR` | `1e-4` | AdamW 학습률 |
| `device` | `TRAIN_DEVICE` | `mps` | 학습 디바이스 |

---

## 6. 벤치마크 방법론

### 구조

`LatencyBenchmark`는 압축 전후 모델의 추론 성능을 일관된 조건에서 비교하기 위한 측정 도구다.

### 측정 항목

| 항목 | 설명 |
|------|------|
| `avg_latency_ms` | 전체 실행의 평균 추론 시간 (밀리초) |
| `min_latency_ms` | 최소 추론 시간 |
| `max_latency_ms` | 최대 추론 시간 |
| `total_params` | 모델 전체 파라미터 수 |
| `param_size_mb` | 파라미터 크기 (float32 기준, MB) |

파라미터 크기는 float32 가정 하에 계산된다:
```
param_size_mb = total_params * 4 bytes / (1024^2)
```

### Warmup 및 측정 절차

```python
# 1단계: Warmup (10회) — JIT 컴파일, CUDA/MPS 컨텍스트 초기화 제거
with torch.no_grad():
    for _ in range(10):
        model(**dummy) if isinstance(dummy, dict) else model(dummy)

# 2단계: 실제 측정 (BENCHMARK_RUNS회)
latencies = []
with torch.no_grad():
    for _ in range(runs):
        start = time.perf_counter()
        model(**dummy) if isinstance(dummy, dict) else model(dummy)
        latencies.append((time.perf_counter() - start) * 1000)
```

Warmup이 필요한 이유:
- GPU/MPS 디바이스는 첫 번째 호출에서 커널 컴파일 및 메모리 할당이 발생해 비정상적으로 느리다.
- 10회 warmup으로 디바이스를 "예열"하여 steady-state 성능을 측정한다.

`time.perf_counter()`를 사용하는 이유:
- Python에서 가장 정밀한 단조 증가 시계 (나노초 해상도).
- `time.time()`은 시스템 시계 조정에 영향을 받을 수 있으므로 부적합.

### 더미 입력 구성

```python
# NLP
dummy = {
    "input_ids": torch.zeros(1, config.DATASET_MAX_LENGTH, dtype=torch.long).to(device)
}

# 이미지
s = config.BENCHMARK_INPUT_SIZE
dummy = torch.randn(1, 3, s, s).to(device)
```

배치 크기 1의 단일 샘플로 측정하므로 실제 서빙 시의 단일 요청 지연(latency)에 해당한다. 처리량(throughput) 측정과는 구분된다.

---

## 7. 설정 파라미터 총정리

`Config` 클래스는 환경 변수(`.env` 파일 또는 시스템 환경 변수)에서 모든 설정을 로드한다.

### 공통 설정

| 환경 변수 | 기본값 | 타입 | 설명 |
|-----------|--------|------|------|
| `MODE` | `full` | str | 실행 모드 |
| `METHOD` | `pruning.magnitude` | str | 압축 방법 식별자 |
| `MODEL_TYPE` | `huggingface` | str | 모델 종류 (`huggingface` 등) |
| `MODEL_PATH` | `""` | str | 로컬 모델 경로 또는 HF repo ID |
| `TASK` | `classification` | str | 태스크 종류 |
| `OUTPUT_MODEL_PATH` | `compressed_model.pt` | str | 압축 결과 저장 경로 |

### Teacher 모델 설정 (Distillation 전용)

| 환경 변수 | 기본값 | 타입 | 설명 |
|-----------|--------|------|------|
| `TEACHER_LOADER` | `huggingface` | str | Teacher 로딩 방식 |
| `TEACHER_MODEL_PATH` | `""` | str | Teacher 로컬 경로 |
| `TEACHER_HF_REPO` | `""` | str | Teacher HuggingFace repo ID |
| `TEACHER_HF_FILENAME` | `""` (None) | str or None | Teacher 특정 파일명 |

### 데이터셋 설정

| 환경 변수 | 기본값 | 타입 | 설명 |
|-----------|--------|------|------|
| `DATASET_TYPE` | `hf_datasets` | str | 데이터셋 종류. `hf_datasets`이면 NLP 모드로 간주 |
| `DATASET_NAME` | `""` | str | HuggingFace datasets 이름 또는 로컬 경로 |
| `DATASET_CONFIG` | `""` (None) | str or None | 데이터셋 서브셋 설정 |
| `DATASET_SPLIT` | `train` | str | 사용할 split |
| `DATASET_PATH` | `""` | str | 로컬 데이터셋 경로 |
| `DATASET_BATCH_SIZE` | `16` | int | 배치 크기 |
| `DATASET_MAX_LENGTH` | `128` | int | NLP 최대 토큰 길이 |

### Pruning 설정

| 환경 변수 | 기본값 | 타입 | 설명 |
|-----------|--------|------|------|
| `PRUNING_RATIO` | `0.3` | float | 제거 비율 (0 < ratio < 1) |
| `PRUNING_DEVICE` | `cpu` | str | Pruning 수행 디바이스 |

### Distillation 설정

| 환경 변수 | 기본값 | 타입 | 설명 |
|-----------|--------|------|------|
| `DISTILL_TEMPERATURE` | `4.0` | float | KD softmax 온도 |
| `DISTILL_ALPHA` | `0.7` | float | KD loss 가중치 |
| `TRAIN_EPOCHS` | `20` | int | 학습 epoch 수 |
| `TRAIN_DEVICE` | `mps` | str | 학습 디바이스 |
| `TRAIN_LR` | `1e-4` | float | AdamW 학습률 |

### 벤치마크 설정

| 환경 변수 | 기본값 | 타입 | 설명 |
|-----------|--------|------|------|
| `BENCHMARK_DEVICE` | `mps` | str | 벤치마크 실행 디바이스 |
| `BENCHMARK_RUNS` | `100` | int | 측정 반복 횟수 (warmup 10회 별도) |
| `BENCHMARK_INPUT_SIZE` | `224` | int | 이미지 입력 해상도 (H=W) |

---

## 8. 실전 적용 사례: Video Processing 파이프라인 경량화

`video-processing` 프로젝트는 디지털 사이니지 시청자 분석 시스템으로, 엣지 디바이스에서 실시간 추론이 필수적이다. 아래는 이 프로젝트에서 적용한 경량화 방법론이다.

### 8.1 Width Multiplier를 통한 모델 스케일링 (OSNet)

Re-ID(재식별) 모델로 **OSNet (Omni-Scale Network)** 의 width multiplier 변형을 활용했다. Width multiplier는 네트워크 전체 채널 수를 일정 비율로 축소하는 방법이다.

| 변형 | Width Multiplier | 크기 | 출력 차원 | 비고 |
|------|-----------------|------|----------|------|
| OSNet x1.0 | 1.00 | 8.7 MB | 512 | 최대 정확도 |
| OSNet x0.5 | 0.50 | 2.56 MB | 512 | 중간 |
| **OSNet x0.25** | **0.25** | **886 KB** | **512** | **기본값, 엣지 배포용** |
| YoutuReID 2021 | - | 102 MB | 512 | 레거시 (교체 대상) |

**핵심 결정**: x0.25 변형 채택 → YoutuReID 대비 **113배 크기 감소** (102MB → 886KB), 출력 차원은 동일 512차원 유지.

Width multiplier의 원리:
```
원본 채널 수 C에 대해:
C_new = floor(C * multiplier)

예: Conv(64, 128) with multiplier=0.25
  → Conv(16, 32)
  → 파라미터: 64*128 = 8,192 → 16*32 = 512 (16배 감소)
```

모든 OSNet 변형은 MSMT17 데이터셋에서 사전학습되었으며, ONNX 형식으로 변환하여 배포한다.

### 8.2 INT8 양자화 (Post-Training Quantization)

학습 후 양자화(PTQ)로 float32 가중치를 int8로 변환하여 모델 크기와 추론 속도를 개선했다.

| 모델 | 원본 (float32) | 양자화 (int8) | 압축률 |
|------|---------------|--------------|--------|
| YuNet (얼굴 감지) | ~1.5 MB | int8 변형 사용 | ~4× |
| WHENet (Head Pose) | 17 MB | 4.6 MB | **3.7×** |

**INT8 양자화 원리**:
```
float32 값 범위 [r_min, r_max]를 int8 범위 [0, 255]로 매핑:

scale = (r_max - r_min) / 255
zero_point = round(-r_min / scale)
q = clamp(round(r / scale) + zero_point, 0, 255)

역양자화: r ≈ (q - zero_point) * scale
```

**적용 이유**:
- float32 → int8: 가중치 메모리 4배 감소
- INT8 연산은 대부분의 엣지 하드웨어(ARM NEON, Apple ANE)에서 네이티브 가속 지원
- 재학습 불필요 (PTQ) → 빠른 적용 가능
- 정확도 손실 최소 (YuNet, WHENet 모두 실사용에 문제 없는 수준)

### 8.3 ONNX Runtime 추론 최적화

모든 추론 모델을 **ONNX Runtime**으로 서빙하여 프레임워크 오버헤드를 제거했다.

```
학습 프레임워크 (PyTorch/TensorFlow)
        ↓ torch.onnx.export() 또는 tf2onnx
    ONNX 형식 (.onnx)
        ↓
  ONNX Runtime 추론
        ↓
  하드웨어별 최적화 (자동)
```

**ONNX Runtime이 제공하는 최적화**:
- **그래프 최적화**: 연산자 융합 (Conv+BN+ReLU → 단일 커널), 상수 폴딩, 불필요 연산 제거
- **하드웨어 비의존**: 동일 `.onnx` 파일을 CPU, GPU, Apple MPS, ARM 등에서 실행
- **PyTorch 제거**: 배포 시 PyTorch 의존성 불필요 → 패키지 크기 대폭 감소
- **배치 추론**: 동적 배치 크기 지원으로 throughput 최적화

**프로젝트 내 ONNX 모델 목록**:

| 모델 | 파일 | 용도 |
|------|------|------|
| YOLO11n | `yolo11n_person.pt` | Person Detection (Ultralytics 자체 런타임) |
| OSNet x0.25 | `osnet_x0_25_msmt17.onnx` | Re-ID 임베딩 추출 |
| OSNet x0.5 | `osnet_x0_5_msmt17.onnx` | Re-ID (대안) |
| OSNet x1.0 | `osnet_x1_0_msmt17.onnx` | Re-ID (대안) |
| YuNet | `face_detection_yunet_int8.onnx` | 얼굴 감지 |
| WHENet | `whenet.onnx` / `whenet_int8.onnx` | Head Pose 추정 |

### 8.4 경량화 전략 종합 비교

| 전략 | 방법 | 크기 감소 | 정확도 영향 | 재학습 필요 |
|------|------|----------|-----------|-----------|
| Width Multiplier | 채널 수 축소 (x0.25) | 113× (102MB→886KB) | 중간 (Re-ID 품질 약간 감소) | 사전학습 모델 사용 |
| INT8 양자화 | float32→int8 PTQ | 3.7× (WHENet) | 미미 | 불필요 |
| ONNX Runtime | 그래프 최적화 + 프레임워크 제거 | 추론 시간 감소 | 없음 (동일 연산) | 불필요 |

**실전 교훈**:
- Width multiplier + ONNX가 가장 높은 압축률 대비 정확도 보존율을 보였다
- INT8 양자화는 추가적인 크기 감소가 필요할 때 비용 없이 적용 가능한 보너스 전략
- Re-ID 모델은 갤러리 매칭(EMA, anchor gate, margin check)의 보완 로직 덕분에 경량 모델의 임베딩 품질 저하를 시스템 수준에서 보상할 수 있었다
