# Test Suite Documentation

이 디렉토리는 Model Compression Pipeline 프로젝트의 테스트 스위트를 포함합니다.

## 📁 구조

```
tests/
├── conftest.py              # Pytest fixtures 및 공통 설정
├── unit/                    # Unit tests (개별 컴포넌트 테스트)
│   ├── test_config.py       # Config 클래스 테스트
│   ├── test_model.py        # Model 모듈 테스트
│   ├── test_data.py         # Data 모듈 테스트
│   ├── test_methods.py      # Methods 모듈 테스트
│   ├── test_benchmark.py    # Benchmark 모듈 테스트
│   └── test_reporter.py     # Reporter 모듈 테스트
└── integration/             # Integration tests (컴포넌트 간 통합 테스트)
    ├── test_pipeline.py     # 파이프라인 통합 테스트
    └── test_end_to_end.py   # End-to-end 시나리오 테스트
```

## 🧪 테스트 실행

### 모든 테스트 실행
```bash
pytest
```

### 특정 카테고리 테스트 실행
```bash
# Unit tests만 실행
pytest tests/unit/

# Integration tests만 실행
pytest tests/integration/

# 특정 파일 테스트
pytest tests/unit/test_model.py

# 특정 테스트 케이스
pytest tests/unit/test_model.py::TestPyTorchModel::test_load_model
```

### 마커 기반 테스트 실행
```bash
# Unit tests 마커
pytest -m unit

# Integration tests 마커
pytest -m integration

# 느린 테스트 제외
pytest -m "not slow"
```

### 커버리지 리포트와 함께 실행
```bash
pytest --cov=model_compression --cov=config --cov-report=html --cov-report=term-missing
```

커버리지 HTML 리포트는 `htmlcov/index.html`에서 확인할 수 있습니다.

### Verbose 모드
```bash
pytest -v  # 각 테스트 이름 출력
pytest -vv # 더 자세한 출력
```

### 실패한 테스트만 재실행
```bash
pytest --lf  # Last Failed
pytest --ff  # Failed First
```

## 📋 테스트 범위

### Unit Tests

#### `test_config.py`
- Config 클래스의 기본값 검증
- 환경변수 오버라이드 테스트
- 타입 변환 테스트

#### `test_model.py`
- PyTorchModel 로드/저장 테스트
- HuggingFaceModel 로드/저장 테스트
- Model registry 테스트
- Tokenizer 처리 테스트

#### `test_data.py`
- ImageDataLoader 테스트
- NLPDataLoader 테스트
- DataLoader registry 테스트
- 데이터 전처리 테스트

#### `test_methods.py`
- MagnitudePruner 테스트
- AttentionHeadPruner 테스트
- ResponseBasedDistiller 테스트
- Method registry 테스트
- Validation 로직 테스트

#### `test_benchmark.py`
- LatencyBenchmark 실행 테스트
- 이미지/NLP 모델 벤치마킹
- 결과 타입 검증
- Warmup 동작 테스트

#### `test_reporter.py`
- ConsoleReporter 출력 테스트
- 비교 계산 정확도 테스트
- 리포트 포맷 검증

### Integration Tests

#### `test_pipeline.py`
- Pruning 파이프라인 통합 테스트
- Benchmark 파이프라인 통합 테스트
- Model loader 통합 테스트
- Registry 간 통합 테스트

#### `test_end_to_end.py`
- Magnitude pruning end-to-end 워크플로우
- Distillation end-to-end 워크플로우
- Full mode 실행 테스트
- 에러 처리 테스트

## 🛠 Fixtures

`conftest.py`에 정의된 공통 fixtures:

- `temp_dir`: 임시 디렉토리
- `mock_config`: Mock Config 객체
- `simple_cnn_model`: 테스트용 CNN 모델
- `simple_mlp_model`: 테스트용 MLP 모델
- `mock_tokenizer`: Mock tokenizer
- `sample_image_tensor`: 샘플 이미지 텐서
- `sample_nlp_batch`: 샘플 NLP 배치
- `env_file`: 임시 .env 파일

## 📊 테스트 통계

```bash
# 테스트 개수 확인
pytest --collect-only

# 실행 시간 측정
pytest --durations=10  # 가장 느린 10개 테스트 표시
```

## 🔧 개발 환경 설정

테스트 실행을 위한 개발 의존성 설치:

```bash
pip install -r requirements-dev.txt
```

## ✅ CI/CD

GitHub Actions나 다른 CI/CD 파이프라인에서 테스트를 실행하려면:

```yaml
- name: Run tests
  run: |
    pip install -r requirements-dev.txt
    pytest --cov=model_compression --cov-report=xml
```

## 📝 테스트 작성 가이드

새 기능을 추가할 때:

1. **Unit test 작성**: `tests/unit/`에 해당 모듈의 테스트 파일 생성
2. **Integration test 추가**: 다른 컴포넌트와의 상호작용 테스트
3. **Fixtures 활용**: `conftest.py`의 공통 fixtures 재사용
4. **Mock 사용**: 외부 의존성은 mock으로 대체
5. **명확한 테스트 이름**: `test_<action>_<expected_result>` 패턴 사용

예시:
```python
def test_load_model_from_local_path_succeeds(self):
    """Test that loading model from valid local path succeeds."""
    # Arrange
    model_path = "valid/path.pt"
    
    # Act
    result = loader.load(model_path)
    
    # Assert
    assert result is not None
```

## 🐛 디버깅

테스트 실패 시:

```bash
# 상세 출력과 함께 실행
pytest -vv --tb=long

# 첫 실패에서 중단
pytest -x

# PDB 디버거 진입
pytest --pdb
```

## 📈 목표 커버리지

- Overall: 80% 이상
- Critical modules (methods, model): 90% 이상
- Integration tests: 주요 워크플로우 100% 커버

현재 커버리지 확인:
```bash
pytest --cov=model_compression --cov-report=term-missing
```
