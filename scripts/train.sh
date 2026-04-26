#!/usr/bin/env bash
# 학습 컨테이너 실행 + 산출물/로그 관리
# 사용: bash scripts/train.sh <COMMIT_SHA>
set -euo pipefail

COMMIT_SHA="${1:?commit sha required}"
SHORT_SHA="${COMMIT_SHA:0:8}"

# 환경변수 검증 (workflow에서 주입)
: "${IMAGE:?IMAGE not set}"
: "${MODELS_DIR:?MODELS_DIR not set}"
: "${LOGS_DIR:?LOGS_DIR not set}"

CONTAINER_NAME="ml-train-${SHORT_SHA}"
LOG_FILE="${LOGS_DIR}/train_${COMMIT_SHA}.log"
MODEL_OUT="${MODELS_DIR}/model_${COMMIT_SHA}.tflite"
WORKSPACE_HOST="$(pwd)"                  # runner의 _work 체크아웃 경로

mkdir -p "$MODELS_DIR" "$LOGS_DIR"

# 동일 SHA 재실행 시 잔존 컨테이너 정리
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "[train.sh] image=$IMAGE  container=$CONTAINER_NAME  commit=$COMMIT_SHA"

# 학습 컨테이너를 detached 모드로 실행
# - /workspace : 체크아웃된 소스 (read-only)
# - /output    : 모델 산출물 출력 디렉토리 (read-write)
# - 컨테이너 내부에서 PyTorch 학습 → .tflite 변환까지 수행한다고 가정
#   변환 결과를 /output/model.tflite 로 떨어뜨려야 함 (이미지 책임)
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size=8g \
  -v "${WORKSPACE_HOST}:/workspace:ro" \
  -v "${MODELS_DIR}:/output" \
  -e COMMIT_SHA="$COMMIT_SHA" \
  -w /workspace \
  "$IMAGE" \
  bash -c "set -e; python -m main && cp /workspace/compressed_model.tflite /output/model.tflite"

# 로그 실시간 리다이렉션 (tee로 stdout 동시 유지 → GitHub Actions UI에도 흐름)
docker logs -f "$CONTAINER_NAME" 2>&1 | tee "$LOG_FILE" &
LOG_PID=$!

# 컨테이너 종료 대기 + 종료 코드 수집
EXIT_CODE="$(docker wait "$CONTAINER_NAME")"

# 로그 tail 정리
wait "$LOG_PID" 2>/dev/null || true

# 산출물 commit-suffix로 rename
if [[ "$EXIT_CODE" -eq 0 ]]; then
  if [[ ! -f "${MODELS_DIR}/model.tflite" ]]; then
    echo "[train.sh] ERROR: 컨테이너는 성공했으나 model.tflite 출력이 없음"
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    exit 1
  fi
  mv "${MODELS_DIR}/model.tflite" "$MODEL_OUT"
  echo "[train.sh] OK   → $MODEL_OUT"
fi

# 컨테이너 cleanup (로그는 파일에 이미 보존됨)
docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true

exit "$EXIT_CODE"
