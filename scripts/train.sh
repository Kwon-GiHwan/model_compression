#!/usr/bin/env bash
# 학습 컨테이너 실행 + 산출물/로그 관리
# 사용: bash scripts/train.sh <COMMIT_SHA>
#
# 동작:
#   1. demo 모델 다운로드 (torchvision MobileNetV3-Small)
#   2. main.py로 magnitude pruning 적용 → .pt 산출
#   3. .pt → ONNX → TFLite 변환 → /output/model.tflite
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
WORKSPACE_HOST="$(pwd)"

mkdir -p "$MODELS_DIR" "$LOGS_DIR"

# 동일 SHA 재실행 시 잔존 컨테이너 정리
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "[train.sh] image=$IMAGE  container=$CONTAINER_NAME  commit=$COMMIT_SHA"

# 학습+변환 컨테이너 — demo 시나리오:
#   torchvision MobileNetV3-Small → magnitude pruning(0.3) → TFLite (fp32)
docker run -d \
  --name "$CONTAINER_NAME" \
  --shm-size=2g \
  -v "${WORKSPACE_HOST}:/workspace:ro" \
  -v "${MODELS_DIR}:/output" \
  -e COMMIT_SHA="$COMMIT_SHA" \
  -w /workspace \
  "$IMAGE" \
  bash -c '
    set -e
    echo "=== [1/3] demo 모델 다운로드 ==="
    python /workspace/scripts/prepare_demo_model.py /tmp/demo_model.pt

    echo "=== [2/3] magnitude pruning ==="
    MODEL_PATH=/tmp/demo_model.pt \
    MODEL_TYPE=pytorch \
    MODE=apply \
    METHOD=pruning.magnitude \
    PRUNING_RATIO=0.3 \
    OUTPUT_MODEL_PATH=/tmp/compressed_model.pt \
    INPUT_SIZE=224 \
      python -m main

    echo "=== [3/3] PyTorch → TFLite 변환 ==="
    python /workspace/scripts/convert_to_tflite.py \
      /tmp/compressed_model.pt \
      /output/model.tflite \
      1,3,224,224

    echo "=== Done. /output/model.tflite ==="
    ls -la /output/model.tflite
  '

# 로그 실시간 리다이렉션 (tee로 stdout 동시 유지 → GitHub Actions UI에도 흐름)
docker logs -f "$CONTAINER_NAME" 2>&1 | tee "$LOG_FILE" &
LOG_PID=$!

EXIT_CODE="$(docker wait "$CONTAINER_NAME")"
wait "$LOG_PID" 2>/dev/null || true

# 산출물 commit-suffix로 rename (컨테이너가 /output/model.tflite로 떨어뜨림)
if [[ "$EXIT_CODE" -eq 0 ]]; then
  if [[ ! -f "${MODELS_DIR}/model.tflite" ]]; then
    echo "[train.sh] ERROR: model.tflite missing"
    docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
    exit 1
  fi
  mv "${MODELS_DIR}/model.tflite" "$MODEL_OUT"
  echo "[train.sh] OK   → $MODEL_OUT"
fi

docker rm "$CONTAINER_NAME" >/dev/null 2>&1 || true
exit "$EXIT_CODE"
