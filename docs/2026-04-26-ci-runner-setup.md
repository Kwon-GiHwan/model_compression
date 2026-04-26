# Self-hosted Runner 설치 가이드

> 대상 서버: `gihwan-local` (192.168.45.241:2222, user `gihwan`)  
> 작성일: 2026-04-26

## 사전 요구사항

- Ubuntu/Linux x64
- Docker 설치 완료, daemon 실행 중
- outbound 443 (GitHub API) 연결 가능
- 인바운드 포트 개방 **불필요** — Runner가 GitHub로 long-poll 방식으로 연결

## 설치 절차

### 1. 작업 디렉토리 생성

```bash
ssh gihwan-local

mkdir -p ~/actions-runner && cd ~/actions-runner
```

### 2. Runner 바이너리 다운로드

```bash
RUNNER_VERSION="2.319.1"   # 실제 발급 시점의 최신
curl -o actions-runner-linux-x64.tar.gz -L \
  "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz"
tar xzf actions-runner-linux-x64.tar.gz
```

최신 버전 확인: https://github.com/actions/runner/releases

### 3. GitHub 등록 토큰 발급 후 config

GitHub repo → Settings → Actions → Runners → New self-hosted runner → Linux x64  
화면에 표시되는 1회용 토큰을 아래 `<REGISTRATION_TOKEN_FROM_GITHUB_UI>` 자리에 붙여넣기.

```bash
./config.sh \
  --url https://github.com/Kwon-GiHwan/model_compression \
  --token <REGISTRATION_TOKEN_FROM_GITHUB_UI> \
  --name gihwan-local-runner \
  --labels self-hosted,linux,x64,ml-server \
  --work _work \
  --unattended
```

### 4. systemd 서비스 등록

재부팅 후에도 자동 시작 + polling 자동 재연결.

```bash
sudo ./svc.sh install gihwan
sudo ./svc.sh start
sudo ./svc.sh status   # active (running) 확인
```

### 5. Docker 권한 부여

```bash
sudo usermod -aG docker gihwan
# 그룹 반영을 위해 systemd 서비스 재시작
sudo ./svc.sh stop && sudo ./svc.sh start
```

## 검증

GitHub repo → Settings → Actions → Runners 에서 `gihwan-local-runner`가 **Idle** 상태로 표시되어야 함.

## 서버 측 환경 설정 (.env)

이 워크플로우는 GitHub UI에 Secrets/Variables를 등록하지 않습니다. 모든 설정은
runner 머신의 로컬 파일에서 읽어옵니다 (self-hosted runner의 신뢰 환경 활용).

### 6. `.env` 파일 작성

```bash
# gihwan 사용자 홈 디렉토리에 작성
cat > /home/gihwan/.ml-train.env <<'EOF'
# 학습 컨테이너 이미지 태그
IMAGE=registry.local/ml-train:latest

# 산출물/로그 경로
MODELS_DIR=/home/user/models
LOGS_DIR=/home/gihwan/ml-logs

# NPU-Sim-Repo dispatch 권한 PAT (fine-grained, 만료 30~90일)
NPU_DISPATCH_TOKEN=ghp_새로발급받은토큰값
EOF

# 권한을 600(소유자만 읽기/쓰기)으로 강제 — 워크플로우가 이를 검증함
chmod 600 /home/gihwan/.ml-train.env
ls -l /home/gihwan/.ml-train.env  # -rw------- 확인
```

### 7. NPU_DISPATCH_TOKEN 발급

GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens

- Repository access: `Kwon-GiHwan/NPU-Sim-Repo` 만
- Permissions → Contents: `Read and write` (또는 Actions: Write)
- Expiration: 30~90일 권장

발급된 토큰을 위 `.env`의 `NPU_DISPATCH_TOKEN` 값으로 입력.

> **주의**: 기존에 채팅에 평문 노출된 PAT(`ghp_4YEj...`)는 즉시 Revoke 필요.  
> https://github.com/settings/tokens 에서 해당 토큰 삭제.

### 8. 백업 정책 권장

서버 백업 시 `.env` 파일은 제외하거나 별도 암호화 보관. 토큰이 평문이므로
백업 매체 노출이 곧 토큰 노출이다. `~/.backup-exclude` 같은 rsync exclude
리스트에 `*.env` 추가 권장.

## 운영

- 재부팅 후 자동 시작: systemd가 처리
- Runner 자체 업데이트: 보통 자동. 수동 시 `./config.sh remove` 후 재등록
- 로그: `journalctl -u actions.runner.* -f`
- 설정 변경: `/home/gihwan/.ml-train.env` 직접 수정 (다음 워크플로우 실행부터 반영)
- 토큰 만료 시: `.env`에서 `NPU_DISPATCH_TOKEN` 값만 갱신, GitHub UI 진입 불필요

## GitHub UI에서 등록할 것 (요약)

**없음.** 모든 자격증명/설정은 서버 `.env`에서 읽습니다. self-hosted runner
등록 시 1회용 registration token만 GitHub UI에서 발급받으면 됩니다 (그 외 영구
등록 항목 없음).
