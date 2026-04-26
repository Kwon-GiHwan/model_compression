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

## 운영

- 재부팅 후 자동 시작: systemd가 처리
- Runner 자체 업데이트: 보통 자동. 수동 시 `./config.sh remove` 후 재등록
- 로그: `journalctl -u actions.runner.* -f`

## GitHub 측 사전 설정

`Kwon-GiHwan/model_compression` → Settings → Secrets and variables → Actions 에서 등록.

### Secrets

| 이름 | 값 | 설명 |
|------|----|------|
| `NPU_DISPATCH_TOKEN` | 새로 발급한 fine-grained PAT | NPU-Sim-Repo에 `repository_dispatch` 트리거 용도 |

PAT 발급 시 설정:
- Repository access: `Kwon-GiHwan/NPU-Sim-Repo`만
- Permissions → Contents: `Read and write` (또는 Actions: Write)
- 만료 30~90일 권장

> **주의**: 기존에 채팅에 평문 노출된 PAT(`ghp_4YEj...`)는 즉시 Revoke 필요.  
> https://github.com/settings/tokens 에서 해당 토큰 삭제 후 재발급.

### Variables

| 이름 | 예시값 | 설명 |
|------|--------|------|
| `IMAGE` | `registry.local/ml-train:latest` | 학습 컨테이너 이미지 태그 |
| `MODELS_DIR` | `/home/user/models` | 학습 산출물(.tflite) 저장 경로 |
| `LOGS_DIR` | `/home/gihwan/ml-logs` | 학습 로그 저장 경로 |
