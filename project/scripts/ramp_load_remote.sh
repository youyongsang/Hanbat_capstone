#!/usr/bin/env bash
# 노트북에서 두 폰(s21/s26)에 SSH로 램프형 부하를 동시에 실행하는 오케스트레이터.
# ramp_load.sh(폰 로컬용)를 대체하는 게 아니라 그 위에서 돌아감 — 배포+동시 실행+정지를 대신함.
#
# 사전조건 (한 번만, 각 폰에서):
#   1) Termux에 openssh 설치 + sshd 기동(8022) + 노트북 공개키를 authorized_keys에 등록
#   2) 노트북 ~/.ssh/config의 Host s21/s26 User를 각 폰 `whoami` 값으로 채워넣기
#   3) `ssh s21 echo ok` / `ssh s26 echo ok`로 접속 확인
# (docs/yongsang/demo_api_spec.md "폰 부하 에이전트" 섹션과 동일 컨벤션: s21=191/5201, s26=S26/5202)
#
# 사용법: bash ramp_load_remote.sh [profile] [target_ip] [pkt_len]
#   profile   : step(기본, 계단식) | knee(무릎근처)  — project/scripts/ramp_load.sh와 동일 정의
#   target_ip : 부하 목적지, 기본 192.168.8.226(노트북) — 세션마다 다르면 인자로 덮어쓰기
#   pkt_len   : 기본 1200

set -uo pipefail

PROFILE="${1:-step}"
TARGET_IP="${2:-192.168.8.226}"
PKT_LEN="${3:-1200}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_RAMP="${SCRIPT_DIR}/ramp_load.sh"

HOSTS=(s21 s26)
PORTS=(5201 5202)

echo "=== 원격 램프 부하 오케스트레이터 (profile=${PROFILE}, target=${TARGET_IP}) ==="

echo "[1/2] 두 폰에 ramp_load.sh 배포"
for h in "${HOSTS[@]}"; do
  if ! scp -q "$LOCAL_RAMP" "${h}:ramp_load.sh"; then
    echo "!!! ${h} 배포 실패 — sshd/authorized_keys/~/.ssh/config 확인 (ssh ${h} echo ok로 먼저 테스트)" >&2
    exit 1
  fi
done

PIDS=()
cleanup() {
  echo
  echo ">>> 중단 — 원격 iperf3/ramp_load 정리 중"
  for h in "${HOSTS[@]}"; do
    ssh -o ConnectTimeout=3 "$h" 'pkill -f iperf3 2>/dev/null; pkill -f ramp_load.sh 2>/dev/null; true' &
  done
  wait
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
}
trap cleanup INT TERM

echo "[2/2] 동시 실행 (Ctrl-C로 두 폰 모두 즉시 정지)"
for i in "${!HOSTS[@]}"; do
  h="${HOSTS[$i]}"
  p="${PORTS[$i]}"
  ( ssh "$h" "bash ramp_load.sh ${p} ${PROFILE} ${TARGET_IP} ${PKT_LEN}" 2>&1 | sed "s/^/[${h}] /" ) &
  PIDS+=($!)
done

wait "${PIDS[@]}"
echo
echo "=== 원격 램프 완료 — 파이 collect_metrics.py도 이제 중지할 것 ==="
