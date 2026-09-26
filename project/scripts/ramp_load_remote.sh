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

HOSTS=(${PHONE_HOSTS:-s21 s26})
PORTS=(${PHONE_PORTS:-5201 5202})   # 09-26: 유선 싱크(Pi)는 5211/5212 (Pi의 시스템 iperf3.service가 5201을 상시 점유)

echo "=== 원격 램프 부하 오케스트레이터 (profile=${PROFILE}, target=${TARGET_IP}) ==="

echo "[1/2] 두 폰에 ramp_load.sh 배포"
for h in "${HOSTS[@]}"; do
  if ! scp -q "$LOCAL_RAMP" "${h}:ramp_load.sh"; then
    echo "!!! ${h} 배포 실패 — sshd/authorized_keys/~/.ssh/config 확인 (ssh ${h} echo ok로 먼저 테스트)" >&2
    exit 1
  fi
done

SSHR=(-o ConnectTimeout=8)
PLOG=ramp_last.log   # 폰 안의 로그 파일 (SSH가 끊겨도 남음)
fetch_logs() {
  for h in "${HOSTS[@]}"; do
    timeout 20 ssh "${SSHR[@]}" "$h" "cat ${PLOG} 2>/dev/null" 2>/dev/null | sed "s/^/[${h}] /" || echo "!!! ${h} 로그 회수 실패"
  done
}
cleanup() {
  echo
  echo ">>> 중단 — 원격 iperf3/ramp_load 정리 중"
  for h in "${HOSTS[@]}"; do
    timeout 10 ssh "${SSHR[@]}" "$h" 'pkill -x iperf3 2>/dev/null; pkill -f "[r]amp_load.sh" 2>/dev/null; true' &
  done
  wait
  fetch_logs
  exit 130
}
trap cleanup INT TERM

echo "[2/2] 동시 실행 (Ctrl-C로 두 폰 모두 즉시 정지)"
# 보완(09-24): h4에서 (a) 폰 부하가 출력 0줄로 통째로 빠지고 (b) AP 멈춤 때 SSH가 끊겨 30·40M 단계 로그가 사라짐.
# → 폰에서 nohup으로 분리 실행하고 로그는 폰 안의 파일(${PLOG})에 쓴 뒤 끝나면 회수한다.
#   12초 뒤 폰에서 ramp_load.sh가 안 돌면 그 폰만 재실행(같은 START_EPOCH라 진행 중인 단계에 합류).
START_EPOCH=$(( $(date +%s) + 5 ))   # 두 폰 공통 단계 기준 시각(ramp_load.sh 5번째 인자) — SSH 기동 여유 5초
echo "    단계 기준 시각 START_EPOCH=${START_EPOCH} ($(date -d @${START_EPOCH} +%H:%M:%S)), 폰 로그 ~/${PLOG}"
launch() {  # $1=index, $2=리다이렉트(> 새로 / >> 이어서)
  local h="${HOSTS[$1]}" p="${PORTS[$1]}"
  timeout 15 ssh "${SSHR[@]}" "$h" "nohup env REVERSE=${REVERSE:-0} bash ramp_load.sh ${p} ${PROFILE} ${TARGET_IP} ${PKT_LEN} ${START_EPOCH} ${2:->} ${PLOG} 2>&1 < /dev/null &"     || echo "!!! [$(date +%H:%M:%S)] ${h} 기동 SSH 실패"
}
running() { timeout 10 ssh "${SSHR[@]}" "$1" 'pgrep -f "[r]amp_load.sh" >/dev/null'; }   # 0=실행 중, 1=없음, 그 외=SSH 실패
for i in "${!HOSTS[@]}"; do launch "$i" ">" & done; wait
sleep 12
for i in "${!HOSTS[@]}"; do
  h="${HOSTS[$i]}"
  for try in 1 2; do
    running "$h" && break
    echo "!!! [$(date +%H:%M:%S)] ${h} ramp_load.sh 미실행 — 재실행 (${try}/2)"
    timeout 10 ssh "${SSHR[@]}" "$h" 'pkill -x iperf3 2>/dev/null; true'
    launch "$i" ">>"; sleep 8
  done
done

# 두 폰 모두 끝날 때까지 대기 (SSH 실패는 '실행 중'으로 간주 — AP 멈춤 동안 조기 종료 방지)
while :; do
  sleep 5; any=0
  for h in "${HOSTS[@]}"; do running "$h"; rc=$?; [ "$rc" -ne 1 ] && any=1; done
  [ "$any" -eq 0 ] && break
done
fetch_logs
echo
echo "=== 원격 램프 완료 — 파이 collect_metrics.py도 이제 중지할 것 ==="
