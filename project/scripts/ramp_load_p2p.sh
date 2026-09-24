#!/usr/bin/env bash
# 폰↔폰 부하 오케스트레이터 (2026-09-24). ramp_load_remote.sh와 같은 프로파일·속도를 쓰되,
# 목적지를 노트북이 아니라 상대 폰으로 바꿔 노트북(victim)에는 victim 프로브만 흐르게 한다.
#   s21g(192.168.8.191) → s26g(192.168.8.103):5201,  s26g → s21g:5202
# 사용법: bash ramp_load_p2p.sh [profile] [ignored_target] [pkt_len]   (두 번째 인자는 러너 호환용, 무시)
set -uo pipefail
PROFILE="${1:-step}"; PKT_LEN="${3:-1200}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; LOCAL_RAMP="${SCRIPT_DIR}/ramp_load.sh"
A=s21g; B=s26g; A_IP=192.168.8.191; B_IP=192.168.8.103
echo "=== 폰↔폰 램프 (profile=${PROFILE}) ${A}→${B_IP}:5201, ${B}→${A_IP}:5202 ==="
for h in $A $B; do scp -q "$LOCAL_RAMP" "${h}:ramp_load.sh" || { echo "!!! ${h} 배포 실패" >&2; exit 1; }; done
# 수신측 iperf3 서버 (이전 서버 정리 후 데몬으로)
ssh $B 'pkill -x iperf3 2>/dev/null; iperf3 -s -p 5201 -D' && ssh $A 'pkill -x iperf3 2>/dev/null; iperf3 -s -p 5202 -D' || { echo "!!! 폰 iperf3 서버 기동 실패" >&2; exit 1; }
sleep 1
START_EPOCH=$(( $(date +%s) + 5 ))   # 두 폰 공통 단계 기준 시각 (ramp_load.sh 5번째 인자)
PIDS=()
cleanup(){ for h in $A $B; do ssh -o ConnectTimeout=3 "$h" 'pkill -x iperf3 2>/dev/null; pkill -f "[r]amp_load.sh" 2>/dev/null; true' & done; wait; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup INT TERM
( ssh $A "bash ramp_load.sh 5201 ${PROFILE} ${B_IP} ${PKT_LEN} ${START_EPOCH}" 2>&1 | sed "s/^/[${A}] /" ) & PIDS+=($!)
( ssh $B "bash ramp_load.sh 5202 ${PROFILE} ${A_IP} ${PKT_LEN} ${START_EPOCH}" 2>&1 | sed "s/^/[${B}] /" ) & PIDS+=($!)
wait "${PIDS[@]}"
for h in $A $B; do ssh -o ConnectTimeout=3 "$h" 'pkill -x iperf3 2>/dev/null; true'; done
echo "=== 폰↔폰 램프 완료 ==="
