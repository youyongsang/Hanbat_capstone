#!/usr/bin/env bash
# 유선 싱크(Pi) 부하 오케스트레이터 (2026-09-26). 공장에 가까운 부하 방향을 재현하려고 만듦:
#   up   (셋업 A): 폰 → AP → 유선 Pi   = 카메라·센서가 서버로 올려 보내는 업링크. AP 송신 큐를 거의 안 채움.
#   down (셋업 B): 유선 Pi → AP → 폰   = 서버가 다른 무선 기기에 내려보내는 다운링크. AP 송신 큐가 victim 아닌 기기로 가는 트래픽으로 참.
# 두 셋업 모두 노트북(victim)에는 victim 프로브만 흐른다(기본 셋업은 부하 받는 쪽 = victim이었음, h3는 폰↔폰).
# Pi에 iperf3 서버(5211/5212)를 띄우고 ramp_load_remote.sh를 목적지 Pi로 호출한 뒤, 끝나거나 중단되면 그 서버만 정리한다.
# Pi에서 도는 victim 프로브(ProbeRunner의 iperf3 클라이언트)는 건드리지 않도록 서버 명령줄만 골라 끈다.
# 포트: Pi에는 시스템 iperf3.service가 5201을 상시 점유하므로 부하 서버는 5211/5212를 쓴다(PHONE_PORTS로 오케스트레이터에 전달).
# 사용법: WIRED_MODE=up|down bash ramp_load_wired.sh [profile] [ignored_target] [pkt_len]   (두 번째 인자는 러너 호환용, 무시)
set -uo pipefail
PROFILE="${1:-step}"; PKT_LEN="${3:-1200}"; MODE="${WIRED_MODE:-up}"
PI=capstone@192.168.8.109; PI_IP=192.168.8.109
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSHO=(-o BatchMode=yes -o ConnectTimeout=8)
export PHONE_PORTS="5211 5212"
case "$MODE" in up) export REVERSE=0 ;; down) export REVERSE=1 ;; *) echo "!!! WIRED_MODE는 up|down" >&2; exit 2 ;; esac
echo "=== 유선 싱크 램프 (mode=${MODE}, profile=${PROFILE}, Pi ${PI_IP}:5211/5212) ==="

stop_servers() {
  timeout 15 ssh "${SSHO[@]}" "$PI" 'pkill -f "[i]perf3 -s -p 521[12]" 2>/dev/null; true'
}
start_servers() {
  stop_servers
  for p in 5211 5212; do
    timeout 15 ssh "${SSHO[@]}" "$PI" "iperf3 -s -p $p -D --rcv-timeout 5000 2>/dev/null || iperf3 -s -p $p -D" \
      || { echo "!!! Pi iperf3 서버 :$p 기동 실패" >&2; return 1; }
  done
  sleep 1
  n=$(timeout 15 ssh "${SSHO[@]}" "$PI" 'pgrep -fc "[i]perf3 -s -p 521[12]"' | tr -d '\r\n ')
  [ "${n:-0}" -ge 2 ] || { echo "!!! Pi iperf3 서버 확인 실패 (${n:-0}/2)" >&2; return 1; }
  echo "    Pi iperf3 서버 5211·5212 기동 확인"
}

CP=""
cleanup() {
  echo ">>> 유선 싱크 램프 중단 — 오케스트레이터 정리 후 Pi 서버 종료"
  [ -n "$CP" ] && kill -TERM "$CP" 2>/dev/null && wait "$CP" 2>/dev/null
  stop_servers; exit 130
}
trap cleanup INT TERM

start_servers || exit 1
bash "${SCRIPT_DIR}/ramp_load_remote.sh" "$PROFILE" "$PI_IP" "$PKT_LEN" & CP=$!
wait "$CP"; rc=$?
stop_servers
echo "=== 유선 싱크 램프 완료 (mode=${MODE}, rc=${rc}) ==="
exit $rc
