#!/data/data/com.termux/files/usr/bin/bash
# 램프형 부하 생성 — 폰(Termux)에서 직접 실행.
# 목적: escalation 창(정상->심각으로 서서히 올라가는 구간) 확보.
#       blast형(단발 -b 고정) 부하는 순식간에 포화로 튀어 전이 구간이 안 나옴.
# 폰↔AP 원격제어(SSH)가 아직 없으므로 이 스크립트를 각 폰 Termux에 복사해 로컬로 실행한다.
#
# 사용법: bash ramp_load.sh <port> [profile] [target_ip] [pkt_len]
#   port      : 191=5201, S26=5202 (노트북 iperf3 서버와 맞출 것)
#   profile   : step(기본, 계단식 10/20/30/40M x 60s) | knee(무릎근처, 22M x 240s 고정)
#   target_ip : 기본 192.168.8.226 (노트북) — 세션마다 IP 바뀌면 인자로 덮어쓸 것
#   pkt_len   : 기본 1200 (바이트)
#
# 안전 상한(.work-log/current.md 2026-08-28 기준, 반드시 지킬 것):
#   45M 금지 / 단일 스텝 300~420s 넘지 말 것 / 런 사이 AP 60~90s 쿨다운 / 종료 즉시 프로세스 kill

set -euo pipefail

PORT="${1:?사용법: ramp_load.sh <port> [profile] [target_ip] [pkt_len]}"
PROFILE="${2:-step}"
TARGET_IP="${3:-192.168.8.226}"
PKT_LEN="${4:-1200}"

case "$PROFILE" in
  step)
    # 계단식: 10M -> 20M -> 30M -> 40M, 60초씩 (총 240초)
    STEPS=("10M:60" "20M:60" "30M:60" "40M:60")
    ;;
  knee)
    # 무릎근처: 22M 고정 240초 (두 폰 합계 ~44M, occ 50~75% 오르내림 노림)
    STEPS=("22M:240")
    ;;
  *)
    echo "알 수 없는 profile: $PROFILE (step 또는 knee)" >&2
    exit 1
    ;;
esac

echo "=== 램프형 부하 시작 (profile=${PROFILE}) ==="
echo "대상        : ${TARGET_IP}:${PORT}"
echo "패킷 크기   : ${PKT_LEN}"
echo "단계        : ${STEPS[*]}"
echo "안전 상한   : 45M 금지 / 스텝 300~420s 초과 금지 / 런 후 60~90s 쿨다운"
echo "중단 방법   : Ctrl-C (현재 단계만 즉시 중단, 다음 단계로 안 넘어감)"
echo

cleanup() {
  echo
  echo ">>> 중단됨 — iperf3 정리 중"
  pkill -P $$ iperf3 2>/dev/null || true
}
trap cleanup INT TERM

total=0
for step in "${STEPS[@]}"; do
  rate="${step%%:*}"
  dur="${step##*:}"
  total=$((total + dur))
  echo "--- [$(date +%H:%M:%S)] ${rate} x ${dur}s (누적 ${total}s) ---"
  iperf3 -u -c "${TARGET_IP}" -p "${PORT}" -l "${PKT_LEN}" -b "${rate}" -t "${dur}"
done

echo
echo "=== 램프 완료 (총 ${total}s) — 파이 collect_metrics.py도 이제 중지할 것 ==="
