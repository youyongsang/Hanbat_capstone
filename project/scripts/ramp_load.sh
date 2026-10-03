#!/data/data/com.termux/files/usr/bin/bash
# 램프형 부하 생성 — 폰(Termux)에서 직접 실행.
# 목적: escalation 창(정상->심각으로 서서히 올라가는 구간) 확보.
#       blast형(단발 -b 고정) 부하는 순식간에 포화로 튀어 전이 구간이 안 나옴.
# 폰↔AP 원격제어(SSH)가 아직 없으므로 이 스크립트를 각 폰 Termux에 복사해 로컬로 실행한다.
#
# 사용법: bash ramp_load.sh <port> [profile] [target_ip] [pkt_len]
#   port      : 191=5201, S26=5202 (노트북 iperf3 서버와 맞출 것)
#   profile   : step(기본, 계단식 10/20/30/40M x 60s) | knee(무릎근처, 22M x 240s 고정)
#               | light(경고↔혼잡, 13M x 300s) | gray(혼잡↔심각 회색지대, 19M x 300s)
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
  light)
    # 경고↔혼잡 밴드: 13M 고정 300초 (두 폰 합계 ~26M, occ ~45~58% dwell)
    STEPS=("13M:300")
    ;;
  gray)
    # 혼잡↔심각 회색지대: 19M 고정 300초 (두 폰 합계 ~38M, occ ~58~70% dwell)
    STEPS=("19M:300")
    ;;
  dgray30)
    # 09-27: 2.4GHz 다운링크 심각 확보용 — 30M 고정 300초 (두 폰 합 60M, Pi → 폰 다운링크 셋업 B에서 사용)
    STEPS=("30M:300")
    ;;
  dgray40)
    # 09-27: 다운링크 최대 부하 유지 — 40M 고정 240초 (폰당 45M 미만 안전 규칙 준수, h9 step 40M×2에서 크래시 없었음)
    STEPS=("40M:240")
    ;;
  dstep_hi)
    # 09-27: 2.4GHz 다운링크 심각(손실) 탐색 — 폰당 50→80M, 60초씩. 기존 "45M 금지"(업링크 AP 크래시 기준)를 넘는 시험이라
    # 반드시 러너 감시(AP 무응답·재부팅 감지) 아래에서만 사용. 두 폰 합 100~160M로 채널 용량 초과를 노림.
    STEPS=("50M:60" "60M:60" "70M:60" "80M:60")
    ;;
  dstep_mid)
    # 09-27: 혼합(셋업 C) 고부하 — 폰당 50→70M 60초씩. 업링크 폰 70M는 첫 시도(업링크 크래시 전례는 두 폰 합 160M) → 러너 FAILMAX=3 감시 하
    STEPS=("50M:60" "60M:60" "70M:60")
    ;;
  dgray50)
    STEPS=("50M:240")
    ;;
  dgray60)
    STEPS=("60M:240")
    ;;
  dgray70)
    STEPS=("70M:240")
    ;;
  dgray80)
    STEPS=("80M:240")
    ;;
  knee_lo)
    # 2.4GHz 경계 조건(덜 포화): 18M 고정 240초 (두 폰 합계 ~36M)
    STEPS=("18M:240")
    ;;
  knee_hi_probe)
    # knee_hi 사전 시험(미검증 조건): 26M 60초만 — AP 크래시 여부 확인용
    STEPS=("26M:60")
    ;;
  knee_hi)
    # 더 강한 포화: 26M 고정 240초 (두 폰 합계 ~52M). 미검증 조건 — 먼저 knee_hi_probe로 확인할 것
    STEPS=("26M:240")
    ;;
  heavy5g)
    # 5GHz 전용(2026-09-19): knee(22M)가 occ>35% 겨우 7행 — 2.4GHz 기준 프로파일이
    # 5GHz 용량엔 안 통함. step 마지막 단계(40M)에서만 순간 73% 관측 → 그 강도를 지속.
    STEPS=("40M:240")
    ;;
  heavy5g_probe60)
    # 60M/폰 미검증 강도 — 크래시 안전성 짧게 먼저 확인 (60초만)
    STEPS=("60M:60")
    ;;
  heavy5g_60)
    # 60M/폰 확장(짧은 probe로 안전 확인 후). 폰 합계 120M.
    STEPS=("60M:240")
    ;;
  heavy5g_probe80)
    # 80M/폰 미검증 — 2.4GHz는 80/80 10분에 완전 크래시 기록 있음. 60초만 먼저.
    STEPS=("80M:60")
    ;;
  heavy5g_80)
    STEPS=("80M:240")
    ;;
  heavy5g_80_120)
    # 80M: 60s 안전 확인됨, 240s는 크래시(2026-09-19). 그 사이 120s로 심각 밀도 노림.
    STEPS=("80M:120")
    ;;
  heavy5g_80_150)
    # 120s 안전 확인(2026-09-19, 이번엔 심각 저조 — RF 변동성). 크래시(~180s)에 더 근접.
    STEPS=("80M:150")
    ;;
  smallpkt5g_probe)
    # 소패킷(200B) 시도 — 같은 Mbps라도 PPS가 훨씬 높아짐.
    # 10M/200B ≈ PPS상 60M/1200B와 비슷한 부담(오늘 밤 안전권). 60초만 먼저 확인.
    STEPS=("10M:60")
    ;;
  smallpkt5g)
    STEPS=("10M:240")
    ;;
  step5g_30to70)
    # 5GHz 부하 패턴 다양성 확보용 계단형(26차 후속) — 30/40/50/60/70M x 60s.
    # 전부 개별 검증된 안전 구간 이하(75M까지 최대 400s 안전 확인됨).
    STEPS=("30M:60" "40M:60" "50M:60" "60M:60" "70M:60")
    ;;
  step5g40_5to40)
    # 10-03: 5GHz 40MHz 폭 학습 수집용. 40MHz에선 폰당 30M에서 이미 점유율 83%·심각(w40test) →
    # 정상~심각을 고르게 담도록 5M부터 촘촘히. 상한 40M(80MHz 70M 안전 기준보다 훨씬 아래).
    STEPS=("5M:45" "10M:45" "15M:45" "20M:45" "25M:45" "30M:45" "40M:45")
    ;;
  smallpkt5g40_10_20)
    # 10-03: 40MHz 소패킷(200B). 80MHz에선 10~20M/200B 거의 무반응이었음 — 40MHz에서 재확인.
    STEPS=("10M:60" "20M:60")
    ;;
  fixed5g40_30)
    # 10-03: 40MHz 고정 30M — 지속 심각(혼잡↔심각 경계 오가는 구간) 표본.
    STEPS=("30M:180")
    ;;
  fixed5g40_30_60)
    # 10-03: 재연결 직후 링크 확인용 짧은 고정 30M (w40test 점유율 83% vs f5 40% 원인 가리기).
    STEPS=("30M:60")
    ;;
  step5g40_10to60)
    # 10-03: f5에서 정상 링크면 40MHz도 폰당 40M까지 경고 수준 → 60M까지 확장(w40test에서 40MHz 70M 무크래시 확인).
    STEPS=("10M:45" "20M:45" "30M:45" "40M:45" "50M:45" "60M:45")
    ;;
  smallpkt5g40_10_25)
    # 10-03: f5 소패킷 20M/200B에서 점유율 80~90%·심각 확인 → 10~25M 계단으로 경계 구간까지.
    STEPS=("10M:45" "15M:45" "20M:45" "25M:45")
    ;;
  fixed5g40_50)
    STEPS=("50M:120")
    ;;
  step5g_40to80)
    # 마지막 단계 80M은 60초만 유지 — probe80/heavy80_120에서 이미 안전 확인된 지속시간.
    STEPS=("40M:60" "50M:60" "60M:60" "70M:60" "80M:60")
    ;;
  smallpkt5g_50_60)
    # 50M/200B는 20초에서 반응 확인됨 — 60초로 연장.
    STEPS=("50M:60")
    ;;
  smallpkt5g_60_45)
    # 60M/200B는 30초에서 load avg 3.40(오늘 밤 최고) — 45초로 소폭만 연장.
    STEPS=("60M:45")
    ;;
  smallpkt5g_probe20)
    # 10M/200B는 거의 무반응(occ 36.7%, idle 수준) — 2배로 짧게 재확인.
    # PPS가 2배로 뛰므로(크래시가 PPS/CPU 기반일 가능성) 짧게부터.
    STEPS=("20M:60")
    ;;
  smallpkt5g_probe50_micro)
    # 20M/200B도 거의 무반응 — 50M로 대폭 상향. PPS 합산 ~62,500(80M/1200B 크래시 때
    # ~16,667의 3.7배) → 안전 검증 전혀 없음. 20초만 초단기 확인.
    STEPS=("50M:20")
    ;;
  smallpkt5g_probe60)
    # 50M/200B 20초 확인됨(반응 있음, 크래시 없음). 60M은 더 높음 — 30초 먼저.
    STEPS=("60M:30")
    ;;
  smallpkt5g_60_120)
    STEPS=("60M:120")
    ;;
  heavy5g_probe70)
    # 60M 안전 / 80M 크래시 확인됨(2026-09-19) — 그 사이 70M 짧게 먼저 확인.
    STEPS=("70M:60")
    ;;
  heavy5g_70)
    STEPS=("70M:240")
    ;;
  heavy5g_probe75)
    # 70M 안전 / 80M 크래시 확인됨(2026-09-19) — 그 사이 75M 짧게 먼저 확인.
    STEPS=("75M:60")
    ;;
  heavy5g_75)
    STEPS=("75M:240")
    ;;
  heavy5g_75_long)
    # 75M 안전 확인됨(2026-09-19, 240s) — 심각 표본 확보 위해 400s로 연장.
    # 2.4GHz 안전 상한(300~420s) 그대로 준수.
    STEPS=("75M:400")
    ;;
  *)
    echo "알 수 없는 profile: $PROFILE (step | knee | light | gray)" >&2
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

# 보완(09-24): 단계 경계를 절대 시각(START_EPOCH, 5번째 인자)으로 맞춘다. 폰마다 단계 시작이 어긋나면
# 먼저 연결한 폰의 UDP가 큐를 채워 나중 폰의 제어 연결(TCP)이 30초씩 멈췄음(h4 step, 양쪽 폰 재현).
# 각 단계는 경계 1초 전에 끝나고, 두 폰이 링크가 빈 같은 순간에 다음 단계를 연결한다.
START_EPOCH="${5:-$(date +%s)}"
# 보완(09-26): REVERSE=1이면 iperf3 -R — 서버(유선 Pi)가 이 폰으로 보내는 다운링크 부하(셋업 B, ramp_load_wired.sh down)
DIR_FLAG=""; [ "${REVERSE:-0}" = "1" ] && DIR_FLAG="-R"
[ -n "$DIR_FLAG" ] && echo "방향        : 다운링크(-R, ${TARGET_IP} → 이 폰)"
total=0
for step in "${STEPS[@]}"; do
  rate="${step%%:*}"
  dur="${step##*:}"
  start=$(( START_EPOCH + total ))
  total=$((total + dur))
  wait_s=$(( start - $(date +%s) )); [ "$wait_s" -gt 0 ] && sleep "$wait_s"
  echo "--- [$(date +%H:%M:%S)] ${rate} x ${dur}s (누적 ${total}s) ---"
  # 보완(09-24): 혼잡 중 연결이 멈추거나 서버가 busy면 단계 전체가 조용히 빠지던 문제 —
  # 단계 마감 시각까지 남은 시간으로 재시도(타임라인 유지). hang 대비 timeout으로 감쌈.
  end=$(( START_EPOCH + total - 1 )); tries=0
  while :; do
    left=$(( end - $(date +%s) ))
    [ "$left" -lt 3 ] && break
    timeout $((left + 8)) iperf3 -u -c "${TARGET_IP}" -p "${PORT}" -l "${PKT_LEN}" -b "${rate}" -t "${left}" --connect-timeout 3000 ${DIR_FLAG} && break
    tries=$((tries + 1))
    echo "!! [$(date +%H:%M:%S)] iperf3 실패 (재시도 ${tries}) — 단계 남은 ${left}s"
    sleep 2
  done
done

echo
echo "=== 램프 완료 (총 ${total}s) — 파이 collect_metrics.py도 이제 중지할 것 ==="
