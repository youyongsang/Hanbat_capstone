#!/usr/bin/env bash
# 11-07: 유선 Pi → AP → 노트북 다운링크 부하 오케스트레이터.
#   공장 시나리오: 유선 서버가 라인 끝·랙 뒤의 멀거나 가려진 무선 기기(HMI 태블릿·AGV 단말·무선 브리지)로
#   영상·지도·레시피를 대량으로 내려보내는 경우. 노트북 = 그 기기(부하 수신 + victim 프로브 수신).
#   질문: Opal 자리 3(노트북 −69)의 LSTM 우위가 "보내는 쪽이 무선 폰"이라서가 아니라 "AP → 약한 기기 구간"
#   때문인지. 이 부하는 폰 → AP 구간을 없애고 AP → 노트북 구간만 남긴다(양·단계는 노트북 방향 std와 같음).
# run_profile.sh의 RAMP_CMD 훅 호환: bash ramp_load_pi2lap.sh <profile> <target_ip> <pkt_len>
#   profile은 ramp_load.sh와 같은 이름(step·gray 등). 두 스트림(5201·5202)을 Pi에서 같은 START_EPOCH로 띄운다
#   = 폰 2대가 노트북으로 보내던 것과 같은 포트·같은 단계. 노트북 iperf3 서버 5201·5202는 run_profile.sh가 켠다.
# victim 프로브(collect_metrics ProbeRunner, Pi → 노트북 :5203)는 건드리지 않는다 — 정리는 520[12] 클라이언트만.
set -uo pipefail
PROFILE="${1:-step}"; TARGET_IP="${2:-192.168.8.226}"; PKT_LEN="${3:-1200}"
PI=capstone@192.168.8.109
SSHO=(-o BatchMode=yes -o ConnectTimeout=8)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORTS=(${PI2LAP_PORTS:-5201 5202})
pi(){ timeout 20 ssh "${SSHO[@]}" "$PI" "$@"; }
stop_load(){ pi "pkill -f '[r]amp_load.sh' 2>/dev/null; pkill -f '[i]perf3 -u -c ${TARGET_IP} -p 520[12]' 2>/dev/null; true"; }
fetch_logs(){ for p in "${PORTS[@]}"; do pi "cat ~/ramp_pi_${p}.log 2>/dev/null" | sed "s/^/[pi:${p}] /"; done; }

echo "=== Pi → 노트북 다운링크 램프 (profile=${PROFILE}, ${TARGET_IP}:${PORTS[*]}, pkt=${PKT_LEN}) ==="
pi 'command -v iperf3 >/dev/null' || { echo "!!! Pi에 iperf3 없음" >&2; exit 1; }
scp -q -o BatchMode=yes -o ConnectTimeout=8 "${SCRIPT_DIR}/ramp_load.sh" "${PI}:ramp_load.sh" || { echo "!!! Pi로 ramp_load.sh 복사 실패" >&2; exit 1; }
stop_load >/dev/null 2>&1

cleanup(){ echo ">>> 중단 — Pi 부하 정리"; stop_load; fetch_logs; exit 130; }
trap cleanup INT TERM

START_EPOCH=$(( $(date +%s) + 5 ))
for p in "${PORTS[@]}"; do
  pi "nohup bash ramp_load.sh ${p} ${PROFILE} ${TARGET_IP} ${PKT_LEN} ${START_EPOCH} > ~/ramp_pi_${p}.log 2>&1 < /dev/null &" \
    || echo "!!! [$(date +%H:%M:%S)] Pi :${p} 기동 SSH 실패"
done
sleep 10
n=$(pi "pgrep -fc '[r]amp_load.sh'" | tr -d '\r\n ')
[ "${n:-0}" -ge "${#PORTS[@]}" ] || echo "!!! [$(date +%H:%M:%S)] Pi ramp_load.sh ${n:-0}/${#PORTS[@]}개만 실행 중"

# 끝날 때까지 대기 (SSH 실패는 '실행 중'으로 간주 — 링크가 잠깐 막혀도 조기 종료하지 않음. Pi는 유선이라 드묾)
while :; do
  sleep 5
  r=$(pi "pgrep -fc '[r]amp_load.sh'"); rc=$?
  [ $rc -ne 0 ] && [ -z "$r" ] && continue     # SSH 실패
  [ "$(echo "$r" | tr -d '\r\n ')" = 0 ] && break
done
fetch_logs
echo "=== Pi → 노트북 다운링크 램프 완료 ==="
