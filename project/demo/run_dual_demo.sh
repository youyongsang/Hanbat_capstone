#!/usr/bin/env bash
# 2026-10-06 비교 화면 실행: 2.4GHz 서버(:8000) + 5GHz 서버(:8001)를 같이 띄우고 http://localhost:8000/dual 을 연다.
# 사용(Git Bash, 저장소 어디서나):  bash project/demo/run_dual_demo.sh        # 시작
#                                   bash project/demo/run_dual_demo.sh stop   # 둘 다 종료
# 환경변수: PY(파이썬 경로, 기본 capstone conda env), P24/P5(포트), S21G/S26G(2.4GHz 폰 ssh 이름), S21/S26(5GHz 폰 ssh 이름)
# 전제: AP(Opal) 2.4GHz·5GHz 둘 다 켜짐(5GHz는 시연용 40MHz), Pi 유선 연결(iperf3 서버 5211/5212는 이 스크립트가 띄움), 폰 Termux sshd.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-/c/Users/dkssu/anaconda3/envs/capstone/python.exe}
P24=${P24:-8000}; P5=${P5:-8001}
LOG="$HERE/../.tmp/dual_demo_logs"; mkdir -p "$LOG"
# 부하 받는 쪽 = 유선 Pi(기본). 노트북으로 받으면 폰이 5GHz로 옮겨도 AP→노트북 구간이 노트북 대역(2.4GHz)에 남아
# 비교가 섞인다(그 구간은 Opal 점유율에 안 잡힘). Pi는 유선이라 부하가 폰이 붙은 대역에만 실린다. TARGET=192.168.8.226 이면 예전 방식.
TARGET=${TARGET:-192.168.8.109}; PORT21=${PORT21:-5211}; PORT26=${PORT26:-5212}; PI=${PI:-capstone@192.168.8.109}
# 종료와 기동을 따로 보낸다 — 한 명령줄에 "iperf3 -s -p 5211"이 들어 있으면 pkill -f가 그 SSH 셸 자신을 죽인다(10-06 README 주의와 같은 함정).
pi_kill(){ timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=6 "$PI" "pkill -f '[i]perf3 -s -p $PORT21' ; pkill -f '[i]perf3 -s -p $PORT26' ; true" ; }
pi_run(){ timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=6 "$PI" "$1" ; }

stop_port(){ powershell -NoProfile -Command "\$p=(Get-NetTCPConnection -LocalPort $1 -State Listen -ErrorAction SilentlyContinue).OwningProcess | Select-Object -First 1; if (\$p) { \$c=(Get-CimInstance Win32_Process -Filter \"ProcessId=\$p\").CommandLine; if (\$c -like '*demo_server.py*') { Stop-Process -Id \$p -Force; 'stopped :$1' } else { 'port $1 is not demo_server' } } else { 'nothing on :$1' }"; }

if [ "$1" = stop ]; then stop_port "$P24"; stop_port "$P5"; [ "$TARGET" = "${PI#*@}" ] && pi_kill && echo "Pi iperf3 종료"; exit 0; fi
if [ "$TARGET" = "${PI#*@}" ]; then
  pi_kill; pi_run "iperf3 -s -p $PORT21 -D && iperf3 -s -p $PORT26 -D && sleep 1 && echo Pi iperf3 \$(pgrep -c -x iperf3) 개 기동" || { echo "Pi iperf3 기동 실패 — Pi 전원·유선 확인"; exit 1; }
fi

export PYTHONIOENCODING=utf-8
nohup "$PY" -u "$HERE/demo_server.py" --band 24g --port "$P24" --s21 "${S21G:-s21g}" --s26 "${S26G:-s26g}" --iperf-target "$TARGET" --s21-port "$PORT21" --s26-port "$PORT26" > "$LOG/dual_24g.log" 2>&1 &
nohup "$PY" -u "$HERE/demo_server.py" --band 5g  --port "$P5"  --s21 "${S21:-s21}"  --s26 "${S26:-s26}" --iperf-target "$TARGET" --s21-port "$PORT21" --s26-port "$PORT26"  > "$LOG/dual_5g.log" 2>&1 &
for i in $(seq 1 40); do
  a=$(curl -s -m 2 "http://127.0.0.1:$P24/health"); b=$(curl -s -m 2 "http://127.0.0.1:$P5/health")
  [ -n "$a" ] && [ -n "$b" ] && break; sleep 1
done
echo "2.4GHz :$P24 → ${a:-응답 없음}   5GHz :$P5 → ${b:-응답 없음}   (로그: $LOG)"
URL="http://localhost:$P24/dual"; [ "$P5" != 8001 ] && URL="$URL?p5=$P5"
echo "비교 화면: $URL   (끄기: bash project/demo/run_dual_demo.sh stop)"
cmd //c start "" "$URL" >/dev/null 2>&1 || true
