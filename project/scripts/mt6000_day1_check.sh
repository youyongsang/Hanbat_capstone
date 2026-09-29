#!/usr/bin/env bash
# MT6000 도착 첫날 점검 (2026-09-29 준비). 노트북/Pi에서 실행: bash mt6000_day1_check.sh [AP_IP]
# 확인할 것: ① 펌웨어·드라이버 ② 2.4GHz 인터페이스 이름 ③ station dump의 tx/rx duration·airtime weight
# ④ survey의 transmit/receive/BSS receive time ⑤ busy ≈ receive + transmit(= busy가 AP 자기 송신 포함)인지.
# 근거: OpenWrt 포럼 스레드 173524 #3114 (5GHz·순정 OpenWrt 스냅샷 출력). 2.4GHz·기본 펌웨어에서도 나오는지가 이번 확인 대상.
AP=${1:-${AP_IP:-192.168.8.1}}
SSH="ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new root@$AP"
echo "== ① 펌웨어·커널·무선 드라이버"
$SSH 'cat /etc/openwrt_release 2>/dev/null | grep -E "DISTRIB_(ID|RELEASE|REVISION)"; cat /etc/glversion 2>/dev/null; uname -r; ls /sys/module | grep -E "^mt76|^mt79|^mt7915" | tr "\n" " "; echo'
echo "== ② 무선 인터페이스 (2.4GHz 이름을 AP_INTERFACE로 쓸 것)"
$SSH 'for i in $(iw dev | awk "/Interface/{print \$2}"); do f=$(iw dev $i info | awk "/channel/{print \$2, \$3, \$4}"); echo "$i : $f"; done'
IF24=$($SSH 'for i in $(iw dev | awk "/Interface/{print \$2}"); do iw dev $i info | grep -qE "channel ([1-9]|1[0-4]) " && echo $i && break; done')
echo "   → 2.4GHz 인터페이스 후보: ${IF24:-(못 찾음 — 2.4GHz 라디오가 켜져 있는지 확인)}"
[ -z "$IF24" ] && exit 1
echo "== ③ station dump 필드 (기기 1대 이상 연결된 상태에서)"
$SSH "iw dev $IF24 station dump" | grep -E "^Station|tx duration|rx duration|airtime weight|expected throughput|tx bitrate" | head -20
$SSH "iw dev $IF24 station dump" | grep -q "tx duration" && echo "   ✅ tx/rx duration 있음" || echo "   ❌ tx duration 없음 — 순정 OpenWrt 설치 또는 WED 오프로드 끄기 검토"
echo "== ④⑤ survey (사용 중 채널, 5초 간격 두 번 → 차분)"
S1=$($SSH "iw dev $IF24 survey dump" | awk '/in use/{f=1} f&&/active|busy|receive|transmit/{gsub(/ms/,""); print $NF} f&&/transmit time/{exit}' | tr '\n' ' ')
sleep 5
S2=$($SSH "iw dev $IF24 survey dump" | awk '/in use/{f=1} f&&/active|busy|receive|transmit/{gsub(/ms/,""); print $NF} f&&/transmit time/{exit}' | tr '\n' ' ')
echo "   누적(active busy receive BSS-receive transmit): $S1 → $S2"
python3 - "$S1" "$S2" <<'PY' 2>/dev/null || echo "   (python3 없음 — 위 두 줄로 직접 계산)"
import sys
a = [float(x) for x in sys.argv[1].split()]; b = [float(x) for x in sys.argv[2].split()]
if len(a) < 5 or len(b) < 5: print("   ❌ survey 필드 부족:", len(a), "개"); sys.exit()
d = [y - x for x, y in zip(a, b)]; act = d[0] or 1
busy, rx, bss, tx = (100 * d[i] / act for i in (1, 2, 3, 4))
print(f"   5초 동안: busy {busy:.1f}% · receive {rx:.1f}% · BSS receive {bss:.1f}% · transmit {tx:.1f}% · busy-(rx+tx) {busy-rx-tx:.1f}%")
print("   ✅ busy가 AP 송신 포함(≈ rx+tx)" if abs(busy - (rx + tx)) < max(5, 0.2 * busy) else "   ⚠ busy ≠ rx+tx — 외부 간섭이 크거나 정의가 다름, 다운링크 step으로 재확인")
PY
echo "== 다음: 다운링크 step(폰당 10→40M, iperf3 -R) 한 번 돌리며 transmit % 와 busy % 가 부하를 따라 오르는지 확인 (Opal은 busy가 20→10%로 내려갔음)"
echo "   수집기: AP_IP=$AP AP_INTERFACE=$IF24 python3 collect_metrics.py <scenario>"
