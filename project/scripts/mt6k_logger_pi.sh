#!/bin/sh
# 11-10: MT6000(mt76, 순정 OpenWrt) 전달 실패·대기열 로거. Pi에서 실행, 지속 SSH 하나로 AP를 1초마다 읽어 CSV 한 줄/단말로 기록.
# txq_logger_pi.sh는 Opal(siwifi·wlan0) 경로만 읽어 MT6000 세션 전부 "No such file"만 남았음(11-10 확인) → 이 스크립트로 대체.
# 사용: mt6k_logger_pi.sh <out.csv> <초> [phy0] [phy0-ap0]
# 열:
#   ts            AP 시계(초)
#   mac           단말 MAC
#   ba_miss       mt76/tx_stats "BA miss count" — AP 전체 누적(Block Ack 미수신; 단말 쪽에서 프레임이 깨지면 증가)
#   aql_pend_us   AP 전체 AQL pending airtime 합계(us)
#   if_txdrop     인터페이스 tx_dropped 누적
#   aql_vo/vi/be/bk_us  단말별 AQL 큐 깊이(us, 전달 대기 airtime)
#   q_bytes q_pkts q_drops q_txpkts  단말별 aqm(TID 합): backlog 바이트·패킷, 큐 버림 누적, 송신 패킷 누적
#   tx_retries tx_failed  station 누적
#   ack_last ack_avg      단말이 보낸 ACK의 AP 수신 세기(dBm)
#   sig_avg               단말 신호 평균(dBm)
OUT=$1; SEC=$2; PHY=${3:-phy0}; IF=${4:-phy0-ap0}
D=/sys/kernel/debug/ieee80211/$PHY
echo "ts,mac,ba_miss,aql_pend_us,if_txdrop,aql_vo_us,aql_vi_us,aql_be_us,aql_bk_us,q_bytes,q_pkts,q_drops,q_txpkts,tx_retries,tx_failed,ack_last,ack_avg,sig_avg" > $OUT
exec timeout $SEC ssh -o BatchMode=yes -o ServerAliveInterval=5 root@192.168.8.1 "
while true; do
  T=\$(date +%s)
  BA=\$(awk '/BA miss count/{print \$4; exit}' $D/mt76/tx_stats 2>/dev/null)
  AP=\$(awk '/^total/{print \$2}' $D/aql_pending 2>/dev/null)
  TD=\$(cat /sys/class/net/$IF/statistics/tx_dropped 2>/dev/null)
  for S in $D/netdev:$IF/stations/*; do
    [ -d \"\$S\" ] || continue
    M=\${S##*/}
    Q=\$(awk '/Q depth/{print \$4\",\"\$7\",\"\$10\",\"\$13}' \$S/aql 2>/dev/null)
    A=\$(awk 'NR>1{b+=\$3;p+=\$4;d+=\$6;t+=\$11} END{print b\",\"p\",\"d\",\"t}' \$S/aqm 2>/dev/null)
    W=\$(iw dev $IF station get \$M 2>/dev/null | awk -F'\t' '/tx retries:/{r=\$NF} /tx failed:/{f=\$NF} /last ack signal:/{split(\$0,x,\":\");l=x[2]} /avg ack signal:/{a=\$NF} /signal avg:/{s=\$NF} END{gsub(/ dBm| /,\"\",l);gsub(/ dBm/,\"\",a);split(s,y,\" \");print r\",\"f\",\"l\",\"a\",\"y[1]}')
    echo \"\$T,\$M,\$BA,\$AP,\$TD,\$Q,\$A,\$W\"
  done
  sleep 1
done" >> $OUT 2>&1
