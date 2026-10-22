"""5GHz 유도 (band steering) — 원인이 '우리 부하'일 때의 조치 (2026-10-22, MT6000 순정 OpenWrt).

채널 전환 실측(m28~m33)에서 우리 기기 부하가 원인이면 2.4GHz 안에서 채널을 옮겨도 점유율이 −1~−7pt뿐이었다.
부하가 따라오기 때문이다. 이때는 부하를 내는 기기를 같은 공유기의 5GHz로 옮겨야 2.4GHz가 비워진다.
cause_advisor.SteerRunner가 전환 알림(3폴링 중 2번 심각) + 원인 '우리 부하'일 때 이 모듈을 부른다.

순서
  1) 계획: 2.4GHz(phy0-ap0) 기기별 송신+수신 airtime(station dump "tx/rx duration" 차분, GAP초)을 재서
     가장 많이 쓰는 기기를 고른다(KEEP에 든 MAC 제외). 5GHz(phy1-ap0) 채널·점유율로 옮길 여유가 있는지 같이 본다.
  2) 실행(apply일 때만):
     a. 802.11v BSS Transition Management 요청 — 5GHz BSS의 neighbor report(rrm_nr_get_own)를 붙여
        "이쪽으로 옮겨 달라"고 요청한다. 연결을 끊지 않는 정식 방법이지만 AP 설정에 bss_transition·ieee80211k가
        켜져 있어야 하고, SSID가 다르면(지금 GL-MT6000-bac vs GL-MT6000-bac-5G) 기기가 무시할 수 있다.
     b. BTM_WAIT초 안에 5GHz에 안 나타나면 2.4GHz에서 내보내고(deauth) BAN_MS 동안 2.4GHz 재접속을 막는다.
        기기에 5GHz SSID가 저장돼 있으면 스스로 5GHz에 붙는다(저장 안 돼 있으면 연결이 끊긴 채로 남음 — 주의).
     c. WAIT초까지 5GHz station 목록에서 그 MAC이 보이면 성공.
  AP 설정은 바꾸지 않는다(uci 안 건드림). BTM을 쓰려면 사용자가 직접:
     uci set wireless.default_radio0.bss_transition=1; uci set wireless.default_radio0.ieee80211k=1
     uci set wireless.default_radio1.ieee80211k=1; uci commit wireless; wifi reload

사용:
  python project/scripts/band_steer.py                      # 계획만 (누구를 옮길지, 5GHz 여유)
  python project/scripts/band_steer.py --apply              # 가장 많이 쓰는 기기를 실제로 옮김
  python project/scripts/band_steer.py --apply --mac aa:bb:cc:dd:ee:ff
  python project/scripts/band_steer.py --keep <노트북 MAC>   # 옮기면 안 되는 기기 제외
"""
import argparse, json, os, re, subprocess, sys, time

AP = f"root@{os.environ.get('AP_IP', '192.168.8.1')}"
IF24 = os.environ.get("AP_INTERFACE", "phy0-ap0")
IF5 = os.environ.get("AP5_INTERFACE", "phy1-ap0")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6"]


def run(cmd, timeout=30):
    r = subprocess.run(SSH + [AP, cmd], capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr.strip()


def sh(cmd, timeout=30):
    return run(cmd, timeout)[1]


def station_dump(iface):
    """mac → {tx_us, rx_us, signal, tx_bitrate}"""
    out, cur = {}, None
    for line in sh(f"iw dev {iface} station dump").splitlines():
        s = line.strip()
        if s.startswith("Station "):
            cur = out.setdefault(s.split()[1].lower(), {"tx_us": None, "rx_us": None, "signal": None, "tx_bitrate": None}); continue
        if cur is None: continue
        if s.startswith("tx duration:"): cur["tx_us"] = int(re.search(r"(\d+)", s).group(1))
        elif s.startswith("rx duration:"): cur["rx_us"] = int(re.search(r"(\d+)", s).group(1))
        elif s.startswith("signal:"): cur["signal"] = float(re.search(r"(-?\d+)", s).group(1))
        elif s.startswith("tx bitrate:"):
            m = re.search(r"([\d.]+) MBit/s", s); cur["tx_bitrate"] = float(m.group(1)) if m else None
    return out


def airtime(iface, gap=5.0):
    """기기별 (송신+수신) airtime % — 두 번 읽은 누적 µs 차분."""
    a = station_dump(iface); t0 = time.time(); time.sleep(gap); b = station_dump(iface); dt = max(time.time() - t0, 1e-3)
    rows = []
    for mac, s in b.items():
        p = a.get(mac)
        if not p or None in (s["tx_us"], p["tx_us"], s["rx_us"], p["rx_us"]):
            continue
        d = (s["tx_us"] - p["tx_us"]) + (s["rx_us"] - p["rx_us"])
        rows.append({"mac": mac, "airtime": round(max(d, 0) / 1e6 / dt * 100, 1), "signal": s["signal"], "tx_bitrate": s["tx_bitrate"]})
    return sorted(rows, key=lambda r: -r["airtime"])


def band_status(iface, gap=3.0):
    """그 대역의 채널·폭·점유율 % (survey in-use 차분). 인터페이스가 없으면 None."""
    info = sh(f"iw dev {iface} info")
    m = re.search(r"channel (\d+) \((\d+) MHz\), width: (\d+) MHz", info)
    if not m:
        return None
    def busy():
        cur = None; act = bsy = None
        for line in sh(f"iw dev {iface} survey dump").splitlines():
            s = line.strip()
            if s.startswith("frequency:"): cur = "in use" in s
            elif cur and s.startswith("channel active time:"): act = int(re.search(r"(\d+)", s).group(1))
            elif cur and s.startswith("channel busy time:"): bsy = int(re.search(r"(\d+)", s).group(1))
        return act, bsy
    (a0, b0) = busy(); time.sleep(gap); (a1, b1) = busy()
    occ = (b1 - b0) / (a1 - a0) * 100 if None not in (a0, b0, a1, b1) and a1 > a0 else None
    return {"channel": int(m.group(1)), "width": int(m.group(3)), "occupancy": None if occ is None else round(occ, 1),
            "clients": len(station_dump(iface))}


def plan(keep=(), gap=5.0, min_airtime=5.0):
    keep = {k.lower() for k in keep}
    stas = airtime(IF24, gap)
    five = band_status(IF5)
    cands = [s for s in stas if s["mac"] not in keep and s["airtime"] >= min_airtime]
    target = cands[0] if cands else None
    rec = None
    if target and five:
        rec = {"mac": target["mac"], "airtime": target["airtime"], "five_channel": five["channel"],
               "commands": [f"ubus call hostapd.{IF24} bss_transition_request '{{\"addr\":\"{target['mac']}\",\"neighbors\":[<5GHz neighbor report>],\"disassociation_imminent\":true,\"disassociation_timer\":20}}'",
                            f"ubus call hostapd.{IF24} del_client '{{\"addr\":\"{target['mac']}\",\"reason\":5,\"deauth\":true,\"ban_time\":120000}}'"]}
    return {"stations_24": stas, "five": five, "recommendation": rec,
            "why_none": None if rec else ("5GHz 인터페이스 없음" if not five else "airtime이 큰 2.4GHz 기기 없음")}


def on_band(mac, iface):
    return mac.lower() in station_dump(iface)


def steer(mac, wait=30.0, btm_wait=8.0, ban_ms=120000):
    mac = mac.lower(); t0 = time.time(); log = []
    # a) 802.11v BTM (연결 유지)
    rc, out, err = run(f"ubus call hostapd.{IF5} rrm_nr_get_own")
    nr = None
    if rc == 0:
        try: nr = json.loads(out)["value"][2]
        except Exception: nr = None
    if nr:
        req = json.dumps({"addr": mac, "neighbors": [nr], "abridged": True, "disassociation_imminent": True,
                          "disassociation_timer": 20, "validity_period": 30, "dialog_token": 1})
        rc, _o, err = run(f"ubus call hostapd.{IF24} bss_transition_request '{req}'")
        log.append(f"BTM 요청 {'보냄' if rc == 0 else '실패: ' + (err or str(rc))}")
        if rc == 0:
            while time.time() - t0 < btm_wait:
                time.sleep(1.5)
                if on_band(mac, IF5):
                    return {"mac": mac, "moved": True, "method": "BTM", "seconds": round(time.time() - t0, 1), "log": log}
    else:
        log.append("BTM 불가 (5GHz neighbor report 없음 — ieee80211k 꺼짐)")
    # b) 내보내기 + 2.4GHz 재접속 잠시 금지 → 기기가 저장된 5GHz SSID로 붙기를 기다림
    req = json.dumps({"addr": mac, "reason": 5, "deauth": True, "ban_time": int(ban_ms)})
    rc, _o, err = run(f"ubus call hostapd.{IF24} del_client '{req}'")
    log.append(f"2.4GHz 내보내기(재접속 {ban_ms // 1000}초 금지) {'완료' if rc == 0 else '실패: ' + (err or str(rc))}")
    while time.time() - t0 < wait:
        time.sleep(2)
        if on_band(mac, IF5):
            return {"mac": mac, "moved": True, "method": "내보내기", "seconds": round(time.time() - t0, 1), "log": log}
    return {"mac": mac, "moved": False, "method": "내보내기" if rc == 0 else None, "seconds": round(time.time() - t0, 1), "log": log}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제로 옮김 (기본은 계획만)")
    ap.add_argument("--mac", default=None, help="옮길 기기를 직접 지정")
    ap.add_argument("--keep", nargs="*", default=[], help="옮기면 안 되는 기기 MAC")
    ap.add_argument("--gap", type=float, default=5.0)
    ap.add_argument("--json", action="store_true"); a = ap.parse_args()
    p = plan(a.keep, a.gap)
    if not a.json:
        f = p["five"]
        print(f"5GHz {IF5}: " + (f"ch{f['channel']} {f['width']}MHz · 점유율 {f['occupancy']}% · 기기 {f['clients']}대" if f else "없음"))
        print(f"2.4GHz {IF24} 기기별 airtime ({a.gap:.0f}초):")
        for s in p["stations_24"]:
            print(f"  {s['mac']}  {s['airtime']:5.1f}%  신호 {s['signal']}dBm  {s['tx_bitrate']}Mbps")
        r = p["recommendation"]
        print(f"→ 5GHz로 옮길 후보: {r['mac']} (airtime {r['airtime']}%)" if r else f"→ 후보 없음: {p['why_none']}")
    mac = a.mac or (p["recommendation"] or {}).get("mac")
    res = steer(mac) if a.apply and mac else None
    if a.json:
        print(json.dumps({**p, "result": res}, ensure_ascii=False)); return
    if res:
        for l in res["log"]: print("   " + l)
        print(f"→ {'5GHz로 옮김' if res['moved'] else '5GHz에서 안 보임'} ({res['method']}, {res['seconds']}초)")


if __name__ == "__main__":
    sys.exit(main())
