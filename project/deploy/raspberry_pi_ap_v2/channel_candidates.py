"""채널 전환 명령 후보 생성 (2026-10-20, MT6000 순정 OpenWrt 2.4GHz 40MHz).

캡스톤 목표 문장 "혼잡 판단 → 채널 전환 필요 여부 판단 → 전환 명령 후보 생성"의 마지막 단계.
원인 규칙(mt6000_results.html 10절)이 "이웃 간섭 → 채널 전환"이라고 판단했을 때만 호출한다.

측정 (AP에 SSH, 노트북·Pi 어디서든):
  1) 지금 채널: survey 누적 카운터를 10초 간격으로 두 번 읽어 점유율 · 우리 몫(AP 송신 + 우리 BSS 수신) · 외부 몫(나머지).
  2) 다른 채널: `iw scan ap-force`를 SCANS번(기본 2, GAP초 간격). 스캔 한 번에 채널마다 약 73ms 머물며 바쁨을 재고
     (survey dump에 그 스캔 값만 남음, 누적 안 됨), 이웃 BSS 목록(채널·폭·세기)도 얻는다.
     주의: 스캔 동안 AP가 자기 채널을 비워서, 연달아 세 번 스캔하자 노트북 연결이 끊긴 적이 있다(10-20 20:13).
     → 스캔은 판단이 필요할 때만, 간격을 두고 한다.
후보 (40MHz 배치, 1/6/11 계열): 1+ (2402~2442MHz) · 6+ (2427~2467) · 6− (2407~2447) · 11− (2432~2472).
점수 = 후보의 주 채널(20MHz) 바쁨 % (스캔 평균). 우리 기기는 아직 거기 없으므로 전부 외부 몫이다.
     → 지금 채널의 외부 몫과 같은 의미로 비교할 수 있다. 보조 채널 바쁨은 40MHz 유지 가능성 참고값으로만 보고.
확인(--verify, 권장): 스캔 한 번은 73ms 표본이라 같은 채널도 15~41%로 흔들려 비슷한 후보끼리 순위가 뒤집힌다
     (10-20 20:21 보정: 스캔 평균 1+ 52 · 6+ 20 · 11− 32% vs 실제로 옮겨 잰 외부 몫 38 · 29 · 25%).
     → 스캔 상위 VERIFY(기본 2)개 후보로 CSA(연결 유지) 전환해 DWELL초(기본 20) 실측한 뒤 원래 채널로 돌아온다.
     같은 채널 반복 실측은 0.4~5pt 안에서 일치했다. 최종 순위·권고는 실측값으로 낸다.
권고: 최선 후보의 바쁨이 지금 외부 몫보다 MARGIN(기본 15)pt 이상 낮을 때만 "전환 명령 후보"로 낸다.
     (우리 몫은 채널을 옮겨도 따라가므로 비교에서 뺀다 — 전환 효과 실측(m28~m33)과 같은 논리)

사용:
  python project/scripts/channel_candidates.py                 # 사람이 읽는 표
  python project/scripts/channel_candidates.py --json          # 기계용 (ubus switch_chan 인자 포함)
"""
import argparse, json, os, re, subprocess, sys, time

# collect_metrics.py와 같은 환경변수 (live_congestion.py --mt6000 이 phy0-ap0으로 맞춤)
AP = f"root@{os.environ.get('AP_IP', '192.168.8.1')}"
IFACE = os.environ.get("AP_INTERFACE", "phy0-ap0")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6"]
# (이름, 주 채널, 주파수, 보조 방향 +1/-1)
PLACEMENTS = [("1+", 1, 2412, 1), ("6+", 6, 2437, 1), ("6-", 6, 2437, -1), ("11-", 11, 2462, -1)]


def sh(cmd, timeout=40):
    r = subprocess.run(SSH + [AP, cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout


def parse_survey(text):
    out, cur = {}, None
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"frequency:\s+(\d+) MHz(.*)", line)
        if m:
            cur = int(m.group(1)); out[cur] = {"in_use": "in use" in m.group(2)}; continue
        if cur is None: continue
        for key, pat in (("active", "channel active time"), ("busy", "channel busy time"), ("bss_rx", "channel BSS receive time"),
                         ("tx", "channel transmit time"), ("rx", "channel receive time")):
            if line.startswith(pat + ":"):
                out[cur][key] = float(re.search(r"(\d+) ms", line).group(1))
    return out


def parse_scan(text):
    bss, cur = [], None
    for line in text.splitlines():
        if line.startswith("BSS "):
            cur = {"bssid": line.split()[1][:17], "freq": None, "signal": None, "ssid": "", "sec": 0}; bss.append(cur); continue
        if cur is None: continue
        s = line.strip()
        if s.startswith("freq:"): cur["freq"] = int(float(s.split()[1]))
        elif s.startswith("signal:"): cur["signal"] = float(s.split()[1])
        elif s.startswith("SSID:"): cur["ssid"] = s[5:].strip()
        elif "secondary channel offset: above" in s: cur["sec"] = 1
        elif "secondary channel offset: below" in s: cur["sec"] = -1
    return [b for b in bss if b["freq"] and b["freq"] < 2500 and b["signal"] is not None]


def current_channel():
    t = sh(f"iw dev {IFACE} info")
    m = re.search(r"channel (\d+) \((\d+) MHz\), width: (\d+) MHz, center1: (\d+)", t)
    ch, f, w, c1 = map(int, m.groups())
    return {"channel": ch, "freq": f, "width": w, "center1": c1}


def current_airtime(gap=10):
    a = parse_survey(sh(f"iw dev {IFACE} survey dump")); time.sleep(gap); b = parse_survey(sh(f"iw dev {IFACE} survey dump"))
    f = next(k for k, v in b.items() if v["in_use"]); d = {k: b[f][k] - a[f][k] for k in ("active", "busy", "bss_rx", "tx")}
    occ = d["busy"] / d["active"] * 100; own = (d["bss_rx"] + d["tx"]) / d["active"] * 100
    return {"occupancy": occ, "own": own, "ext": occ - own}


def scan_once():
    scan = sh(f"iw dev {IFACE} scan ap-force", timeout=30)
    sv = parse_survey(sh(f"iw dev {IFACE} survey dump"))
    busy = {f: v["busy"] / v["active"] * 100 for f, v in sv.items() if not v["in_use"] and v.get("active")}
    return busy, parse_scan(scan)


def overlaps(f_a, f_b):  # 20MHz 채널 두 개가 주파수로 겹치나
    return abs(f_a - f_b) < 20


def csa(name):
    p = next(x for x in PLACEMENTS if x[0] == name); f, sec = p[2], p[3]
    j = json.dumps({"freq": f, "bcn_count": 10, "sec_channel_offset": sec, "center_freq1": f + 10 * sec, "bandwidth": 40, "ht": True, "he": True})
    sh(f"ubus call hostapd.{IFACE} switch_chan '{j}'"); time.sleep(6)


def candidates(scans=2, gap=15, margin=15.0, skip_current_airtime=False, verify=0, dwell=20):
    cur = current_channel()
    air = None if skip_current_airtime else current_airtime()
    busy_runs, bss_all = [], {}
    for i in range(scans):
        if i: time.sleep(gap)
        b, bss = scan_once(); busy_runs.append(b)
        for x in bss: bss_all[x["bssid"]] = x
    rows = []
    for name, ch, f, sec in PLACEMENTS:
        f2 = f + 20 * sec
        pb = [r[f] for r in busy_runs if f in r]; sb = [r[f2] for r in busy_runs if f2 in r]
        is_cur = (f == cur["freq"] and cur["center1"] == f + 10 * sec)
        nb = [x for x in bss_all.values() if overlaps(x["freq"], f)]
        strongest = max(nb, key=lambda x: x["signal"]) if nb else None
        rows.append({"name": name, "channel": ch, "freq": f, "sec_offset": sec, "center1": f + 10 * sec, "current": is_cur,
                     "primary_busy": sum(pb) / len(pb) if pb else None, "secondary_busy": sum(sb) / len(sb) if sb else None,
                     "neighbors_on_primary": len([x for x in nb if x["signal"] >= -82]),
                     "strongest": f"{strongest['ssid'] or '?'} {strongest['signal']:.0f}dBm" if strongest else "-"})
    others = [r for r in rows if not r["current"] and r["primary_busy"] is not None
              and not (r["freq"] == cur["freq"])]  # 같은 주 채널(보조만 다른 배치)은 이웃 간섭을 못 피함
    others.sort(key=lambda r: (r["primary_busy"], -(r["neighbors_on_primary"] == 0)))
    if verify and others:   # 상위 후보에 잠깐 옮겨 가서 실측 (CSA, 연결 유지) 후 원래 배치로 복귀
        home = next(r["name"] for r in rows if r["current"])
        for r in others[:verify]:
            csa(r["name"]); m = current_airtime(dwell); r["verified_ext"] = m["ext"]
        csa(home)
        others = sorted(others[:verify], key=lambda r: r["verified_ext"]) + others[verify:]
    best = others[0] if others else None
    rec = None
    if best and air:
        gain = air["ext"] - best.get("verified_ext", best["primary_busy"])
        rec = {"switch": gain >= margin, "target": best["name"], "expected_ext_drop_pt": round(gain, 1),
               "verified": "verified_ext" in best, "ubus": {"freq": best["freq"], "bcn_count": 10, "sec_channel_offset": best["sec_offset"],
                        "center_freq1": best["center1"], "bandwidth": 40, "ht": True, "he": True}}
    return {"current": cur, "airtime": air, "candidates": rows, "recommendation": rec, "scans": scans}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", action="store_true")
    ap.add_argument("--scans", type=int, default=2); ap.add_argument("--gap", type=float, default=15)
    ap.add_argument("--margin", type=float, default=15.0)
    ap.add_argument("--verify", type=int, default=0, help="스캔 상위 N개 후보에 잠깐 옮겨 가서 실측 (권장 2)")
    ap.add_argument("--dwell", type=float, default=20); a = ap.parse_args()
    r = candidates(a.scans, a.gap, a.margin, verify=a.verify, dwell=a.dwell)
    if a.json:
        print(json.dumps(r, ensure_ascii=False)); return
    c, air = r["current"], r["airtime"]
    print(f"지금 ch{c['channel']} ({c['width']}MHz, center {c['center1']}) — 점유율 {air['occupancy']:.0f}% = 우리 {air['own']:.0f}% + 외부 {air['ext']:.0f}%")
    print(f"후보 (스캔 {r['scans']}회 평균, 주 채널 바쁨 = 옮겼을 때 예상 외부 몫)")
    for x in r["candidates"]:
        pb = "-" if x["primary_busy"] is None else f"{x['primary_busy']:.0f}%"; sb = "-" if x["secondary_busy"] is None else f"{x['secondary_busy']:.0f}%"
        print(f"  {x['name']:4s} 주 채널 바쁨 {pb:>4s} · 보조 {sb:>4s} · 주 채널 이웃 {x['neighbors_on_primary']}개 (가장 센 것 {x['strongest']}){f" · 실측 외부 {x['verified_ext']:.0f}%" if 'verified_ext' in x else ''}{'  ← 지금' if x['current'] else ''}")
    rec = r["recommendation"]
    if rec:
        verdict = "전환 명령 후보" if rec["switch"] else f"전환 이득 부족(< {a.margin:.0f}pt) — 유지"
        print(f"→ {verdict}: {rec['target']} (외부 몫 {'실측' if rec['verified'] else '스캔 예상'} −{rec['expected_ext_drop_pt']}pt)")
        print(f"   ubus call hostapd.{IFACE} switch_chan '{json.dumps(rec['ubus'])}'")


if __name__ == "__main__":
    sys.exit(main())
