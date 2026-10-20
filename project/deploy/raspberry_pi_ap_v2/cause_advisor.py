"""혼잡 원인 3분류 + 전환 명령 후보 생성 연결 — live_congestion.py·데모가 공유 (2026-10-20, MT6000 전용).

캡스톤 목표 문장 "혼잡 판단 → 채널 전환 필요 여부 판단 → 전환 명령 후보 생성" 중 뒤 두 단계를 라이브에 붙인다.
모델(LSTM)은 "얼마나 혼잡한가"만 말한다. 이 모듈은 "왜 혼잡한가"를 AP 계측으로 가르고, 원인에 맞는 조치를 낸다.

1) 원인 3분류 (score_cause3.py · analyze_csw.py와 같은 고정 규칙, 10-19 17:37 고정 · 10-20 m28~m33 6/6 일치)
   우선순위: ① 약한 기기 → ② 외부 몫 > 우리 몫 → 이웃 간섭 → ③ 그 외 우리 부하.
   - 우리 몫 = survey "channel transmit time"(AP 송신) + "channel BSS receive time"(우리 기기에서 받음), active 대비 %.
   - 외부 몫 = 점유율(busy) − 우리 몫 (이웃 AP 수신 + 해독 못한 busy). 둘 다 최근 5폴링 중앙값.
   - 약한 기기 = 송신 airtime ≥ 20%인 기기 중 (보낸 패킷 ≥ 20 & tx bitrate ≤ 50Mbps)
                 또는 (재전송 비율 최대 기기 & 그 최대값의 최근 5폴링 중앙값 ≥ 0.35).
   Opal(siwifi)은 BSS 수신 시간을 안 내서 원인 구분 불가 → available=False.
2) 조치
   - 약한 기기: 그 기기 위치 이동·로밍 확인 (채널을 옮겨도 그대로).
   - 우리 부하: 5GHz 유도·AP 분산 (실측 m29~m31: 채널 전환으로 점유율 −1~−7pt뿐).
   - 이웃 간섭: 채널 전환. 모델 전환 알림(SwitchAdvisor, 3폴링 중 2번 심각)이 났을 때만 channel_candidates.py를
     백그라운드로 돌려 후보를 실측 확인(--verify 2, 약 90초)한다. 스캔·CSA 실측은 AP가 잠깐 채널을 비우므로
     쿨다운(기본 10분) 안에는 다시 돌리지 않는다. 실제 전환은 apply=True일 때만(기본은 명령 후보만 표시).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from statistics import median

WEAK, EXT, OWN = "약한 기기", "외부", "우리 부하"
ACTION = {
    WEAK: "약한 기기 {sta} — 위치 이동·로밍 확인 (채널을 옮겨도 그대로)",
    EXT: "이웃 간섭 — 다른 채널로 전환",
    OWN: "우리 기기 부하 — 5GHz 유도·AP 분산 (채널 전환 효과 작음)",
}


def _pct(prev, cur, prev_active, cur_active):
    """survey 누적 카운터(ms) 차분 → active 대비 %. 첫 폴링·카운터 리셋·채널 이동(카운터가 채널별)이면 None."""
    if None in (prev, cur, prev_active, cur_active) or cur_active - prev_active <= 0 or cur - prev < 0:
        return None
    return (cur - prev) / (cur_active - prev_active) * 100.0


def stations_from_counters(stations, prev_stations, interval_s):
    """iw station dump 파싱 결과 두 폴링 → 기기별 (tag, Δ보낸 패킷, tx bitrate, 송신 airtime %, 재전송 비율|None)."""
    out = []
    for mac, s in stations.items():
        p = (prev_stations or {}).get(mac)
        if p is None:
            continue
        dp = s.get("tx_packets", 0) - p.get("tx_packets", 0)
        dr = s.get("tx_retries", 0) - p.get("tx_retries", 0)
        df = s.get("tx_failed", 0) - p.get("tx_failed", 0)
        if dp < 0 or dr < 0 or df < 0:
            continue
        air = 0.0
        if s.get("tx_duration_us") is not None and p.get("tx_duration_us") is not None and interval_s:
            dtx = s["tx_duration_us"] - p["tx_duration_us"]
            air = dtx / 1e6 / interval_s * 100 if dtx >= 0 else 0.0
        ratio = (dr + df) / (dr + df + dp) if dp >= 20 else None
        out.append((mac[-5:].replace(":", ""), dp, float(s.get("tx_bitrate") or 0.0), air, ratio))
    return out


def stations_from_csv(retry_detail, airtime_detail, interval_s):
    """수집 CSV의 sta_retry_detail("tag:Δpkts:Δretries:Δfailed:bitrate|…") · sta_airtime_detail("tag:Δtx_us:Δrx_us:w|…")
    → stations_from_counters와 같은 형식 (재생 모드용)."""
    def parse(s):
        return {p.split(":")[0]: p.split(":")[1:] for p in s.split("|") if p} if isinstance(s, str) and s else {}
    R, A = parse(retry_detail), parse(airtime_detail)
    out = []
    for tag, v in R.items():
        dp, dr, df, br = int(v[0]), int(v[1]), int(v[2]), float(v[3])
        air = float(A[tag][0]) / 1e6 / interval_s * 100 if tag in A and interval_s else 0.0
        out.append((tag, dp, br, air, (dr + df) / (dr + df + dp) if dp >= 20 else None))
    return out


class CauseAdvisor:
    def __init__(self, window: int = 5, weak_bitrate: float = 50.0, weak_retry: float = 0.35,
                 weak_airtime: float = 20.0, min_pkts: int = 20):
        self.window, self.weak_bitrate, self.weak_retry = window, weak_bitrate, weak_retry
        self.weak_airtime, self.min_pkts = weak_airtime, min_pkts
        self.reset()

    def reset(self) -> None:
        """채널을 옮긴 뒤 등 — 이력 비우고 다시 쌓는다."""
        self.own_h: deque = deque(maxlen=self.window)
        self.ext_h: deque = deque(maxlen=self.window)
        self.retry_h: deque = deque(maxlen=self.window)
        self.prev_survey = None

    # ------------------------------------------------------------ 라이브: survey 누적 카운터
    def update_counters(self, survey, stations, prev_stations, interval_s) -> dict:
        """survey = parse_ap_cycle()의 (active, busy, rx, tx, noise, bss_rx) 누적 ms."""
        active, busy, _rx, tx, _noise, bss_rx = survey
        if tx is None or bss_rx is None:
            return {"available": False}
        p, self.prev_survey = self.prev_survey, survey
        if p is None:
            return {"available": True, "ready": False}
        occ, txp, bssp = _pct(p[1], busy, p[0], active), _pct(p[3], tx, p[0], active), _pct(p[5], bss_rx, p[0], active)
        if None in (occ, txp, bssp):
            return {"available": True, "ready": False}
        return self.update_pct(occ, txp, bssp, stations_from_counters(stations, prev_stations, interval_s))

    # ------------------------------------------------------------ 공통: 폴링 % (재생 모드는 CSV 열을 바로 넣음)
    def update_pct(self, occ_pct: float, tx_pct: float, bss_rx_pct: float, stas: list) -> dict:
        own = tx_pct + bss_rx_pct
        self.own_h.append(own)
        self.ext_h.append(occ_pct - own)
        ratios = [(r, tag) for tag, _dp, _br, _air, r in stas if r is not None]
        worst_r, worst_tag = max(ratios) if ratios else (None, None)
        self.retry_h.append(worst_r)
        rr = [r for r in self.retry_h if r is not None]
        rtm = median(rr) if rr else None
        weak = None
        for tag, dp, br, air, _r in stas:
            if air < self.weak_airtime:
                continue
            if (dp >= self.min_pkts and br <= self.weak_bitrate) or (tag == worst_tag and rtm is not None and rtm >= self.weak_retry):
                weak = tag
                break
        own_m, ext_m = median(self.own_h), median(self.ext_h)
        cause = WEAK if weak else (EXT if ext_m > own_m else OWN)
        return {"available": True, "ready": True, "cause": cause, "own": round(own_m, 1), "ext": round(ext_m, 1),
                "occ": round(occ_pct, 1), "weak_sta": weak, "action": ACTION[cause].format(sta=weak or "")}


class CandidateRunner:
    """channel_candidates.candidates()를 백그라운드 스레드로 한 번 돌리고 결과를 들고 있는다."""

    def __init__(self, cooldown_s: float = 600, verify: int = 2, margin: float = 15.0, apply: bool = False,
                 on_done=None):
        self.cooldown_s, self.verify, self.margin, self.apply, self.on_done = cooldown_s, verify, margin, apply, on_done
        self.state = "idle"          # idle | running | done | error
        self.started = self.finished = None
        self.result: dict | None = None
        self.error = ""
        self.applied = False
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.state == "running"

    def cooldown_left(self) -> float:
        if self.finished is None:
            return 0.0
        return max(0.0, self.cooldown_s - (time.time() - self.finished))

    def start(self) -> bool:
        with self._lock:
            if self.running or self.cooldown_left() > 0:
                return False
            self.state, self.started, self.result, self.error, self.applied = "running", time.time(), None, "", False
        threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self) -> None:
        try:
            try:
                import channel_candidates as cc
            except ImportError:
                from scripts import channel_candidates as cc
            r = cc.candidates(scans=2, gap=15, margin=self.margin, verify=self.verify)
            rec = r.get("recommendation")
            if self.apply and rec and rec["switch"]:
                cc.csa(rec["target"])
                self.applied = True
            self.result, self.state = r, "done"
        except Exception as e:  # noqa: BLE001 — SSH 끊김·파싱 실패 등은 화면에 표시하고 루프는 계속
            self.error, self.state = f"{type(e).__name__}: {e}", "error"
        self.finished = time.time()
        if self.on_done:
            self.on_done(self)

    def summary(self) -> dict:
        """화면·JSON용 요약."""
        out = {"state": self.state, "cooldown_left_s": round(self.cooldown_left())}
        if self.running:
            out["elapsed_s"] = round(time.time() - self.started)
        if self.state == "error":
            out["error"] = self.error
        if self.state == "done" and self.result:
            rec, air = self.result.get("recommendation"), self.result.get("airtime") or {}
            cur = self.result.get("current") or {}
            out.update({"current_channel": cur.get("channel"), "current_ext": round(air.get("ext", 0), 1),
                        "candidates": [{"name": c["name"], "scan_busy": None if c["primary_busy"] is None else round(c["primary_busy"], 1),
                                        "verified_ext": round(c["verified_ext"], 1) if "verified_ext" in c else None,
                                        "current": c["current"]} for c in self.result["candidates"]],
                        "applied": self.applied})
            if rec:
                out.update({"switch": rec["switch"], "target": rec["target"], "drop_pt": rec["expected_ext_drop_pt"],
                            "verified": rec["verified"], "command": f"ubus call hostapd.<iface> switch_chan '{_json(rec['ubus'])}'"})
        return out

    def text(self) -> str:
        s = self.summary()
        if s["state"] == "running":
            return f"전환 후보 실측 중 ({s['elapsed_s']}초 경과, 약 90초 — 후보 채널을 잠깐 다녀옴)"
        if s["state"] == "error":
            return f"전환 후보 생성 실패: {s['error']}"
        if s["state"] != "done" or "target" not in s:
            return ""
        how = "실측" if s["verified"] else "스캔 예상"
        if s["switch"]:
            done = " — 적용함" if s["applied"] else " (미실행)"
            return f"전환 명령 후보: ch{s['current_channel']} → {s['target']} (외부 몫 {s['current_ext']:.0f}% → {how} −{s['drop_pt']}pt){done}"
        return f"전환 이득 부족 — 유지 (최선 {s['target']} {how} −{s['drop_pt']}pt < {self.margin:.0f}pt)"


def _json(o) -> str:
    import json
    return json.dumps(o)
