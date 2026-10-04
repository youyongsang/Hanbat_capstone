"""채널 전환 판단 + 다운링크 우세(판정 불확실) 표시 — 라이브 추론(live_congestion.py)·데모가 공유 (2026-10-01).

모델의 매 폴링 원시 예측(0 정상 / 1 경고 / 2 혼잡 / 3 심각)과 feature 원시값을 받아 다음을 낸다.
화면에 보이는 혼잡 단계(--confirm 히스테리시스)와는 **별개**로 돈다 — 표시는 안정적으로, 전환 판단은 빠르게.

1) 전환 판단 (사건 단위)
   - 진입: 최근 `enter_window`(3) 폴링 중 `enter_need`(2) 번 이상 심각 → "전환 필요" 알림 1회 + 명령 후보.
   - 해제: 알림 뒤 `release`(30) 폴링 동안 심각이 한 번도 없어야 다음 알림을 낼 수 있다.
   - 지속: 알림 뒤에도 심각이 이어져 사건이 `persist`(60) 폴링을 넘기면 "심각 지속" 알림 1회
     (실제 전환 뒤라면 "전환 효과 없음" 신호).
   근거: held-out 예측 시뮬레이션(`project/.tmp/exp_24g/switch_entry_rule.py`, h7·h13·h12·8/28·9/1, 1.32시간, 심각 사건 19개)
   EE Fixed 기준 알림 15.5회·헛 알림 2.0·사건 탐지 68%. 같은 조건의 "3중2 + 쿨다운 30"은 알림 31회(긴 사건에서 반복).

2) 다운링크 우세 표시 (판정 불확실)
   - 처리량 ≥ `dl_thr_mbps`(20) 이고 점유율 < `dl_occ_pct`(25) 인 폴링이 최근 5폴링 중 3번 이상이면 켜짐.
   - 이 AP(Opal, siwifi)의 점유율은 AP 자신의 송신을 세지 않아 다운링크 혼잡이 안 보이고, 그 상태에서 모델의
     심각 recall은 0~7%다. 그래서 판정을 고치지 않고 "지금은 판정이 불확실함"만 표시한다.
   근거: 24세션 풀에서 다운링크 부하 행 95.5% 포착, 다른 방향 오표시 0.9%(행 단위).
   주의: 배경 점유율이 높은 장소(학교 idle 약 37%)에서는 다운링크여도 25%를 넘을 수 있다. MT6000(busy에 AP 송신 포함)에서는
   이 표시 대신 송신 시간 계측을 쓴다.

명령 후보는 문자열로만 만든다 — 실제 채널 변경은 하지 않는다(Opal은 고부하에서 불안정).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

CH24 = {1: 2412, 6: 2437, 11: 2462}  # 2.4GHz 겹치지 않는 채널
CH5 = {36: 5180, 44: 5220, 149: 5745, 157: 5785}  # 5GHz 후보(2026-10-05, 학교 스캔에서 한산했던 쪽 + 집 사용 채널)


@dataclass
class SwitchEvent:
    kind: str            # "switch" (전환 필요) | "persist" (심각 지속)
    poll: int            # 몇 번째 폴링에서 났나
    message: str
    candidates: list[str] = field(default_factory=list)


class SwitchAdvisor:
    def __init__(self, enter_need: int = 2, enter_window: int = 3, release: int = 30, persist: int = 60,
                 dl_thr_mbps: float = 20.0, dl_occ_pct: float = 25.0, dl_need: int = 3, dl_window: int = 5,
                 current_channel: int | None = None, iface: str = "wlan0", band: str = "24g"):
        self.enter_need, self.release, self.persist = enter_need, release, persist
        self.recent = deque(maxlen=enter_window)
        self.dl_thr, self.dl_occ, self.dl_need = dl_thr_mbps, dl_occ_pct, dl_need
        self.dl_recent = deque(maxlen=dl_window)
        self.band = band
        self.current_channel = current_channel if current_channel is not None else (44 if band == "5g" else 1)
        self.iface = iface
        self.poll = 0
        self.active = False          # 전환 알림을 낸 사건이 진행 중인가
        self.quiet = 0               # 알림 뒤 심각 없이 지난 폴링 수
        self.event_start = None
        self.persist_sent = False

    # ------------------------------------------------------------
    def candidates(self) -> list[str]:
        out = []
        if self.band == "5g":   # 5GHz: 다른 5GHz 채널로 이동 (2.4GHz로 내려가는 건 후보에서 뺌)
            for ch, freq in CH5.items():
                if ch != self.current_channel:
                    out.append(f"5GHz ch{self.current_channel} → ch{ch}  (예: hostapd_cli -i {self.iface} chan_switch 5 {freq}, 미실행)")
            return out
        for ch, freq in CH24.items():
            if ch != self.current_channel:
                out.append(f"2.4GHz ch{self.current_channel} → ch{ch}  (예: hostapd_cli -i {self.iface} chan_switch 5 {freq}, 미실행)")
        out.append("5GHz 대역으로 단말 유도(밴드 스티어링)  (미실행)")
        return out

    def downlink_dominant(self) -> bool:
        return sum(self.dl_recent) >= self.dl_need

    def update(self, raw_label: int, throughput_mbps: float | None = None,
               occupancy_pct: float | None = None) -> list[SwitchEvent]:
        """폴링 1회 처리. 이번 폴링에 새로 난 알림 목록을 돌려준다(대개 빈 목록)."""
        self.poll += 1
        sev = raw_label == 3
        self.recent.append(sev)
        if throughput_mbps is not None and occupancy_pct is not None:
            self.dl_recent.append(throughput_mbps >= self.dl_thr and occupancy_pct < self.dl_occ)

        events: list[SwitchEvent] = []
        if self.active:
            self.quiet = 0 if sev else self.quiet + 1
            if (not self.persist_sent and sev and self.event_start is not None
                    and self.poll - self.event_start >= self.persist):
                self.persist_sent = True
                events.append(SwitchEvent("persist", self.poll,
                                          f"심각 지속 {self.poll - self.event_start}폴링 — 전환했다면 효과 없음, 다른 후보 검토"))
            if self.quiet >= self.release:
                self.active = False
        elif len(self.recent) >= 2 and sum(self.recent) >= self.enter_need:
            self.active, self.quiet, self.event_start, self.persist_sent = True, 0, self.poll, False
            events.append(SwitchEvent("switch", self.poll,
                                      f"채널 전환 필요 (최근 {len(self.recent)}폴링 중 {sum(self.recent)}번 심각)",
                                      self.candidates()))
        return events
