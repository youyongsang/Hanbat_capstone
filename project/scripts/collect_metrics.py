#!/usr/bin/env python3
import csv
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from statistics import median
from collections import deque

IS_WINDOWS = platform.system() == "Windows"

# ============================================================
# 기본 설정
# ============================================================

AP_IP = "192.168.8.1"
# latency 측정 대상. 2026-08-27: 폰(191)은 ICMP 절전/디프라이어리티제이션
# 때문에 idle RTT가 31~295ms로 요동쳐서 latency 축이 못 쓸 수준이었다.
# 노트북은 ICMP에 즉시·일정하게 응답한다. Pi→AP(유선)→노트북(무선
# downlink) 경로라 혼잡 시 RTT가 오르는 신호는 그대로 유지된다.
SERVER_IP = "192.168.8.226"
INTERFACE = "wlan0"

CSV_FILE = "metrics_v2.csv"
IPERF_JSON_FILE = "iperf3_result.json"

MOVING_AVG_WINDOW = 5

# ============================================================
# 혼잡도 점수 기준
#
# 아래 4개 score는 0~1 범위로 변환된다.
# 실험 부하가 0/20/50/100/150 Mbps인 현재 실험을 기준으로
# throughput은 150 Mbps를 심각한 부하 기준점으로 사용한다.
#
# 필요하면 실제 데이터 확인 후 이 값만 조정하면 된다.
# ============================================================

# ============================================================
# 혼잡 라벨 재설계 (2026-08-27, docs/yongsang/congestion_label_redesign.md)
#
# 각 축을 외부 표준 문턱에 매핑(4 앵커: 경고→0.25 / 혼잡→0.5 /
# 심각→0.75 / 완전→1.0, piecewise-linear, [0,1] clamp).
# congestion_score = max(5개 축). "label 3 = 최소 한 축이 표준
# 심각 문턱 돌파". 가중치 없음.
#
# jitter/loss는 victim 프로브(파이가 쏘는 경량 UDP 스트림)의 실측,
# latency는 ping RTT. 모델 입력에서는 이 세 개를 뺀다(배포 시 없음).
# ============================================================

# (축, 경고, 혼잡, 심각, 완전) — 단위는 각 축 주석 참고
ANCHORS = {
    # channel airtime %  — Cisco/Aruba WLAN 설계 가이드 (>50% 경고, >75% 혼잡)
    "occupancy": (40.0, 55.0, 75.0, 90.0),
    # probe IPDV ms      — ITU-T Y.1541 Class 0/1 (≤50ms), RFC 4594 (~30ms)
    "jitter": (20.0, 30.0, 50.0, 100.0),
    # probe loss %       — Cisco Enterprise QoS (voice <1%, >5% 불가), ITU-T G.113
    "loss": (0.5, 1.0, 5.0, 10.0),
    # 편도(one-way) ms   — ITU-T G.114. ping은 RTT를 재므로 calculate_scores에서
    #                      latency_ms/2 를 편도 추정치로 넣는다(2026-08-27 밤:
    #                      RTT 생값을 편도 앵커에 넣어 label 3 과다 발생 → 수정).
    #                      노트북 대상(방화벽 ICMP 허용). idle RTT ~2ms.
    "latency": (30.0, 60.0, 150.0, 400.0),
    # retry ratio %      — WLAN 헬스 (Cisco/Ekahau/7signal: <10% 정상, >20% 불량)
    "retry": (10.0, 15.0, 25.0, 40.0),
}

# throughput_score는 라벨 max에서 제외(label 2/3 변별력 없음).
# CSV에 정보용으로만 남기며 이 상한으로 0~1 스케일.
THROUGHPUT_MAX_MBPS = 150.0

# ============================================================
# victim 프로브 (파이가 노트북에 쏘는 경량 UDP 스트림)
# ============================================================

PROBE_TARGET = "192.168.8.226"   # 노트북(무선). 파이→AP(유선)→노트북(무선 downlink)
PROBE_PORT = 5203
PROBE_RATE = "300k"
PROBE_LEN = 200                  # bytes. 300k/200B ≈ 187pps ≈ VoIP급
PROBE_TEST_S = 2                 # 짧은 테스트를 백그라운드 스레드가 연속 실행
PROBE_STALE_S = 12              # 결과가 이보다 오래되면 축을 못 쓰는 것으로 간주

# ============================================================
# SSH
# ============================================================

SSH_CMD = [
    "ssh",
    "-o", "HostKeyAlgorithms=+ssh-rsa",
    "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
    "-o", "ConnectTimeout=5",
    # 지속 연결이 끊겼는지 빨리 감지하기 위한 옵션(폴러 전용).
    "-o", "ServerAliveInterval=5",
    "-o", "ServerAliveCountMax=3",
    "-o", "BatchMode=yes",
    f"root@{AP_IP}",
]

# ============================================================
# 최종 CSV 컬럼
#
# 피드백에서 변화가 미미하다고 한 8개 컬럼은 제외한다.
# ============================================================

CSV_COLUMNS = [
    "timestamp",
    "scenario",
    "poll_interval_s",
    "throughput_mbps",
    "channel_occupancy_percent",
    "channel_occupancy_method",
    "latency_ms",                # ping RTT avg (latency 축)
    "probe_jitter_ms",           # victim 프로브 (jitter 축)
    "probe_loss_pct",            # victim 프로브 (loss 축)
    "probe_ok",                  # 프로브 결과가 신선한가 (1/0)
    "packet_loss_udp_percent",   # 부하 iperf3 (레거시, 모델 제외)
    "tx_retx_delta",             # tx retries + tx failed (지난 폴링 이후)
    "tx_packets_delta",          # 지난 폴링 이후 성공 전송 프레임
    "tx_retry_ratio",            # retx / (retx + packets)  (retry 축 + 모델 입력)
    "rssi_dbm",
    "connected_clients",
    "rssi_delta_db",
    "rssi_moving_avg_dbm",
    "sta_tx_bitrate_min",        # 가장 굶는 station PHY rate (모델 입력, victim 프록시)
    "sta_tx_bitrate_mean",       # station 평균 PHY rate (모델 입력)
    "throughput_score",          # 정보용 (라벨 max에서 제외)
    "occupancy_score",
    "jitter_score",
    "loss_score",
    "latency_score",
    "retry_score",
    "congestion_score",
    "label",
]

# ============================================================
# 명령 실행
# ============================================================

def run_command(command, timeout=10):
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout
    except Exception:
        return ""


# ============================================================
# AP station + survey — 지속 SSH 세션 하나로 폴링
#
# 기존에는 루프마다(약 0.5~1초 간격) 새 SSH 프로세스를 띄워
# TCP+SSH 핸드셰이크를 반복했다. 이 관리 트래픽이 측정 대상과
# 같은 무선 채널을 그대로 타고 가서, 채널이 혼잡할수록 SSH
# 핸드셰이크 자체도 같이 느려지는 자기참조적 지연(폴링 지연
# 스파이크 최대 156초, 2026-08-26 새벽에는 완전 크래시까지
# 재현됨 — docs/yongsang/ap_crash_analysis.md 참고)의 유력한
# 원인이었다. 대신 원격에서 무한루프를 SSH 세션 하나로 계속
# 돌리고 표준출력을 스트리밍으로 읽어, 핸드셰이크를 세션당
# 1회로 줄인다. 연결이 끊기면 백그라운드 스레드가 자동으로
# 재연결을 시도한다.
# ============================================================

class APPoller:
    STATION_BEGIN = "__STATION_BEGIN__"
    SURVEY_BEGIN = "__SURVEY_BEGIN__"
    CYCLE_END = "__CYCLE_END__"

    REMOTE_LOOP_CMD = (
        f"while true; do "
        f"echo '{STATION_BEGIN}'; "
        f"iw dev {INTERFACE} station dump; "
        f"echo '{SURVEY_BEGIN}'; "
        f"iw dev {INTERFACE} survey dump; "
        f"echo '{CYCLE_END}'; "
        f"sleep 0.5; "
        f"done"
    )

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None  # (cycle_id, text)
        self._next_id = 0
        self._stopping = False
        self._proc = None
        self.reconnects = 0

        self._thread = threading.Thread(
            target=self._run, daemon=True
        )
        self._thread.start()

    def _run(self):
        first_attempt = True

        while not self._stopping:
            if not first_attempt:
                self.reconnects += 1
                time.sleep(2)
            first_attempt = False

            try:
                self._proc = subprocess.Popen(
                    SSH_CMD + [self.REMOTE_LOOP_CMD],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
            except Exception:
                continue

            buf = []

            try:
                for line in self._proc.stdout:
                    line = line.rstrip("\n")

                    if line == self.CYCLE_END:
                        text = "\n".join(buf)
                        buf = []

                        with self._lock:
                            self._next_id += 1
                            self._latest = (
                                self._next_id,
                                text,
                            )

                        continue

                    buf.append(line)
            except Exception:
                pass

            try:
                self._proc.terminate()
            except Exception:
                pass

    def wait_for_new_cycle(
        self, last_id, poll_interval=0.2, max_wait=2.0
    ):
        start = time.time()

        while True:
            with self._lock:
                if (
                    self._latest is not None
                    and self._latest[0] != last_id
                ):
                    return self._latest

            if time.time() - start > max_wait:
                return None

            time.sleep(poll_interval)

    def stop(self):
        self._stopping = True

        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass


class ProbeRunner:
    """victim 프로브: 노트북에 경량 UDP 스트림을 짧게, 연속으로 쏘고
    각 테스트의 서버 측정 jitter/loss를 캐시한다(APPoller와 같은 패턴).

    한 번의 폴링과 결합하지 않고 독립 스레드로 돌아서, 메인 루프는
    그냥 최신 캐시(get())를 읽는다. iperf3 3.x 어느 버전에서든 동작하도록
    지속 스트림 파싱 대신 짧은 -t 테스트를 반복한다.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None  # (jitter_ms, loss_pct, wall_ts)
        self._stopping = False
        self._proc = None
        self.runs = 0
        self.failures = 0

        self._thread = threading.Thread(
            target=self._run, daemon=True
        )
        self._thread.start()

    def _run(self):
        cmd = [
            "iperf3", "-u",
            "-c", PROBE_TARGET,
            "-p", str(PROBE_PORT),
            "-b", PROBE_RATE,
            "-l", str(PROBE_LEN),
            "-t", str(PROBE_TEST_S),
            "-J",
        ]

        while not self._stopping:
            self.runs += 1
            jitter = None
            loss = None

            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                out, _ = self._proc.communicate(
                    timeout=PROBE_TEST_S + 6
                )
                data = json.loads(out)
                summary = data.get("end", {}).get("sum", {})
                if "jitter_ms" in summary:
                    jitter = round(float(summary["jitter_ms"]), 3)
                if "lost_percent" in summary:
                    loss = round(float(summary["lost_percent"]), 3)
            except Exception:
                self.failures += 1
                try:
                    if self._proc is not None:
                        self._proc.kill()
                except Exception:
                    pass

            if jitter is not None or loss is not None:
                with self._lock:
                    self._latest = (jitter, loss, time.time())
            else:
                time.sleep(2)

    def get(self):
        """(jitter_ms, loss_pct, fresh, ever_ok).

        fresh=False면 결과가 stale(프로브 테스트가 연속 실패 중이거나 아직
        한 번도 못 돎). ever_ok=True면 이 세션에서 프로브가 최소 한 번은
        정상 완료한 적이 있음 — "되다가 죽음"과 "처음부터 안 됨(서버 미기동
        등)"을 구분하는 데 쓴다."""
        with self._lock:
            latest = self._latest

        if latest is None:
            return None, None, False, False

        jitter, loss, wall_ts = latest
        fresh = (time.time() - wall_ts) <= PROBE_STALE_S
        return jitter, loss, fresh, True

    def stop(self):
        self._stopping = True
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass


def parse_ap_cycle(text):
    station_marker = APPoller.STATION_BEGIN
    survey_marker = APPoller.SURVEY_BEGIN

    if (
        station_marker not in text
        or survey_marker not in text
    ):
        return None, (None, None)

    station_text = text.split(station_marker, 1)[1]
    station_text, survey_text = station_text.split(
        survey_marker, 1
    )

    station = parse_station_info(station_text)
    active, busy = parse_channel_occupancy(survey_text)

    return station, (active, busy)


# ============================================================
# Station 정보
# ============================================================

def parse_station_info(output):
    # MAC 주소별로 개별 추적한다. 여러 station의 바이트/카운터를
    # 그냥 합산해버리면, 어떤 station이 잠깐 station dump에서
    # 빠졌다가(절전모드 등) 다시 나타나는 순간 그 station의 전체
    # 누적 카운터가 "한 폴링 주기 동안의 증가분"으로 잘못 계산된다.
    stations = {}
    current_mac = None
    current = None

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if line.startswith("Station "):
            if current_mac is not None:
                stations[current_mac] = current

            mac_match = re.match(
                r"Station\s+([0-9A-Fa-f:]{17})",
                line,
            )
            current_mac = (
                mac_match.group(1)
                if mac_match
                else line
            )

            current = {
                "rx_bytes": 0,
                "tx_bytes": 0,
                "tx_retries": 0,
                "tx_failed": 0,
                "tx_packets": 0,
                "signal_avg": None,
                "tx_bitrate": 0.0,
                "rx_bitrate": 0.0,
            }
            continue

        if current is None:
            continue

        if line.startswith("rx bytes:"):
            try:
                current["rx_bytes"] = int(
                    line.split(":", 1)[1].strip()
                )
            except ValueError:
                pass

        elif line.startswith("tx bytes:"):
            try:
                current["tx_bytes"] = int(
                    line.split(":", 1)[1].strip()
                )
            except ValueError:
                pass

        elif line.startswith("tx retries:"):
            try:
                current["tx_retries"] = int(
                    line.split(":", 1)[1].strip()
                )
            except ValueError:
                pass

        elif line.startswith("tx failed:"):
            try:
                current["tx_failed"] = int(
                    line.split(":", 1)[1].strip()
                )
            except ValueError:
                pass

        elif line.startswith("tx packets:"):
            try:
                current["tx_packets"] = int(
                    line.split(":", 1)[1].strip()
                )
            except ValueError:
                pass

        elif line.startswith("signal avg:"):
            match = re.search(
                r"(-?\d+(?:\.\d+)?)",
                line,
            )
            if match:
                current["signal_avg"] = float(
                    match.group(1)
                )

        elif line.startswith("tx bitrate:"):
            match = re.search(
                r"([\d.]+)\s+MBit/s",
                line,
            )
            if match:
                current["tx_bitrate"] = float(
                    match.group(1)
                )

        elif line.startswith("rx bitrate:"):
            match = re.search(
                r"([\d.]+)\s+MBit/s",
                line,
            )
            if match:
                current["rx_bitrate"] = float(
                    match.group(1)
                )

    if current_mac is not None:
        stations[current_mac] = current

    if not stations:
        return None

    return stations


def summarize_stations(stations):
    signal_values = [
        s["signal_avg"]
        for s in stations.values()
        if s["signal_avg"] is not None
    ]

    signal_avg = (
        sum(signal_values) / len(signal_values)
        if signal_values
        else 0.0
    )

    # station별 PHY tx rate (iw station dump의 "tx bitrate").
    # rate control이 간섭·경합에 물러나면 여기부터 떨어진다 — 채널 전체
    # occupancy엔 안 보이는 신호. min = 가장 굶는 station(= victim 프록시),
    # capture effect(강한 폰이 채널 독점, 약한 쪽 저MCS)가 여기 드러난다.
    # 0.0은 "미보고"라 제외.
    tx_rates = [
        s.get("tx_bitrate", 0.0)
        for s in stations.values()
        if s.get("tx_bitrate", 0.0) > 0.0
    ]
    if tx_rates:
        sta_tx_bitrate_min = min(tx_rates)
        sta_tx_bitrate_mean = sum(tx_rates) / len(tx_rates)
    else:
        sta_tx_bitrate_min = 0.0
        sta_tx_bitrate_mean = 0.0

    return (
        signal_avg,
        len(stations),
        sta_tx_bitrate_min,
        sta_tx_bitrate_mean,
    )


# ============================================================
# Channel Occupancy
# ============================================================

def parse_channel_occupancy(output):
    active = None
    busy = None
    in_use = False

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if line.startswith("frequency:"):
            in_use = "[in use]" in line
            continue

        if (
            in_use
            and line.startswith("channel active time:")
        ):
            try:
                active = float(
                    line.split(":", 1)[1]
                    .replace("ms", "")
                    .strip()
                )
            except ValueError:
                pass

        elif (
            in_use
            and line.startswith("channel busy time:")
        ):
            try:
                busy = float(
                    line.split(":", 1)[1]
                    .replace("ms", "")
                    .strip()
                )
            except ValueError:
                pass

    return active, busy


def calculate_channel_occupancy(
    previous_active,
    previous_busy,
    current_active,
    current_busy,
):
    if (
        current_active is None
        or current_busy is None
        or current_active <= 0
    ):
        return 0.0, "unavailable"

    # First sample: no previous counter exists.
    if (
        previous_active is None
        or previous_busy is None
    ):
        occupancy = (
            current_busy / current_active
        ) * 100.0

        return (
            round(
                max(0.0, min(100.0, occupancy)),
                2,
            ),
            "instantaneous",
        )

    active_delta = (
        current_active - previous_active
    )
    busy_delta = (
        current_busy - previous_busy
    )

    # Use the counter delta only when both counters advanced normally.
    # If the driver resets/reports a non-increasing survey counter,
    # use the current survey ratio instead of producing a fake delta.
    if active_delta > 0 and busy_delta >= 0:
        occupancy = (
            busy_delta / active_delta
        ) * 100.0

        return (
            round(
                max(0.0, min(100.0, occupancy)),
                2,
            ),
            "delta",
        )

    occupancy = (
        current_busy / current_active
    ) * 100.0

    return (
        round(
            max(0.0, min(100.0, occupancy)),
            2,
        ),
        "instantaneous_fallback",
    )


# ============================================================
# Ping
# ============================================================

def get_ping_metrics():
    if IS_WINDOWS:
        command = ["ping", "-n", "4", "-w", "1000", SERVER_IP]
    else:
        # 2026-08-27: 파이 유선 수집 도입하면서 폴링 주기를 줄이려고
        # -c 4 -> -c 2 로 축소(핑 4번이 행당 ~3초 병목이었음).
        # jitter는 여전히 mdev를 쓴다(2 샘플이면 |rtt1-rtt2|/2 수준).
        command = ["ping", "-c", "2", "-W", "1", SERVER_IP]

    output = run_command(command, timeout=8)

    latency = 0.0
    jitter = 0.0
    packet_loss = 0.0

    if IS_WINDOWS:
        # Windows ping output is locale-dependent (Korean/English/...),
        # but "TTL=" and the "<N>ms" reply time token are not localized,
        # so parse per-reply lines instead of the summary sentence.
        reply_times = [
            float(value)
            for line in output.splitlines()
            if "TTL=" in line.upper()
            for value in re.findall(r"(\d+(?:\.\d+)?)\s*ms", line)
        ]

        if reply_times:
            latency = sum(reply_times) / len(reply_times)
            jitter = max(reply_times) - min(reply_times)

        # The only "%" in ping output is the packet loss percentage,
        # regardless of display language.
        loss_match = re.search(r"(\d+(?:\.\d+)?)\s*%", output)

        if loss_match:
            packet_loss = float(loss_match.group(1))

        return latency, jitter, packet_loss

    rtt_match = re.search(
        r"=\s*"
        r"([\d.]+)/"
        r"([\d.]+)/"
        r"([\d.]+)/"
        r"([\d.]+)\s*ms",
        output,
    )

    if rtt_match:
        latency = float(rtt_match.group(2))
        jitter = float(rtt_match.group(4))

    loss_match = re.search(
        r"(\d+(?:\.\d+)?)%"
        r"\s*packet loss",
        output,
    )

    if loss_match:
        packet_loss = float(
            loss_match.group(1)
        )

    return latency, jitter, packet_loss


# ============================================================
# iperf3 JSON
#
# 중요:
# collector는 JSON 파일이 실제로 새로 생성/갱신된 경우에만
# UDP loss를 기록한다. 오래된 결과를 반복 복사하지 않는다.
# ============================================================

def get_iperf_json_mtime():
    if not os.path.exists(IPERF_JSON_FILE):
        return None

    try:
        return os.path.getmtime(
            IPERF_JSON_FILE
        )
    except OSError:
        return None


def get_iperf_udp_loss():
    if not os.path.exists(IPERF_JSON_FILE):
        return None

    try:
        with open(
            IPERF_JSON_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        ValueError,
    ):
        return None

    try:
        end = data.get("end", {})

        sum_received = end.get(
            "sum_received",
            {},
        )

        if "lost_percent" in sum_received:
            return float(
                sum_received["lost_percent"]
            )

        sum_data = end.get("sum", {})

        if "lost_percent" in sum_data:
            return float(
                sum_data["lost_percent"]
            )

        streams = end.get("streams", [])

        for stream in streams:
            udp = stream.get("udp", {})

            if "lost_percent" in udp:
                return float(
                    udp["lost_percent"]
                )

    except (
        TypeError,
        ValueError,
        AttributeError,
    ):
        return None

    return None


# ============================================================
# Throughput
# ============================================================

def calculate_station_deltas(
    previous_stations,
    current_stations,
):
    # 이전 폴링에 있던 station과 MAC이 일치하는 경우에만 델타를
    # 누적한다. 새로 나타난 station(최초 연결이든, 절전모드 등으로
    # 빠졌다가 재등장한 것이든)은 기준값을 알 수 없으므로 이번
    # 폴링 한 번은 델타 0으로 건너뛴다. 그래야 재등장 시 station의
    # 전체 누적 카운터가 한 폴링 주기 증가분으로 잘못 계산되는
    # 것을 막을 수 있다.
    # 주의: 이 AP(GL-SFT1200)는 `tx retries` / `tx failed`를 station별이
    # 아니라 라디오 전체 카운터로 보고한다(모든 station이 같은 값). 그래서
    # station마다 더하면 delta가 station 수만큼 뻥튀기된다. retries/failed는
    # station 하나에서만 읽어 단일 delta로 계산하고, 나머지(rx/tx bytes,
    # tx packets)만 station별로 합산한다.
    rx_delta = 0
    tx_delta = 0
    packets_delta = 0

    if not previous_stations:
        return rx_delta, tx_delta, 0, 0, packets_delta

    for mac, current in current_stations.items():
        previous = previous_stations.get(mac)

        if previous is None:
            continue

        rx_delta += max(
            0, current["rx_bytes"] - previous["rx_bytes"]
        )
        tx_delta += max(
            0, current["tx_bytes"] - previous["tx_bytes"]
        )
        packets_delta += max(
            0, current["tx_packets"] - previous["tx_packets"]
        )

    # 라디오 전체 retries/failed: 현재/이전 모두에 존재하는 station 중
    # 최대값으로 단일 delta (모두 같은 값이라 어느 것을 써도 되지만,
    # 재등장 station의 stale 값을 피하려고 max).
    def _radio(stations, key):
        vals = [s[key] for s in stations.values() if s.get(key)]
        return max(vals) if vals else 0

    common = set(previous_stations) & set(current_stations)
    if common:
        cur_retx = _radio(
            {m: current_stations[m] for m in common}, "tx_retries"
        )
        prev_retx = _radio(
            {m: previous_stations[m] for m in common}, "tx_retries"
        )
        cur_fail = _radio(
            {m: current_stations[m] for m in common}, "tx_failed"
        )
        prev_fail = _radio(
            {m: previous_stations[m] for m in common}, "tx_failed"
        )
        retries_delta = max(0, cur_retx - prev_retx)
        failed_delta = max(0, cur_fail - prev_fail)
    else:
        retries_delta = 0
        failed_delta = 0

    return rx_delta, tx_delta, retries_delta, failed_delta, packets_delta


# ============================================================
# Score
# ============================================================

def clamp01(value):
    return max(0.0, min(1.0, value))


def anchor_score(value, anchors):
    """(경고, 혼잡, 심각, 완전) 4앵커에 piecewise-linear 매핑
    → 0 / 0.25 / 0.5 / 0.75 / 1.0. [0,1] clamp.
    value가 None이면 이 축을 못 쓰는 것 → 0.0 (max에 기여 안 함)."""
    if value is None:
        return 0.0

    a_warn, a_cong, a_sev, a_full = anchors
    knots = [
        (0.0, 0.0),
        (a_warn, 0.25),
        (a_cong, 0.5),
        (a_sev, 0.75),
        (a_full, 1.0),
    ]

    if value <= knots[0][0]:
        return 0.0
    if value >= knots[-1][0]:
        return 1.0

    for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
        if value <= x1:
            if x1 == x0:
                return y1
            return clamp01(
                y0 + (y1 - y0) * (value - x0) / (x1 - x0)
            )

    return 1.0


def calculate_scores(
    throughput,
    occupancy,
    latency_ms,
    probe_jitter_ms,
    probe_loss_pct,
    retry_ratio_pct,
    ping_hard_fail=False,
    probe_hard_fail=False,
):
    """혼잡 라벨 재설계(2026-08-27): 각 축을 표준 문턱에 매핑한 뒤
    congestion_score = max(라벨 축).

    라벨 축 = occupancy + jitter(프로브) + loss(프로브) + latency(ping). 4축.

    retry는 라벨 축에서 뺐다(2026-08-27 v4 베이스라인): 이 2.4GHz AP는
    RF가 열악해서 idle에도 retry_ratio가 18~36%인데(자는 폰·백그라운드
    버스트) victim 프로브는 완벽(jitter 1ms, loss 0%). retry는 jitter/loss를
    유발하는 원인이지 QoS 피해의 독립 증거가 아니다. retry_ratio는 모델
    입력 feature로만 유지(모델이 "재전송 많다 → victim 곧 깨질 것" 추론).
    throughput_score도 정보용으로만(label 2/3 변별력 없음).
    """
    throughput_score = clamp01(throughput / THROUGHPUT_MAX_MBPS)

    occupancy_score = anchor_score(occupancy, ANCHORS["occupancy"])
    jitter_score = anchor_score(probe_jitter_ms, ANCHORS["jitter"])
    loss_score = anchor_score(probe_loss_pct, ANCHORS["loss"])
    # ANCHORS["latency"]는 G.114 편도 값. ping은 RTT라 절반을 편도 추정치로.
    latency_oneway = latency_ms / 2.0 if latency_ms else latency_ms
    latency_score = anchor_score(latency_oneway, ANCHORS["latency"])
    retry_score = anchor_score(retry_ratio_pct, ANCHORS["retry"])  # 정보용

    # 실패=max: 채널이 실제로 바쁜데(caller가 판단) victim 경로가 완전히
    # 죽은 경우 — ping이 응답 0 / 프로브 스트림이 완료 불가 — 는 "증거 없음"이
    # 아니라 "실시간 흐름이 완전히 깨짐" = 해당 축 최악(1.0)으로 본다.
    # AP 텔레메트리(유선)는 이 지점에서 이미 정상 파싱됐으므로 "AP 다운"이
    # 아니라 "무선 채널 포화"가 원인이다. 상세: congestion_label_redesign.md
    if ping_hard_fail:
        latency_score = 1.0
    if probe_hard_fail:
        loss_score = 1.0

    congestion_score = round(
        max(
            occupancy_score,
            jitter_score,
            loss_score,
            latency_score,
        ),
        4,
    )

    if congestion_score < 0.25:
        label = 0
    elif congestion_score < 0.50:
        label = 1
    elif congestion_score < 0.75:
        label = 2
    else:
        label = 3

    return (
        round(throughput_score, 4),
        round(occupancy_score, 4),
        round(jitter_score, 4),
        round(loss_score, 4),
        round(latency_score, 4),
        round(retry_score, 4),
        congestion_score,
        label,
    )


# ============================================================
# CSV 구조 확인
# ============================================================

def prepare_csv():
    if not os.path.exists(CSV_FILE):
        return

    try:
        with open(
            CSV_FILE,
            "r",
            encoding="utf-8",
            newline="",
        ) as f:
            reader = csv.reader(f)
            existing_header = next(
                reader,
                None,
            )

    except Exception as e:
        print("CSV 기존 파일 확인 실패:")
        print(e)
        sys.exit(1)

    if existing_header == CSV_COLUMNS:
        return

    print()
    print(
        "ERROR: 기존 metrics_v2.csv의 "
        "컬럼 구조가 현재 최종 코드와 다릅니다."
    )
    print(
        "기존 데이터를 자동으로 덮어쓰지 않습니다."
    )
    print()
    print(
        "기존 metrics_v2.csv를 백업한 뒤 "
        "다시 실행하세요."
    )
    sys.exit(1)


# ============================================================
# CSV 저장
# ============================================================

def save_csv(row):
    file_exists = os.path.exists(CSV_FILE)

    with open(
        CSV_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(CSV_COLUMNS)

        writer.writerow(row)


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        print(
            "사용법: python3 collect_metrics.py <scenario>"
        )
        sys.exit(1)

    scenario = sys.argv[1]

    prepare_csv()

    print("--------------------------------")
    print("Wi-Fi 최종 지표 수집 시작")
    print(f"Scenario : {scenario}")
    print(f"CSV File : {CSV_FILE}")
    print("--------------------------------")

    last_iperf_json_mtime = (
        get_iperf_json_mtime()
    )

    previous_stations = {}
    previous_time = None

    previous_active = None
    previous_busy = None

    previous_rssi = None

    rssi_history = deque(
        maxlen=MOVING_AVG_WINDOW
    )

    # retry ratio를 폴링 단위로 계산하면 idle에서 표본이 작아(프레임
    # 수십개) 비율이 0~60%로 요동친다. 최근 몇 폴링을 rolling sum 해서
    # 안정화한다.
    retx_history = deque(maxlen=MOVING_AVG_WINDOW)
    pkts_history = deque(maxlen=MOVING_AVG_WINDOW)

    # 단일 폴링 스파이크 제거용 3-폴링 median
    occ_history = deque(maxlen=3)
    lat_history = deque(maxlen=3)

    sample = 0

    poller = APPoller()
    probe = ProbeRunner()
    last_cycle_id = None

    try:
        while True:
            loop_start = time.time()

            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # ------------------------------------------------
            # 1. AP Station + Survey (지속 SSH 세션에서 폴링)
            # ------------------------------------------------

            cycle = poller.wait_for_new_cycle(
                last_cycle_id
            )

            if cycle is None:
                print(
                    "AP 폴링 대기 중입니다 "
                    "(SSH 세션 재연결 중이거나 응답 없음, "
                    f"재연결 {poller.reconnects}회)..."
                )
                continue

            cycle_id, cycle_text = cycle
            last_cycle_id = cycle_id

            station, survey = parse_ap_cycle(
                cycle_text
            )

            if station is None:
                print(
                    "현재 연결된 Wi-Fi Station이 없습니다."
                )
                time.sleep(0.2)
                continue

            current_active, current_busy = survey

            (
                signal_avg,
                connected_clients,
                sta_tx_bitrate_min,
                sta_tx_bitrate_mean,
            ) = summarize_stations(station)

            # ------------------------------------------------
            # 2. Channel Occupancy
            # ------------------------------------------------

            (
                occupancy_raw,
                occupancy_method,
            ) = calculate_channel_occupancy(
                previous_active,
                previous_busy,
                current_active,
                current_busy,
            )

            # ------------------------------------------------
            # 3. Ping
            # ------------------------------------------------

            (
                latency_raw,
                jitter,
                ping_packet_loss,
            ) = get_ping_metrics()

            # ------------------------------------------------
            # 2b/3b. occupancy·latency 단일 폴링 스파이크 제거
            #   occupancy: instantaneous_fallback 경로가 가끔 100%를
            #     뱉음(survey 카운터 리셋/이상 read). latency: 노트북
            #     wifi가 가끔 순간적으로 나빠져 RTT 스파이크.
            #   3-폴링 median으로 스코어·CSV에 쓰는 값을 안정화한다.
            # ------------------------------------------------

            occ_history.append(occupancy_raw)
            lat_history.append(latency_raw)
            occupancy = median(occ_history)
            latency = median(lat_history)

            # ------------------------------------------------
            # 4. UDP Loss
            # ------------------------------------------------

            current_json_mtime = (
                get_iperf_json_mtime()
            )

            udp_packet_loss = None

            if (
                current_json_mtime is not None
                and (
                    last_iperf_json_mtime is None
                    or current_json_mtime
                    != last_iperf_json_mtime
                )
            ):
                new_udp_loss = (
                    get_iperf_udp_loss()
                )

                if new_udp_loss is not None:
                    udp_packet_loss = round(
                        max(
                            0.0,
                            min(
                                100.0,
                                new_udp_loss,
                            ),
                        ),
                        2,
                    )

                    last_iperf_json_mtime = (
                        current_json_mtime
                    )

            # ------------------------------------------------
            # 5. Throughput
            # ------------------------------------------------

            now = time.time()

            (
                rx_delta,
                tx_delta,
                tx_retries_delta,
                tx_failed_delta,
                tx_packets_delta,
            ) = calculate_station_deltas(
                previous_stations,
                station,
            )

            elapsed_since_previous = (
                now - previous_time
                if previous_time is not None
                else 0
            )

            if elapsed_since_previous <= 0:
                throughput = 0.0
                poll_interval_s = 0.0
            else:
                throughput = round(
                    (rx_delta + tx_delta)
                    * 8
                    / elapsed_since_previous
                    / 1_000_000,
                    2,
                )
                poll_interval_s = round(elapsed_since_previous, 3)

            # ------------------------------------------------
            # 6. Retry ratio (재설계: 절대 개수 -> 비율, rolling)
            #    retry_ratio = Σretx / (Σretx + Σpkts)  (최근 N 폴링)
            #    폴링 주기 무관. WLAN 헬스 표준 문턱(10/15/25/40%)에 매핑.
            # ------------------------------------------------

            tx_retx_delta = tx_retries_delta + tx_failed_delta
            retx_history.append(tx_retx_delta)
            pkts_history.append(tx_packets_delta)

            retx_sum = sum(retx_history)
            retry_denom = retx_sum + sum(pkts_history)

            if retry_denom >= 50:
                tx_retry_ratio = round(
                    retx_sum / retry_denom, 4
                )
            else:
                # 표본이 너무 작으면(idle 초반) 비율이 무의미
                tx_retry_ratio = 0.0

            # ------------------------------------------------
            # 6b. victim 프로브 (jitter / loss 축)
            # ------------------------------------------------

            probe_jitter_ms, probe_loss_pct, probe_fresh, probe_ever_ok = (
                probe.get()
            )
            if not probe_fresh:
                probe_jitter_ms = None
                probe_loss_pct = None

            # ------------------------------------------------
            # 6c. 실패=max 판정 (victim 경로 완전 붕괴)
            #   channel_active: 실제로 트래픽이 흐르는 중 (throughput 또는
            #     occupancy 기준). idle에 ping/프로브가 실패하는 건 셋업
            #     문제지 혼잡이 아니므로 override 안 함.
            #   ping_hard_fail: 3-폴링 median latency가 0 = 연속 무응답.
            #   probe_hard_fail: 프로브가 되다가(ever_ok) 죽어서 stale.
            #     한 번도 못 돈 경우(서버 미기동 등)는 override 안 함.
            # ------------------------------------------------

            channel_active = (throughput >= 3.0) or (occupancy >= 40.0)
            ping_hard_fail = channel_active and (latency == 0.0)
            probe_hard_fail = (
                channel_active
                and probe_ever_ok
                and not probe_fresh
            )

            # ------------------------------------------------
            # 7. RSSI + delta + moving average
            # ------------------------------------------------

            current_rssi = signal_avg

            if previous_rssi is None:
                rssi_delta = 0.0
            else:
                rssi_delta = (
                    current_rssi
                    - previous_rssi
                )

            rssi_history.append(
                current_rssi
            )

            rssi_moving_avg = (
                sum(rssi_history)
                / len(rssi_history)
            )

            # ------------------------------------------------
            # 8. Congestion Score + Label
            # ------------------------------------------------

            (
                throughput_score,
                occupancy_score,
                jitter_score,
                loss_score,
                latency_score,
                retry_score,
                congestion_score,
                label,
            ) = calculate_scores(
                throughput,
                occupancy,
                latency,
                probe_jitter_ms,
                probe_loss_pct,
                tx_retry_ratio * 100.0,
                ping_hard_fail=ping_hard_fail,
                probe_hard_fail=probe_hard_fail,
            )

            # ------------------------------------------------
            # 9. CSV
            # ------------------------------------------------

            save_csv([
                timestamp,
                scenario,
                poll_interval_s,
                round(throughput, 2),
                round(occupancy, 2),
                occupancy_method,
                round(latency, 3),
                (
                    round(probe_jitter_ms, 3)
                    if probe_jitter_ms is not None
                    else ""
                ),
                (
                    round(probe_loss_pct, 3)
                    if probe_loss_pct is not None
                    else ""
                ),
                1 if probe_fresh else 0,
                (
                    round(udp_packet_loss, 2)
                    if udp_packet_loss is not None
                    else ""
                ),
                int(tx_retx_delta),
                int(tx_packets_delta),
                tx_retry_ratio,
                round(current_rssi, 2),
                int(connected_clients),
                round(rssi_delta, 2),
                round(rssi_moving_avg, 2),
                round(sta_tx_bitrate_min, 1),
                round(sta_tx_bitrate_mean, 1),
                throughput_score,
                occupancy_score,
                jitter_score,
                loss_score,
                latency_score,
                retry_score,
                congestion_score,
                label,
            ])

            sample += 1

            # ------------------------------------------------
            # 10. 출력
            # ------------------------------------------------

            print("--------------------------------")
            print(
                f"Sample             : {sample}"
            )
            print(
                f"Throughput         : {throughput:.2f} Mbps"
            )
            print(
                f"Channel Occupancy  : {occupancy:.2f} %"
            )
            print(
                f"Occupancy Method   : {occupancy_method}"
            )
            print(
                f"Poll Interval      : {poll_interval_s:.2f} s"
            )
            print(
                f"Latency (ping RTT) : {latency:.1f} ms"
                f"{'  [HARD-FAIL -> lat=1.0]' if ping_hard_fail else ''}"
            )
            _pj = (
                f"{probe_jitter_ms:.2f} ms"
                if probe_jitter_ms is not None
                else "N/A"
            )
            _pl = (
                f"{probe_loss_pct:.2f} %"
                if probe_loss_pct is not None
                else "N/A"
            )
            print(
                f"Probe Jitter/Loss  : {_pj} / {_pl}"
                f"  ({'fresh' if probe_fresh else 'STALE'})"
                f"{'  [HARD-FAIL -> loss=1.0]' if probe_hard_fail else ''}"
            )
            print(
                f"Retry Ratio        : {tx_retry_ratio * 100:.1f} %  "
                f"(retx {tx_retx_delta} / pkts {tx_packets_delta})"
            )
            print(
                f"RSSI / MovAvg      : {current_rssi:.1f} / "
                f"{rssi_moving_avg:.1f} dBm   Clients {connected_clients}"
            )
            print(
                f"Sta tx rate min/avg: {sta_tx_bitrate_min:.0f} / "
                f"{sta_tx_bitrate_mean:.0f} Mbit/s"
            )
            print(
                f"Scores  occ={occupancy_score:.2f} "
                f"jit={jitter_score:.2f} loss={loss_score:.2f} "
                f"lat={latency_score:.2f} retry={retry_score:.2f}  "
                f"(thr={throughput_score:.2f}, 라벨 제외)"
            )
            print(
                f"Congestion (max)   : {congestion_score:.4f}"
            )
            print(
                f"Label              : {label}"
            )
            print(
                f"CSV 저장 완료      : {CSV_FILE}"
            )

            # ------------------------------------------------
            # 11. 이전값 갱신
            # ------------------------------------------------

            previous_stations = station
            previous_time = now

            previous_active = current_active
            previous_busy = current_busy

            previous_rssi = current_rssi

            # ------------------------------------------------
            # 루프 주기
            #
            # 측정에 걸린 시간을 고려하여 추가 sleep은
            # 최소화한다. 현재 수집 작업 자체가 주기를 결정한다.
            # ------------------------------------------------

            elapsed = time.time() - loop_start

            if elapsed < 0.5:
                time.sleep(0.5 - elapsed)

    except KeyboardInterrupt:
        print()
        print("--------------------------------")
        print("지표 수집 종료")
        print(f"CSV 파일 : {CSV_FILE}")
        print(
            f"AP 폴링 재연결 횟수 : {poller.reconnects}"
        )
        print(
            f"프로브 실행/실패    : {probe.runs} / {probe.failures}"
        )
        print("--------------------------------")

    finally:
        poller.stop()
        probe.stop()


if __name__ == "__main__":
    main()
