#!/usr/bin/env python3
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import deque

# ============================================================
# 기본 설정
# ============================================================

AP_IP = "192.168.8.1"
SERVER_IP = "192.168.8.109"
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

THROUGHPUT_MAX_MBPS = 150.0
RETRY_FAILED_MAX = 100.0
JITTER_MAX_MS = 1.0

# ============================================================
# SSH
# ============================================================

SSH_CMD = [
    "ssh",
    "-o", "HostKeyAlgorithms=+ssh-rsa",
    "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
    "-o", "ConnectTimeout=5",
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
    "throughput_mbps",
    "channel_occupancy_percent",
    "channel_occupancy_method",
    "latency_ms",
    "jitter_ms",
    "packet_loss_udp_percent",
    "tx_retries_delta",
    "tx_failed_delta",
    "rssi_dbm",
    "connected_clients",
    "rssi_delta_db",
    "rssi_moving_avg_dbm",
    "scenario",
    "throughput_score",
    "occupancy_score",
    "retry_failed_score",
    "jitter_score",
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


def run_ap_command(command, timeout=10):
    return run_command(
        SSH_CMD + [command],
        timeout=timeout,
    )


# ============================================================
# AP station + survey를 한 번의 SSH 연결로 수집
# ============================================================

def get_ap_metrics():
    command = (
        f"echo '__STATION_BEGIN__'; "
        f"iw dev {INTERFACE} station dump; "
        f"echo '__SURVEY_BEGIN__'; "
        f"iw dev {INTERFACE} survey dump"
    )

    output = run_ap_command(command, timeout=10)

    if not output:
        return None, None

    station_marker = "__STATION_BEGIN__"
    survey_marker = "__SURVEY_BEGIN__"

    if station_marker not in output or survey_marker not in output:
        return None, None

    station_text = output.split(station_marker, 1)[1]
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
    stations = []
    current = None

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if line.startswith("Station "):
            if current is not None:
                stations.append(current)

            current = {
                "rx_bytes": 0,
                "tx_bytes": 0,
                "tx_retries": 0,
                "tx_failed": 0,
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

    if current is not None:
        stations.append(current)

    if not stations:
        return None

    rx_bytes = sum(s["rx_bytes"] for s in stations)
    tx_bytes = sum(s["tx_bytes"] for s in stations)
    tx_retries = sum(s["tx_retries"] for s in stations)
    tx_failed = sum(s["tx_failed"] for s in stations)

    signal_values = [
        s["signal_avg"]
        for s in stations
        if s["signal_avg"] is not None
    ]

    signal_avg = (
        sum(signal_values) / len(signal_values)
        if signal_values
        else 0.0
    )

    return {
        "rx_bytes": rx_bytes,
        "tx_bytes": tx_bytes,
        "tx_retries": tx_retries,
        "tx_failed": tx_failed,
        "signal_avg": signal_avg,
        "connected_clients": len(stations),
    }


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
    output = run_command(
        [
            "ping",
            "-c", "4",
            "-W", "1",
            SERVER_IP,
        ],
        timeout=8,
    )

    latency = 0.0
    jitter = 0.0
    packet_loss = 0.0

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

def calculate_station_throughput(
    previous,
    current,
    elapsed,
):
    if previous is None or elapsed <= 0:
        return 0.0

    rx_delta = max(
        0,
        current["rx_bytes"]
        - previous["rx_bytes"],
    )

    tx_delta = max(
        0,
        current["tx_bytes"]
        - previous["tx_bytes"],
    )

    total_bytes = rx_delta + tx_delta

    mbps = (
        total_bytes
        * 8
        / elapsed
        / 1_000_000
    )

    return round(mbps, 2)


# ============================================================
# Score
# ============================================================

def clamp01(value):
    return max(0.0, min(1.0, value))


def calculate_scores(
    throughput,
    occupancy,
    tx_retries_delta,
    tx_failed_delta,
    jitter,
):
    # 부하가 높을수록 혼잡도가 높아진다는 실험 목적의 점수.
    throughput_score = clamp01(
        throughput / THROUGHPUT_MAX_MBPS
    )

    occupancy_score = clamp01(
        occupancy / 100.0
    )

    retry_failed_score = clamp01(
        (
            tx_retries_delta
            + tx_failed_delta
        )
        / RETRY_FAILED_MAX
    )

    jitter_score = clamp01(
        jitter / JITTER_MAX_MS
    )

    congestion_score = (
        0.35 * throughput_score
        + 0.35 * occupancy_score
        + 0.20 * retry_failed_score
        + 0.10 * jitter_score
    )

    congestion_score = round(
        clamp01(congestion_score),
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
        round(retry_failed_score, 4),
        round(jitter_score, 4),
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

    previous_station = None
    previous_time = None

    previous_retries = None
    previous_failed = None

    previous_active = None
    previous_busy = None

    previous_rssi = None

    rssi_history = deque(
        maxlen=MOVING_AVG_WINDOW
    )

    sample = 0

    try:
        while True:
            loop_start = time.time()

            timestamp = time.strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # ------------------------------------------------
            # 1. AP Station + Survey
            # ------------------------------------------------

            station, survey = get_ap_metrics()

            if station is None:
                print(
                    "현재 연결된 Wi-Fi Station이 없습니다."
                )
                time.sleep(1)
                continue

            current_active, current_busy = survey

            connected_clients = (
                station["connected_clients"]
            )

            # ------------------------------------------------
            # 2. Channel Occupancy
            # ------------------------------------------------

            (
                occupancy,
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
                latency,
                jitter,
                ping_packet_loss,
            ) = get_ping_metrics()

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

            if previous_station is None:
                throughput = 0.0
            else:
                throughput = (
                    calculate_station_throughput(
                        previous_station,
                        station,
                        now - previous_time,
                    )
                )

            # ------------------------------------------------
            # 6. Retry / Failed delta
            # ------------------------------------------------

            if previous_retries is None:
                tx_retries_delta = 0
            else:
                tx_retries_delta = max(
                    0,
                    station["tx_retries"]
                    - previous_retries,
                )

            if previous_failed is None:
                tx_failed_delta = 0
            else:
                tx_failed_delta = max(
                    0,
                    station["tx_failed"]
                    - previous_failed,
                )

            # ------------------------------------------------
            # 7. RSSI + delta + moving average
            # ------------------------------------------------

            current_rssi = (
                station["signal_avg"]
            )

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
                retry_failed_score,
                jitter_score,
                congestion_score,
                label,
            ) = calculate_scores(
                throughput,
                occupancy,
                tx_retries_delta,
                tx_failed_delta,
                jitter,
            )

            # ------------------------------------------------
            # 9. CSV
            # ------------------------------------------------

            save_csv([
                timestamp,
                round(throughput, 2),
                round(occupancy, 2),
                occupancy_method,
                round(latency, 3),
                round(jitter, 3),
                (
                    round(udp_packet_loss, 2)
                    if udp_packet_loss is not None
                    else ""
                ),
                int(tx_retries_delta),
                int(tx_failed_delta),
                round(current_rssi, 2),
                int(connected_clients),
                round(rssi_delta, 2),
                round(rssi_moving_avg, 2),
                scenario,
                throughput_score,
                occupancy_score,
                retry_failed_score,
                jitter_score,
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
                f"Latency            : {latency:.3f} ms"
            )
            print(
                f"Jitter             : {jitter:.3f} ms"
            )

            if udp_packet_loss is not None:
                print(
                    f"UDP Loss (iperf3)  : "
                    f"{udp_packet_loss:.2f} %"
                )
            else:
                print(
                    "UDP Loss (iperf3)  : N/A "
                    "(new JSON result 없음)"
                )

            print(
                f"TX Retries(delta)  : {tx_retries_delta}"
            )
            print(
                f"TX Failed(delta)   : {tx_failed_delta}"
            )
            print(
                f"RSSI               : {current_rssi:.1f} dBm"
            )
            print(
                f"Connected Clients  : {connected_clients}"
            )
            print(
                f"RSSI Moving Avg    : "
                f"{rssi_moving_avg:.1f} dBm"
            )
            print(
                f"Throughput Score   : "
                f"{throughput_score:.4f}"
            )
            print(
                f"Occupancy Score    : "
                f"{occupancy_score:.4f}"
            )
            print(
                f"Retry/Failed Score : "
                f"{retry_failed_score:.4f}"
            )
            print(
                f"Jitter Score       : "
                f"{jitter_score:.4f}"
            )
            print(
                f"Congestion Score   : "
                f"{congestion_score:.4f}"
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

            previous_station = station
            previous_time = now

            previous_retries = (
                station["tx_retries"]
            )
            previous_failed = (
                station["tx_failed"]
            )

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
        print("--------------------------------")


if __name__ == "__main__":
    main()
