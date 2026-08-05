# collect_metrics.py

import subprocess
import csv
import time


def collect_wifi_metrics():
    # 채널 점유율 수집
    survey = subprocess.run(
        ["iw", "dev", "wlan0", "survey", "dump"],
        capture_output=True,
        text=True
    )

    # 인터페이스 통계
    link = subprocess.run(
        ["ip", "-s", "link", "show", "wlan0"],
        capture_output=True,
        text=True
    )

    return parse_metrics(survey.stdout, link.stdout)


def parse_metrics(survey_output, link_output):
    channel_occupancy = parse_channel_occupancy(survey_output)
    packet_loss = parse_packet_loss(link_output)

    return channel_occupancy, packet_loss


def parse_channel_occupancy(survey_output):
    active_time = None
    busy_time = None

    for line in survey_output.splitlines():
        line = line.strip()

        if "channel active time:" in line:
            active_time = float(
                line.split(":")[1].replace("ms", "").strip()
            )

        elif "channel busy time:" in line:
            busy_time = float(
                line.split(":")[1].replace("ms", "").strip()
            )

    if active_time is None or busy_time is None or active_time == 0:
        return 0.0

    return round((busy_time / active_time) * 100, 2)


def parse_packet_loss(link_output):
    # 1단계에서는 임시값 사용
    return 0.0


def save_csv(channel_occupancy, packet_loss):
    filename = "raw_metrics.csv"

    with open(filename, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            0.0,                    # throughput_mbps
            channel_occupancy,
            packet_loss,
            0.0,                    # latency
            -1,                     # label
            "test"                  # scenario
        ])

    print("CSV 저장 완료 :", filename)


if __name__ == "__main__":

    channel_occupancy, packet_loss = collect_wifi_metrics()

    print(f"Channel Occupancy : {channel_occupancy:.2f}%")
    print(f"Packet Loss       : {packet_loss:.2f}%")

    save_csv(channel_occupancy, packet_loss)
