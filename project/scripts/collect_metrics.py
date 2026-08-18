import subprocess
import csv
import time
import re
import os
import sys

AP_IP = "192.168.8.1"

SSH_CMD = [
    "ssh",
    "-o", "HostKeyAlgorithms=+ssh-rsa",
    "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
    f"root@{AP_IP}"
]


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )
    return result.stdout


def get_throughput():
    output = run_command(["iperf3", "-c", AP_IP])

    for line in output.splitlines():
        if "receiver" in line:
            match = re.search(r'([\d.]+)\s+Mbits/sec', line)
            if match:
                return float(match.group(1))

    return 0.0


def get_latency():
    output = run_command(["ping", "-c", "4", AP_IP])

    for line in output.splitlines():
        if "rtt min/avg/max" in line:
            avg = line.split("=")[1].split("/")[1]
            return float(avg)

    return 0.0


def get_survey():
    return run_command(
        SSH_CMD + ["iw dev wlan0 survey dump"]
    )


def get_link():
    return run_command(
        SSH_CMD + ["ip -s link show wlan0"]
    )


def parse_channel_occupancy(survey_output):
    active = None
    busy = None

    for line in survey_output.splitlines():
        line = line.strip()

        if "channel active time:" in line:
            active = float(
                line.split(":")[1].replace("ms", "").strip()
            )

        elif "channel busy time:" in line:
            busy = float(
                line.split(":")[1].replace("ms", "").strip()
            )

    if active is None or busy is None or active == 0:
        return 0.0

    return round((busy / active) * 100, 2)


def parse_packet_loss(link_output):

    rx_errors = 0
    tx_errors = 0

    lines = link_output.splitlines()

    for i, line in enumerate(lines):

        if "RX:" in line:
            values = lines[i + 1].split()
            rx_errors = int(values[2])

        if "TX:" in line:
            values = lines[i + 1].split()
            tx_errors = int(values[2])

    return rx_errors + tx_errors


def save_csv(
        throughput,
        occupancy,
        packet_loss,
        latency,
        scenario):

    filename = "raw_metrics.csv"

    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="") as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "throughput_mbps",
                "channel_occupancy",
                "packet_loss",
                "latency",
                "label",
                "scenario"
            ])

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            throughput,
            occupancy,
            packet_loss,
            latency,
            -1,
            scenario
        ])


def main():

    scenario = "test"

    if len(sys.argv) >= 2:
        scenario = sys.argv[1]

    throughput = get_throughput()

    survey = get_survey()
    occupancy = parse_channel_occupancy(survey)

    link = get_link()
    packet_loss = parse_packet_loss(link)

    latency = get_latency()

    print("--------------------------------")
    print(f"Throughput         : {throughput:.1f} Mbps")
    print(f"Channel Occupancy  : {occupancy:.2f} %")
    print(f"Packet Loss        : {packet_loss}")
    print(f"Latency            : {latency:.3f} ms")
    print("--------------------------------")

    save_csv(
        throughput,
        occupancy,
        packet_loss,
        latency,
        scenario
    )

    print("CSV 저장 완료 : raw_metrics.csv")


if __name__ == "__main__":
    main()
