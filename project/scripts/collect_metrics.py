# collect_metrics.py
import subprocess
import time
import csv

def collect_wifi_metrics():
    # 채널 점유율 수집
    survey = subprocess.run(
        ['iw', 'dev', 'wlan0', 'survey', 'dump'],
        capture_output=True, text=True
    )
    # 패킷 손실률 수집
    link = subprocess.run(
        ['ip', '-s', 'link', 'show', 'wlan0'],
        capture_output=True, text=True
    )
    return parse_metrics(survey.stdout, link.stdout)

def parse_metrics(survey_output, link_output):
    # 파싱 로직 구현
    channel_occupancy = parse_channel_occupancy(survey_output)
    packet_loss = parse_packet_loss(link_output)
    return channel_occupancy, packet_loss
