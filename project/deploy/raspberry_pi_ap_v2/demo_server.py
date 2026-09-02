"""데모 서버 진입점 — 참가형 웹사이트로 폰별 부하를 걸고, 모델의 실시간 혼잡 예측을 웹으로 본다.

노트북에서 실행 (기본값):
    python project/demo/demo_server.py
라즈베리 파이(엣지 추론)에서 실행:
    python3 demo_server.py --iperf-target <파이 IP> --s21 <user@phone> --s26 <user@phone>
브라우저에서 http://<host>:8000/ 열기.

구성이 4개 모듈로 나뉜다 (2026-09-02, API가 한 파일에 다 들어있어 혼잡하다는 피드백 반영):
  - demo_state.py       설정(인자 파싱)·공유 상태(_state/_load_state)·sys.path 부트스트랩
  - demo_inference.py   AP 폴링 → 7-feature → window 12 → ONNX 추론 루프
  - demo_load.py        폰별 부하 제어(SSH iperf3) + 연결확인
  - demo_api.py         HTTP 라우팅(Handler) — 이 파일이 "API"
  - demo_server.py (이 파일)  실행 진입점: iperf3 -s 기동, 추론 스레드 시작, HTTP 서버 구동

엔드포인트 (계약 상세는 API.md):
  - GET  /            데모 페이지
  - GET  /events      SSE — 상태를 ~2Hz 로 스트리밍
  - POST /load        {"phone": "s21"|"s26", "rate": "10M"|"20M"|"30M"|"40M"|"off",
                        "packet_size": 1400|200 (생략 시 1400)}
                       해당 폰에만 iperf3 부하. "off" 외 rate는 LOAD_DURATION_S(10초) 뒤 서버가 자동 종료.
  - GET  /signal      두 폰의 현재 신호세기 (부하 전 대칭 확인용)
  - GET  /check       Pi(서버 자신)·AP·S21·S26 연결 상태 일괄 확인 ("연결확인" 버튼용)
  - GET  /health

폰별 최대 40M · 10초 자동종료 (2026-09-02 참가형 데모용 — 기존 "두 폰 동시 30M 무제한"보다
훨씬 짧고 한 폰씩이라 안전 여유가 큼. combined 다중분 부하의 크래시 위험은
docs/yongsang/ap_crash_analysis.{md,html} 참고, 단발 10초는 그 영역 밖).
iperf3 -s 서버(포트 2개)는 서버 시작 시 --iperf-target 이 로컬이면 자동 기동.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from http.server import ThreadingHTTPServer

from demo_state import ARGS, PHONES
from demo_inference import inference_loop
from demo_load import set_phone_load
from demo_api import Handler


def _iperf_target_is_local() -> bool:
    t = ARGS.iperf_target
    if t in ("127.0.0.1", "localhost", "0.0.0.0"):
        return True
    try:
        local_ips = subprocess.run(["hostname", "-I"], capture_output=True, text=True,
                                   timeout=4).stdout.split()
        return t in local_ips
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    if not ARGS.model.exists() or not ARGS.scaler.exists():
        sys.exit(f"모델/스케일러 없음: {ARGS.model}  {ARGS.scaler}")

    iperf_procs = []
    spawn = not ARGS.no_iperf_server and _iperf_target_is_local()
    if spawn:
        for ph in PHONES.values():
            iperf_procs.append(subprocess.Popen(
                ["iperf3", "-s", "-p", str(ph["port"])],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
        print(f"iperf3 서버 기동: {[p['port'] for p in PHONES.values()]}  (대상 {ARGS.iperf_target})")
    else:
        print(f"iperf3 -s 자동 기동 안 함 — 대상 {ARGS.iperf_target} 에서 직접 "
              f"iperf3 -s -p {ARGS.s21_port} / -p {ARGS.s26_port} 를 띄워 둘 것")

    print(f"모델 : {ARGS.model.name}")
    print(f"폰   : s21={ARGS.s21}  s26={ARGS.s26}")

    threading.Thread(target=inference_loop, daemon=True).start()

    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    print(f"데모 서버: http://{ARGS.host}:{ARGS.port}/   (Ctrl-C 종료)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n정리 중... 부하 정지")
        for name in PHONES:
            set_phone_load(name, "off")
        for p in iperf_procs:
            p.terminate()


if __name__ == "__main__":
    main()
