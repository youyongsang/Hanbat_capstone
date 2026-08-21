# Capstone-Design 현재 상태
최종 업데이트: 2026-08-21 (시간 미상)

## 프로젝트 개요
산업 무선망(AP) 트래픽 혼잡을 Early Exit LSTM으로 실시간 분류하고, Raspberry Pi + ONNX/INT8로 엣지 배포하는 캡스톤 프로젝트. 방학 중 교수 피드백에 따라 1학기 4-feature 시뮬레이터 기반에서 실제 GL.iNet AP 실측 9-feature(`ap_metrics_cleaned_strict`) 기반으로 피벗함. 팀: 유용상(모델 설계), 장예나(데이터), 김호중(경량화·배포).

## 완료된 작업
- (8/17까지) AP strict 9-feature SDN 백본 신규 학습, ONNX/INT8/Pi 추론 파이프라인 전체 신규 구축, 관련 문서/가이드 정리 — 자세한 내용은 progress.md 8/17 항목 참고
- **역할 재확인**: 호중 노트북 발열 문제로 라즈베리파이를 용상이 직접 이어받게 됨. 이 과정에서 "AP 기기 측정"이라는 표현이 두 가지 다른 작업을 가리킬 수 있음을 확인하고 구분함
  1. Pi ONNX 추론 속도 재측정 (`project/deploy/raspberry_pi_ap/`, 오프라인, 이미 저장된 `test.csv` 기반 — AP 전원 불필요)
  2. AP 기기 라이브 트래픽 측정/원본 CSV 수집 (`project/scripts/collect_metrics.py`, AP가 켜져서 실제 트래픽이 흘러야 함 — 원래 호중 담당이었는데 이것도 용상이 맡게 됨)
- `collect_metrics.py` 코드 구조 분석: WiFi Client 노트북에서 실행, AP(`192.168.8.1`)에 SSH로 station/survey dump 수집 + ping(SERVER_IP) + 로컬 `iperf3_result.json` 폴링 → `metrics_v2.csv`에 누적 저장. 시나리오별(iperf3 -c 대역폭 조정) 실행 절차 정리해서 안내함
- 라즈베리파이 헤드리스 SSH 접속 방법 안내: SD카드는 호중이 이미 구워놨으므로 Imager 고급설정(SSH 활성화) 단계는 스킵, 랜선으로 AP LAN 포트 연결 → `raspberrypi.local` 또는 IP로 노트북에서 `ssh pi@...` 접속하는 절차만 남음
- 메모리 저장: "AP 측정 담당자가 호중→용상으로 바뀜" project memory 기록 (`project_ap_measurement_owner.md`)

## 현재 작업 중
- **막힘**: AP 기기(GL.iNet Opal) WiFi SSID가 스캔 목록에 안 뜸. 노트북을 AP LAN 포트에 유선 직결해서 `http://192.168.8.1` 관리 페이지 접속되는지 확인해보라고 안내한 상태 — 결과 대기 중
- 이 문제가 풀려야 (a) 노트북이 AP WiFi에 붙어서 `raspberrypi.local`로 Pi에 SSH 접속 가능해지고, (b) `collect_metrics.py` 실행에 필요한 WiFi Client 부하 생성도 가능해짐

## 다음 할 일
- [ ] AP 관리 페이지(`192.168.8.1`) 유선 접속 여부 확인
  - 접속 안 되면: AP 전원/부팅 자체 문제
  - 접속 되면: WiFi 라디오 꺼짐 / 리피터·클라이언트 모드로 설정됨 / SSID 숨김 / 밴드(2.4·5GHz) 문제 중 확인
- [ ] AP WiFi 살아나면 노트북을 AP WiFi에 연결 → Pi에 `ssh pi@raspberrypi.local`(또는 IP)로 헤드리스 접속
  - Pi 계정 정보(비밀번호/SSH 키)를 호중이 세팅해서 용상이 모를 수 있음 — 확인 필요
- [ ] Pi 접속되면 `project/deploy/raspberry_pi_ap/` 번들 scp로 전송 → SDN FP32/INT8 재측정 2줄 실행 (`--mode staged-confidence`, threshold 0.85, `--max-samples 82`) → `analyze_pi_results.py` 분석 → `project/results/yongsang/`에 저장 후 커밋
- [ ] 별도로 `collect_metrics.py` 라이브 AP 측정도 진행 필요: iperf3 서버(유선)/클라이언트(WiFi) 준비, `SERVER_IP` 하드코딩값(`192.168.8.109`)이 당일 실제 서버 IP와 일치하는지 확인
- [ ] 결과 받으면 `ap_model_comparison_cleaned_strict.*`의 SDN 행 최종 교체
- [ ] (여유 시) AP strict용 실시간 추론 파이프라인 설계 착수 — 현재 어느 브랜치에도 코드 없음

## 주요 파일
- `project/README_AP_STRICT.md` — AP strict 파이프라인 전체 기준 문서
- `docs/hochung/ap_strict_sdn_pi_rerun.md` — SDN Pi 재측정 가이드 (호중 파이프라인 기준 최소 diff)
- `docs/hochung/ap_traffic_measurement_guide.md` — AP 라이브 트래픽 측정 방법(토폴로지, iperf3, `iw` 수집 항목) 정리 문서, 원래 담당은 호중이었으나 지금은 용상 참고용
- `project/scripts/collect_metrics.py` — AP 라이브 측정 실행 스크립트 (WiFi Client 노트북에서 `<scenario>` 인자로 실행)
- `project/deploy/raspberry_pi_ap/README.md` — Pi ONNX 8개 조합(FP32/INT8 × Baseline/SDN/Fixed/Dynamic) 재측정 명령어 전체
- `project/checkpoints/ap_cleaned_strict/ap_sdn_lstm_best.pth` — 신규 학습한 독립 SDN 백본
- `docs/hochung/Raspberry_Pi_AP_9feature_FP32_INT8_최종비교표.xlsx` — 기존 Pi 실측 (SDN 행만 재측정 필요)

## 특이사항 / 결정 사항
- "AP 기기 측정"이라는 말이 (1) Pi ONNX 오프라인 속도 벤치마크와 (2) AP 라이브 트래픽 원본 CSV 수집, 두 가지를 다 가리킬 수 있어 혼동하기 쉬움 — 항상 어느 쪽인지 먼저 확인할 것
- AP 모델은 GL.iNet Opal(GL-SFT1200), 관리 IP `192.168.8.1`. `collect_metrics.py`의 `SERVER_IP`(`192.168.8.109`)는 하드코딩값이라 실제 네트워크 구성과 다르면 코드 상단에서 직접 수정해야 함
- AP strict `test.csv`는 82개 샘플뿐 (1학기 `project/data/real`은 351개) — 헷갈리기 쉬우니 항상 데이터셋 구분해서 명령어 확인할 것
- Fixed/Dynamic은 같은 체크포인트(`ap_early_exit_lstm_best.pth`)에서 threshold 정책만 다르게 평가하는 것이고, SDN은 반드시 독립 학습해야 공정 비교가 됨
- `torch`가 시스템 기본 python(anaconda base)에서 DLL 로딩 실패함 — 반드시 `C:\Users\PC\anaconda3\envs\capstone\python.exe` conda 환경 사용해야 함
