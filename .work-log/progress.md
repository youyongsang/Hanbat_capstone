## 2026-08-22
- (8/21 밤부터 이어지는 세션, 새 노트북 DESKTOP-29GLQJF) AP WiFi SSID는 뜨는데 `192.168.8.1` 관리 페이지/SSH가 안 열리는 문제의 원인을 추적: 노트북이 Opal이 아니라 집 공유기(다른 물리 기기, MAC 다름)가 준 IP(`192.168.75.x`)를 받고 있었음을 확인. 원인은 집 랜선을 Opal의 LAN 포트에 꽂아서 브릿지 모드처럼 동작한 것
- 호중에게 확인해서 원래 성공했던 구성("집 공유기 → Opal WAN 포트")을 알아냄. 랜선을 WAN 포트로 옮겨 연결 → 노트북이 `192.168.8.226` 받고 인터넷+AP 관리 페이지 동시에 정상 작동 확인 (Claude Code 세션 끊김 없이 유지)
- AP(`root@192.168.8.1`) SSH는 호중이 알려준 비번으로 성공(단 `HostKeyAlgorithms=+ssh-rsa` 등 구형 dropbear 호환 옵션 필요). RSA 키 생성 후 `ssh-copy-id`로 등록, `~/.ssh/config`에 등록해서 비밀번호 없이 자동 인증되도록 설정 완료
- Pi(`pi@192.168.8.109`) SSH는 호중이 알려준 비번이 3회 다 실패. 호중 본인은 된다고 하는데 우리 쪽만 안 됨 → 계정 이름이 `pi`가 아닐 가능성으로 결론, 호중에게 실제 명령어 전체를 복사해서 보내달라고 요청. **아직 미해결**, SD카드 재굽기는 대안으로만 검토 중
- `collect_metrics.py`의 Windows ping 파싱 버그 발견 및 수정: 기존 코드가 Linux `ping -c/-W` 형식만 파싱해서 Windows(한글 로캘)에서 latency/jitter/packet_loss가 조용히 전부 0으로 찍히는 문제였음. `platform.system()` 분기 + `TTL=` 줄에서 로케일 무관 `(숫자)ms` 정규식 추출로 수정
- 이 노트북(WiFi Client) + 폰 Termux(`iperf3 -s` 서버) 구성으로 5개 시나리오(`normal_idle`/`low_load`/`medium_load`/`high_load`/`stress_load`) 라이브 실측 완료, 총 91행 `project/scripts/metrics_v2.csv`에 저장 (git 미커밋). 부하 증가에 따라 UDP 손실률이 1.2%→68.1%로 뚜렷하게 증가하는 것 확인
- `low_load` 첫 수집 시 프로세스 종료 타이밍이 늦어서 157초 중 대부분이 무부하 상태로 잘못 라벨링된 오염 데이터 발견 → 삭제 후 정확히 68초로 재수집
- `connected_clients`가 1→2로 바뀌는 순간(station 재연결) throughput/재전송이 비현실적으로 튀는 구조적 버그 발견(스크립트 미수정, 팀 공유용으로만 기록)
- torch DLL 로딩 실패(이 노트북 anaconda base에서도 재현) → 새 conda 환경 `capstone` 생성해서 torch(CPU)+pandas+numpy 설치로 해결
- `prepare_ap_metrics_dataset.py`로 새 실측 데이터를 별도 폴더(`project/data/ap_metrics_new_collection/`, 기존 `ap_metrics_cleaned_strict`는 안 건드림)에 윈도우 변환 (41 샘플)
- **중요 발견**: 원본 학습 스케일러(`ap_metrics_cleaned_strict/scaler_params.json`)와 실측 데이터 범위가 완전히 다름 — latency_ms(원본 0.047~0.163 vs 실측 2~841), tx_retries_delta(원본 최대 23 vs 실측 최대 20만대), rssi_dbm(원본 -30~-17 vs 실측 -67~-53.5). 1학기/AP strict 원본 데이터 측정 방식에 단위 버그가 있거나 완전히 다른 물리 조건(매우 근접 거리)에서 수집됐을 가능성 — 예나·팀 공유 필요
- `evaluate_ap_early_exit.py`로 새 데이터 평가 (`project/results/yongsang/ap_new_collection_eval_report.txt`): 정확도 50%(test 샘플 6개뿐이라 통계적 의미 낮음), 정상/경고는 100% 맞히지만 혼잡/심각은 0% — 사전학습 모델이 새 측정 환경에 일반화 안 되는 것을 확인
- `.work-log/current.md`, `.work-log/progress.md` 갱신

## 2026-08-21
- 호중 노트북 발열 문제로 라즈베리파이를 용상이 직접 이어받게 되면서, "AP 기기 측정"이 서로 다른 두 작업을 가리킬 수 있음을 정리: (1) Pi ONNX 추론 속도 재측정(오프라인, AP 불필요) vs (2) AP 라이브 트래픽 원본 CSV 수집(`collect_metrics.py`, AP 전원+트래픽 필요, 원래 호중 담당)
- Pi ONNX 추론 속도 재측정: `project/deploy/raspberry_pi_ap/README.md` 기준 8개 명령어(Baseline/SDN/Fixed/Dynamic × FP32/INT8) 실행 절차와 결과 회수 방법 안내
- `collect_metrics.py` 코드 읽고 실행 구조 파악: WiFi Client 노트북에서 실행, AP SSH station/survey dump + ping + 로컬 iperf3 JSON 폴링 → CSV 누적. 시나리오별(normal_idle~multi_client_load) iperf3 서버/클라이언트 실행 절차와 사전 체크리스트(AP 전원, SSH 키, IP 설정 일치) 정리
- 라즈베리파이 헤드리스 SSH 접속 방법 안내: SD카드 굽기(호중이 이미 완료)는 스킵, 랜선 연결 → `raspberrypi.local`/IP로 노트북에서 SSH 접속하는 절차만 설명
- 실제 진행 중 AP 기기 WiFi SSID가 안 뜨는 하드웨어 문제 발생 → 유선 직결로 관리 페이지(`192.168.8.1`) 접속 여부부터 확인하는 트러블슈팅 절차 안내, 결과 대기 중
- memory에 "AP 측정 담당자 호중→용상 변경" project memory 신규 저장

## 2026-08-17
- CLAUDE.md/docs 검토해서 프로젝트 현황 파악 (1학기 4-feature → 방학 중 AP strict 9-feature 피벗 확인), 계획서 PDF/PPTX 등 새로 추가된 문서도 검토
- README_AP_STRICT.md에 4→9 feature 확장 사유 섹션(사용자가 이미 추가한 것) 확인
- AP용 SDN-style LSTM을 독립 백본으로 신규 학습: `models/sdn_lstm.py`, `models/ap_sdn_lstm.py`, `scripts/train_ap_sdn.py`, `scripts/evaluate_ap_sdn.py` 작성 후 학습 (Test Acc 91.5%, Label2 72.7%) → `ap_sdn_lstm_best.pth`
- `generate_ap_comparison.py`를 기존 confidence-only 임시 SDN 재사용 방식에서 신규 독립 SDN 백본 사용으로 교체, 비교표 재생성 및 커밋/push (0d9ecb7)
- 호중이 올린 `Raspberry_Pi_AP_9feature_FP32_INT8_최종비교표.xlsx` 검토 → SDN 행이 옛날 임시 방식 기준임을 확인
- hojung 브랜치 fetch해서 AP strict 관련 코드가 전혀 없음을 확인(로컬에서만 작업했던 것으로 추정)
- AP strict 9-feature 전용 ONNX export/INT8 양자화/Pi 추론 파이프라인 신규 구축: `export_onnx_ap.py`, `export_onnx_ap_sdn.py`, `export_onnx_int8_ap.py`, `inference_pi_ap.py`, `prepare_pi_bundle_ap.py` → `project/deploy/raspberry_pi_ap/` 번들 생성, PC에서 4가지 추론 모드 전부 smoke test 통과, 커밋/push (1378a16)
- 문서 버그 발견 및 수정: AP strict `test.csv`가 실제 82개 샘플인데 가이드에 1학기 기준 351로 잘못 적혀있던 것 수정 (061b4af)
- 호중용 SDN 재측정 가이드 `docs/hochung/ap_strict_sdn_pi_rerun.md` 작성 → 이후 사용자 피드백 받아 "새 스크립트로 갈아타기"가 아니라 "호중 기존 파이프라인에서 체크포인트 경로 + threshold(0.50→0.85)만 바꾸는 최소 diff" 형태로 재작성 (4f0dc8f, 189b987)
- 호중이 보낸 상세 Pi 실험 리포트 분석: SDN 모델이 Early Exit 내부 classifier 재사용 + threshold 0.50 사용 확인, 테스트 샘플 82개 확인
- AP 실측 네트워크 토폴로지(호중 설명) 검토 → `ap_traffic_measurement_guide.md` 권장 구성과 일치함을 확인
- 호중 노트북 발열 문제로 유용상이 라즈베리파이를 직접 받아 SDN 재측정을 이어가기로 결정. AP 기기는 이번 작업엔 불필요, 추후 실시간 추론 파이프라인 구현 시에는 AP+Pi 둘 다 필요함을 논의. 실시간 파이프라인은 현재 어느 브랜치에도 구현되어 있지 않음을 확인
- git: yongsang 브랜치에 총 5개 커밋 push 완료
