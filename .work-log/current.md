# Capstone-Design 현재 상태
최종 업데이트: 2026-08-22 (새벽, yongsang 새 노트북 DESKTOP-29GLQJF 세션)

## 프로젝트 개요
산업 무선망(AP) 트래픽 혼잡을 Early Exit LSTM으로 실시간 분류하고, Raspberry Pi + ONNX/INT8로 엣지 배포하는 캡스톤 프로젝트. 방학 중 교수 피드백에 따라 1학기 4-feature 시뮬레이터 기반에서 실제 GL.iNet AP 실측 9-feature(`ap_metrics_cleaned_strict`) 기반으로 피벗함. 팀: 유용상(모델 설계), 장예나(데이터), 김호중(경량화·배포).

## 완료된 작업 (2026-08-21 밤 ~ 2026-08-22 새벽 세션)

### AP(Opal) 네트워크 문제 원인 파악 및 해결
- 처음엔 AP WiFi SSID(`GL-SFT1200-a08`)에는 연결됐지만, `192.168.8.1` 관리 페이지/SSH가 전혀 안 열리는 문제 발생. IP를 확인해보니 노트북이 Opal 자신이 아니라 **집 공유기(다른 물리 기기)가 나눠준 IP(`192.168.75.x`)**를 받고 있었음
- 원인: Opal의 LAN 포트에 집 인터넷 랜선을 꽂아서 Opal이 브릿지/익스텐더처럼 동작 → WiFi 클라이언트가 Opal이 아닌 상위 공유기 서브넷을 그대로 받음
- 호중에게 확인 결과, 원래 성공했던 구성은 **"집 공유기(LAN) → Opal의 WAN 포트"**로 연결하는 것이었음 (LAN이 아니라 WAN에 꽂아야 Opal이 독립적으로 `192.168.8.0/24`를 WiFi로 뿌리면서 인터넷도 별도로 받음)
- 랜선을 Opal의 WAN 포트로 옮겨 연결 → 노트북이 `192.168.8.226`을 받고, **인터넷도 되고 `192.168.8.1` 관리 페이지도 열림** (Claude Code 세션도 안 끊기고 유지됨)
- 중간에 Opal 본체의 "MODE" 슬라이드 스위치를 만져봤으나 효과 없었음(아마 라우터/AP 모드 전환이 아니라 펌웨어 부팅 슬롯 선택 스위치로 추정) — 원위치로 되돌려둠

### SSH 인증 정리
- **AP(root@192.168.8.1)**: 호중이 알려준 비밀번호로 로그인 성공. 단, `ssh-rsa` 알고리즘을 명시적으로 허용해야 함(`-o HostKeyAlgorithms=+ssh-rsa -o PubkeyAcceptedAlgorithms=+ssh-rsa`) — 오래된 dropbear라 최신 SSH 클라이언트 기본값과 안 맞음
- 이 노트북에 RSA 키(`~/.ssh/id_rsa_ap`, ed25519는 이 dropbear가 거부해서 RSA로 재생성)를 만들어 `ssh-copy-id`로 AP에 등록 → `~/.ssh/config`에 `Host 192.168.8.1` 항목 추가해서 **비밀번호 없이 자동 인증**되도록 설정 완료 (`collect_metrics.py`가 내부적으로 `BatchMode=yes`를 쓰기 때문에 키 인증이 필수였음)
- **Pi(pi@192.168.8.109)**: 호중이 알려준 비밀번호가 안 먹힘(3회 시도 실패). 호중 본인은 "그 비번 맞다"고 하는데 우리 쪽에서 안 됨 — **계정 이름이 `pi`가 아닐 가능성**이 유력. 호중에게 실제 로그인 명령어 전체(아이디 포함)를 그대로 복사해서 보내달라고 요청한 상태, **아직 미해결**
- 라즈베리파이 SD카드 재굽기(Imager 고급설정으로 새 계정 지정)도 대안으로 검토했으나, 프로젝트 파일은 이미 git에 다 커밋되어 있어 손실 위험이 크지 않다는 결론 → 호중 답 기다리는 쪽으로 결정, 아직 안 구움

### `collect_metrics.py` 버그 수정
- **Windows ping 파싱 버그 (수정 완료)**: 기존 코드가 Linux `ping -c/-W` 문법과 `rtt min/avg/max/mdev` 출력 형식만 파싱해서, Windows(특히 한글 로캘)에서 실행하면 latency/jitter/packet_loss가 에러 없이 조용히 전부 0으로 찍히는 버그였음. `platform.system()`으로 분기해서 Windows에서는 `ping -n/-w` 사용 + `TTL=`이 포함된 응답 줄에서 로케일 무관하게 `(숫자)ms` 패턴을 추출하도록 수정 (`project/scripts/collect_metrics.py`)
- **station 재연결 스파이크 버그 (미수정, 팀 공유 필요)**: `connected_clients`가 1→2로 바뀌는 순간(폰 WiFi 절전모드 등으로 station 목록에서 빠졌다 다시 나타날 때), 그 station의 누적 rx/tx bytes·재전송 카운터 전체가 "순간 증가분"으로 잘못 계산되어 throughput이 수백~수천 Mbps로 튀는 현상 확인. 여러 station의 바이트를 그냥 합산하는 `calculate_station_throughput` 로직의 구조적 문제 — 코드 수정은 안 했고, 데이터 후처리 시 `connected_clients` 전환 시점 행을 걸러내는 방식으로 대응 필요

### 실측 데이터 수집 완료
- 이 노트북(WiFi Client, `iperf3` winget으로 설치) + 폰(Termux, `iperf3 -s` 서버, IP `192.168.8.103`) 구성으로 5개 시나리오 라이브 수집
- `normal_idle`(26행) / `low_load`(19행, 20Mbps) / `medium_load`(17행, 50Mbps) / `high_load`(17행, 100Mbps) / `stress_load`(12행, 150Mbps×4병렬) → 총 91행, `project/scripts/metrics_v2.csv` (**아직 git 미커밋**)
- 부하를 올릴수록 UDP 손실률이 뚜렷하게 증가(1.2% → 20.0% → 28.4% → 68.1%)해서 시나리오 설계가 의도대로 작동함을 확인
- 첫 `low_load` 수집 때 프로세스를 늦게 종료해서 157초 중 대부분이 부하 없는 상태로 잘못 라벨링된 데이터 오염 발견 → 삭제 후 깨끗하게(68초 정확히 맞춰서) 재수집함

### 모델 파이프라인 end-to-end 검증 + 중요 발견
- 이 노트북 anaconda base 환경은 기존에 알려진 것과 동일한 torch DLL 로딩 실패 문제 있음 → **새 conda 환경 `capstone` 생성**(`C:\Users\dkssu\anaconda3\envs\capstone`)에 torch(CPU)+pandas+numpy 설치해서 해결
- `prepare_ap_metrics_dataset.py --input project/scripts/metrics_v2.csv --out-dir project/data/ap_metrics_new_collection`로 윈도우 변환 (41 샘플: train 28/val 7/test 6) — **기존 `ap_metrics_cleaned_strict` 폴더는 건드리지 않고 별도 폴더로 생성**
- **중요 발견**: 원본 학습 스케일러(`ap_metrics_cleaned_strict/scaler_params.json`)와 우리 실측 데이터의 실제 범위가 완전히 다름
  - `latency_ms`: 원본 0.047~0.163 vs 실측 2~841 (원본이 ms 단위가 맞는지 의심스러움)
  - `tx_retries_delta`: 원본 최대 23 vs 실측 최대 20만대
  - `rssi_dbm`: 원본 -30~-17(매우 근접 측정) vs 실측 -67~-53.5
  - → 1학기/AP strict 원본 학습 데이터의 측정 방식 자체에 단위 버그가 있거나, 완전히 다른 물리적 실험 조건(매우 가까운 거리)에서 수집됐을 가능성. **예나·팀에 공유 필요**
- `evaluate_ap_early_exit.py`로 `ap_early_exit_lstm_best.pth` 평가 (자체 스케일러 사용, `project/results/yongsang/ap_new_collection_eval_report.txt`에 저장): 전체 정확도 50%(단, test 샘플 6개뿐이라 통계적으로 거의 무의미), Label 0/1(정상/경고)은 100% 정확했지만 **Label 2/3(혼잡/심각)은 0%** — 사전학습된 모델이 이 새로운 측정 환경에 일반화되지 않음을 시사

## 다음 할 일
- [ ] 호중에게 Pi SSH 실제 로그인 명령어(아이디 포함) 받기 — 안 풀리면 SD카드 재굽기 고려
- [ ] Pi 접속되면 `project/deploy/raspberry_pi_ap/` 번들로 SDN FP32/INT8 재측정 (`--mode staged-confidence`, threshold 0.85, `--max-samples 82`) → `analyze_pi_results.py` 분석 → 결과 저장 후 커밋
- [ ] `project/scripts/metrics_v2.csv`, `project/data/ap_metrics_new_collection/`, `project/results/yongsang/ap_new_collection_eval_report.txt`, `collect_metrics.py` Windows ping 수정 git 커밋
- [ ] 스케일러 불일치 발견 사항을 예나·팀에 공유 (원본 latency_ms/tx_retries_delta 측정 방식 재검토 필요)
- [ ] 각 시나리오를 훨씬 오래(몇 분 단위) 반복 수집해서 샘플 수 늘리기 — 지금 41샘플은 통계적 결론 내리기엔 너무 적음
- [ ] `connected_clients` 전환 시점 스파이크를 걸러내는 후처리 로직 추가 논의 (예나 또는 스크립트 수정)
- [ ] 장기적으로 이 실측 방식으로 모델을 새로 학습/파인튜닝할지 팀 논의 필요 (기존 `ap_cleaned_strict` 학습 데이터와 스케일이 안 맞음)
- [ ] (여유 시) AP strict용 실시간 추론 파이프라인 설계 착수 — 현재 어느 브랜치에도 코드 없음

## 주요 파일
- `project/scripts/collect_metrics.py` — AP 라이브 측정 스크립트. AP_IP=`192.168.8.1`, SERVER_IP=`192.168.8.103`(폰)로 갱신됨, Windows ping 파싱 버그 수정 완료
- `project/scripts/metrics_v2.csv` — 이번 세션에서 수집한 원본 실측 데이터 (91행, 5개 시나리오, git 미커밋)
- `project/data/ap_metrics_new_collection/` — 새 실측 데이터 기반 windowed train/val/test (자체 스케일러, `ap_metrics_cleaned_strict`와 별개)
- `project/results/yongsang/ap_new_collection_eval_report.txt` — 새 데이터로 `ap_early_exit_lstm_best.pth` 평가한 리포트
- `~/.ssh/config`, `~/.ssh/id_rsa_ap*` — 이 노트북 로컬 SSH 키/설정 (git에는 없음, 이 기기에서만 유효). AP(`root@192.168.8.1`) 비밀번호 없이 접속 가능
- `C:\Users\dkssu\anaconda3\envs\capstone` — 이 노트북에서 torch(CPU)+pandas+numpy 설치된 conda 환경 (base는 DLL 로딩 실패)
- `project/README_AP_STRICT.md` — AP strict 파이프라인 전체 기준 문서
- `docs/hochung/ap_traffic_measurement_guide.md` — AP 라이브 트래픽 측정 방법 정리 문서
- `project/deploy/raspberry_pi_ap/README.md` — Pi ONNX 8개 조합 재측정 명령어 (아직 미착수, Pi 로그인 대기 중)

## 특이사항 / 결정 사항
- **Opal 포트 구분 중요**: 집 인터넷은 반드시 Opal의 **WAN 포트**에 꽂아야 함. LAN 포트에 꽂으면 Opal이 브릿지 모드처럼 동작해서 자기 관리 IP(`192.168.8.1`)가 WiFi 클라이언트에서 안 열림
- **AP 비번 ≠ Pi 비번**: 호중이 알려준 비밀번호는 `root@192.168.8.1`(AP)엔 맞지만 `pi@192.168.8.109`(Pi)엔 안 맞음 — 팀 내에서 이 둘을 같은 걸로 착각하기 쉬우니 항상 어느 기기 얘기인지 구분할 것
- AP 모델은 GL.iNet Opal(GL-SFT1200), 관리 IP `192.168.8.1`(WAN 포트로 인터넷 연결 시 정상적으로 이 IP로 접속 가능)
- AP dropbear SSH가 오래돼서 `ssh-rsa`/RSA 키만 지원함 (ed25519 거부됨, `HostKeyAlgorithms`/`PubkeyAcceptedAlgorithms` 옵션 필수)
- AP strict `test.csv`(`ap_metrics_cleaned_strict`)는 82개 샘플, 이번에 새로 만든 `ap_metrics_new_collection`은 41개 샘플 — 서로 다른 데이터셋이니 항상 경로 확인할 것
- `torch`가 시스템 기본 python(anaconda base)에서 DLL 로딩 실패하는 문제는 노트북이 바뀌어도(이전 DESKTOP-5A9LEGQ, 이번 DESKTOP-29GLQJF) 계속 재현됨 — 항상 별도 conda 환경에 torch 설치할 것
- Fixed/Dynamic은 같은 체크포인트(`ap_early_exit_lstm_best.pth`)에서 threshold 정책만 다르게 평가하는 것이고, SDN은 반드시 독립 학습해야 공정 비교가 됨
