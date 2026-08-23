# Capstone-Design 현재 상태
최종 업데이트: 2026-08-23 (밤, yongsang DESKTOP-29GLQJF 세션 — Claude Code와 함께 진행)

## 완료된 작업 (2026-08-23 밤 세션, Claude Code와 진행)

### AP 재시도: "다중 station이 크래시 원인"이라는 가설 재검증, 단일 폰으로 label 3 추가 확보
- AP를 몇 시간 쉬게 한 뒤 재시도. **재연결 반복(2~5초마다 iperf3 재접속)이 크래시 원인이라는 가설을 먼저 재검토** — 오늘 저녁 run5(2대, 100Mbps, 재연결 루프 이미 제거된 상태)도 ~22초 만에 크래시했던 기록을 다시 보니, 재연결 루프 유무와 무관하게 **"동시 station 2개 이상"이 실제 변수였을 가능성이 더 크다는 결론으로 정정**
- 폰 1대(`192.168.8.191`)만 단일 연속 iperf3 스트림으로 테스트: 70Mbps 2분 완주(크래시 없음, label 0/1/2만) → 100Mbps 5분 완주(크래시 없음, **label 3 4개 신규 확보**, 그중 2개는 `channel_occupancy_percent=100.0` 완전 포화) → **같은 세션에서 3번째로 20분 시도했다가 2분 41초 만에 크래시** (label 3 1개는 그 전에 추가 확보)
- **새로운 관찰**: 단일 station이 다중 station보다 훨씬 안정적인 건 맞지만 완전히 안전하진 않음 — 휴식 없이 연속으로 여러 번 부하를 주면 단일 station이라도 누적 피로(추정: 열, 원인 미확정)로 결국 크래시함. "재부팅 직후 첫 시도가 제일 잘 버틴다"는 패턴이 오늘 밤 내내 반복 관찰됨(3대 첫 시도 90초 완주, 단일 폰 첫/둘째 시도 각각 2분·5분 완주 후 셋째 시도부터 급격히 나빠짐)
- **다음 시도 권장**: 폰 1대 단일 스트림으로 시작하되, **세션 사이에 AP를 몇 분씩 쉬게 하는 휴식 시간을 의도적으로 넣을 것**. 연속으로 몰아붙이지 말 것.
- 밤 세션 총 수집: label 0 5개 / label 1 22개 / label 2 45개 / **label 3 5개** 추가. `metrics_v2.csv` 1265행 → **1338행**

### 데이터 반영: 재라벨링(변화 없음, 검증) + 재변환 + 재학습
- `relabel_metrics_v2.py` 실행 — 신규 수집분도 이미 최신 가중치로 라벨링돼 있어서 재라벨링 전/후 분포 동일함을 확인(정상 동작 검증)
- `prepare_ap_metrics_dataset.py`로 재변환: train 843→**903**, val 181→**194**, test 179→**191**. label 3은 train 23→**27**, val 5→**6**, test 5→5(동일)
- `power=1.0`(기본값)으로 재학습 → 평가 결과: 전체 정확도 65~66%→**73.3%**, **Label 2 recall 36~42%→70.8%로 크게 개선**, 그런데 **Label 3 recall 40%→20%로 하락**(5개 중 1개만 정답)
- **해석**: test label 3이 여전히 5개뿐이라 1개 차이가 20%p를 좌우함 — 통계적 노이즈일 가능성이 높고, Label 2/3 트레이드오프가 실행마다 흔들리는 걸 보면 표본이 더 늘어야 진짜 경향을 볼 수 있음. 지금 숫자 하나하나에 너무 의미 부여하지 말 것.
- `project/README_AP_V2.md`, `CLAUDE.md` 2차 섹션의 라벨 분포/평가 결과 표를 최신 수치로 갱신함

### congestion_score 가중치 재조정 — AP 재부팅 없이 label 3을 21개→33개로 늘림
- AP가 반복 크래시로 더 이상 데이터를 못 모으는 상황에서, 이미 모아둔 `metrics_v2.csv`(1265행)를 분석해보니 **원래 가중치(throughput 35% / occupancy 35% / retry 20% / jitter 10%)가 실제 변별력과 안 맞았음**을 발견
  - stress_load 구간에서 label 2 vs label 3의 sub-score 평균 비교: `throughput_score` 0.665→0.707(거의 차이 없음), `occupancy_score` 0.449→0.898(거의 2배), `jitter_score` 0.512→0.802(큰 차이), `retry_score` 0.724→0.834(약한 차이)
  - 즉 throughput은 정상/경고를 가르는 덴 유용하지만 혼잡/심각을 가르는 덴 거의 기여를 못 하고 있었음
- `collect_metrics.py`의 `calculate_scores()` 가중치를 **throughput 20% / occupancy 45% / retry 20% / jitter 15%**로 재조정 (occupancy·jitter 비중 상향, throughput 비중 하향)
- 새 가중치를 기존 원시 데이터(1265행)에 그대로 재적용하는 `project/scripts/relabel_metrics_v2.py` 신설 — **AP 재수집 없이** raw 데이터 재라벨링만으로 label 3을 21개→33개로 늘림 (label 0: 166→149, label 1: 516→670, label 2: 562→413)
- `ap_metrics_v2` 재변환: label 3 샘플이 train 14→23, val 3→5, test 3→5로 증가

### class weight power 재실험 — power=1.0으로 확정 (트레이드오프가 완만하지 않고 절벽형)
- `train_ap_early_exit.py`에 `--class-weight-power` CLI 인자 추가(기존엔 하드코딩)
- 재라벨링된 데이터로 power 0.7 / 0.85 / 1.0 비교:

  | power | 전체 정확도 | Label 0 | Label 1 | Label 2 | **Label 3** |
  |---|---:|---:|---:|---:|---:|
  | 0.7 | 85.8% | 95.5% | 86.6% | 88.1% | **0%** |
  | 0.85 | 76.5% | 95.5% | 70.1% | 86.4% | **0%** |
  | 1.0 | 65~66% | 95.5% | 75~77% | 36~42% | **40%** |

- **0.7/0.85는 label 3을 아예 0%로 계속 놓침. 1.0(순수 역빈도)에서만 label 3이 잡히기 시작**하며, 그 대신 label 2 정확도가 크게 하락(88%→36~42%) — 완만한 트레이드오프가 아니라 거의 전부/전무에 가까운 절벽
- 팀 판단으로 **power=1.0을 기본값으로 확정**: 이 프로젝트 목적상 심각 혼잡을 놓치는 것(false negative)이 혼잡을 심각으로 과잉 경고하는 것(false positive)보다 더 치명적이라는 이유
- 최종 체크포인트: `project/checkpoints/ap_v2/`(power=1.0), 평가 리포트: `project/results/yongsang/ap_v2_eval_report.txt`
- **주의**: 이 재조정으로 `ap_metrics_v2`의 label 정의(congestion_score 가중치)가 production `ap_cleaned_strict`(588행, 옛 가중치 0.35/0.35/0.20/0.10)와 달라짐. 두 데이터셋을 같은 기준으로 비교하면 안 됨 — `ap_cleaned_strict`를 새 가중치로 재라벨링할지는 아직 미결정(팀 논의 필요)

## 완료된 작업 (2026-08-23 저녁 세션, Claude Code와 진행)

### AP(Opal GL-SFT1200)가 다중 station 부하에서 반복적으로 크래시됨 — 중요 하드웨어 이슈 발견
- 폰 3대(103/191/221) 동시 부하로 label 3 데이터를 더 모으려고 시도. 오늘 Claude Code가 노트북에서 iperf3 부하를 직접 걸고 `collect_metrics.py`를 동시 실행하는 방식으로 자동화 시도
- **1차(3대, 40Mbps 각각, 90초)는 완주 성공** — label 3 1개 포함 6개 샘플 확보(채널 100% 근처 포화, retry 24508/25836, congestion_score 0.756)
- 이후 **총 4번 더 시도했으나 전부 AP 크래시로 조기 중단**: 3대/40Mbps/10분(~40초에 폰 2대 연결 끊김), 3대/25Mbps(~18초), **심지어 예전에 검증됐던 2대/100Mbps 조합조차 ~22초 만에 크래시**(죽기 직전 1.5~1.7 Gbps라는 물리적으로 불가능한 카운터 스파이크 관측 — WiFi 드라이버/라디오 재시작 추정)
- 크래시마다 AP의 WiFi SSID(`GL-SFT1200-a08`) 자체가 사라지고 노트북이 다른 아는 네트워크(`192.168.45.x`)로 자동 전환됨 → 매번 AP 전원을 물리적으로 껐다 켜야 복구됨 (총 4회 재부팅)
- **핵심 관찰: 부하를 낮춰도(40M→25M), station 수를 줄여도(3대→검증된 2대 조합) 크래시가 오히려 더 빨리 발생함(90초 완주 → 40초 → 18초 → 22초)** — 이는 부하 세팅 문제가 아니라 **반복된 크래시-재부팅 사이클 자체가 AP를 점점 더 불안정하게 만들었을 가능성**을 시사함 (열, 메모리 누수, 펌웨어 상태 꼬임 등 추정, 원인 미확정)
- 사용자 판단으로 오늘은 여기서 중단, AP를 충분히 쉬게 한 뒤 재시도하기로 결정
- **폰 `192.168.8.103`이 세션 중간에 오프라인 상태**가 되어(iperf3 서버 응답 없음, ping도 unreachable) `collect_metrics.py`의 하드코딩된 `SERVER_IP`를 `192.168.8.191`로 임시 변경함 (지연시간 측정 대상 폰 교체, 커밋됨)
- **총 성과**: 크래시 반복에도 불구하고 실측 데이터 12행 추가(label 0×4, 1×1, 2×6, **3×1**) → `metrics_v2.csv` 1254행→**1266행**. 다음 세션에서 `prepare_ap_metrics_dataset.py`로 재변환 + 재학습 필요 (아직 안 함, 증가폭이 작아서 우선순위는 낮음)

## 완료된 작업 (2026-08-23 오후 세션, Claude Code와 진행)

### Pi를 TV(HDMI)에 연결, SSH 키 인증 설정 완료 — 새벽 세션의 "최우선" 항목 해결
- TV에 HDMI로 연결해서 부팅 화면 확인 → 정상 부팅되어 `capstone@CapsTone:~ $` 프롬프트까지 뜸 (SD카드 접촉 불량 의심은 해소된 것으로 보임)
- Pi를 와이파이(`192.168.45.x` 대역, 노트북과 같은 네트워크)에 새로 연결. `hostname -I`로 IP 확인 → `192.168.45.31`
- 이 노트북의 `~/.ssh/id_ed25519` 공개키를 Pi의 `~/.ssh/authorized_keys`에 등록해서 비밀번호 없이 `ssh capstone@192.168.45.31` 접속 가능해짐 (AP용 `id_rsa_ap`와는 별개 키)

### `project/deploy/raspberry_pi_ap/` 번들로 Pi 실측 완료 (단, 구버전 588행 `ap_cleaned_strict` 데이터 기준)
- 번들 전체를 scp로 Pi에 전송, `.venv`에 onnxruntime 1.29.0 설치, README의 8개 조합(Baseline/SDN/Fixed/Dynamic × FP32/INT8) 전부 실행 완료
- 정확도는 PC 평가와 완전히 일치(Baseline 92.7%, 나머지 91.5%) — 동일 checkpoint/scaler 확인됨
- **핵심 발견**: PC에서는 안 보이던 Early Exit 속도 우위가 Pi + staged ONNX 실측에서는 실제로 나타남. Proposed Dynamic FP32(1.699ms 평균)가 Baseline FP32(1.837ms)보다 7.5% 빠름. Dynamic theta의 Exit1 비율(37.8%)이 Fixed(15.9%)보다 훨씬 높은 게 원인
- 결과 저장: `project/results/yongsang/pi_ap_measurements/` (원시 CSV 56개 + `pi_ap_measurement_summary.md` 요약표)
- **주의**: 이 실측은 `ap_cleaned_strict`(588행) 파이프라인 기준이고, 아래 `ap_metrics_v2`(1253행, 실제 최신 데이터)과는 별개 — 아직 새 데이터 기준 ONNX/Pi 번들은 없음

### `ap_metrics_v2` 데이터셋이 낡은 스냅샷 기준이었던 버그 발견 및 수정
- 어젯밤 커밋(`bef79fd`, "1254행으로 확장")의 커밋 메시지는 "재생성 완료(train 707/val 151/test 152)"라고 했지만, 실제 `conversion_report.txt`를 까보니 **`raw_rows: 1060`** — 즉 `metrics_v2.csv`가 1253행까지 다 채워지기 *전* 스냅샷으로 변환 스크립트를 돌리고, 그 뒤 193행을 추가 수집한 걸 같은 커밋에 묶어 올린 것으로 보임
- `prepare_ap_metrics_dataset.py --input project/scripts/metrics_v2.csv --out-dir project/data/ap_metrics_v2 --overwrite`로 최신 1253행 전체 기준 재변환 → **train 843 / val 181 / test 179** (raw_rows 1253으로 일치 확인). label 3(심각)도 train 14/val 3/test 3으로 소폭 증가(이전 9/2/2)

### 클래스 가중치 완화 실험 (새벽 세션에 남겨둔 "다음 할 일" 항목 해결)
- `train_ap_early_exit.py`의 `compute_class_weights`가 순수 역빈도(`N/(K*count)`, power=1.0)만 지원해서 label 3에 ~20배 가중치가 붙어 label 2를 3으로 오버슈팅하던 문제(label 2 recall 19%)를 재현할 것으로 예상됨
- `power` 파라미터를 추가해서 `(N/(K*count))**power` 형태로 일반화. power=0.5(sqrt)로 재학습 → label 2 recall 19%→79%로 크게 개선됐지만 label 3 recall이 0%로 떨어짐(트레이드오프가 반대로 과함)
- **power=0.7로 재조정**(현재 값) → val balanced acc 62.8%(0.5일 때 58.3%보다 나음), test 전체 정확도 74.3%, label 0=77.3%/label 1=84.9%/label 2=66.7%/**label 3=0%**
- label 3는 test 샘플이 3개뿐이라 가중치를 아무리 조정해도 한계가 있음 — 알고리즘이 아니라 데이터 부족 문제. 다음 수집에서 label 3(채널 100% 포화) 샘플을 더 확보해야 근본 해결됨
- 재학습된 체크포인트: `project/checkpoints/ap_v2/`(검증용, production `ap_cleaned_strict`와는 별개 그대로 유지)
- 평가 리포트: `project/results/yongsang/ap_v2_eval_report.txt`

## 완료된 작업 (2026-08-23 새벽 세션, Claude Code와 진행)

### Pi SSH 로그인 미스터리 해결: 계정명은 "capstone", "CapsTone" 정체도 파악
- 라즈베리파이가 Opal LAN 포트에 유선 연결되어 있다는데 station dump/DHCP/ARP 어디에도 안 잡히는 문제를 오래 추적함
- SD카드를 노트북에 꽂아 boot 파티션(`bootfs`, FAT32)의 cloud-init 설정을 직접 확인 → **계정명이 `pi`가 아니라 `capstone`**이었음이 확인됨 (지난 세션의 추측이 맞았음). `hostname: CapsTone`으로 설정되어 있는 것도 확인
- **중요한 재해석**: 어젯밤(8/22) 극한 부하 테스트에서 "새 공기계"로 추가했던 `192.168.8.109`(hostname "CapsTone")가 사실 새 폰이 아니라 **이 라즈베리파이 자체**였음을 MAC 벤더 조회(`d8:3a:dd:48:55:97` → Raspberry Pi Trading Ltd, API로 검증)로 확인. 즉 어젯밤엔 파이가 이더넷으로 정상 연결되어 9분 넘게 iperf3 트래픽까지 잘 주고받았음
- 오늘은 케이블/SD카드를 여러 번 재연결해도 Opal 쪽에서도, SD카드에 설정된 백업 Wi-Fi(`SK_0600_5G`, 집 공유기)에서도 전혀 안 잡힘. 중간에 초록 활동 LED가 "깜빡이다 꺼짐" 현상 관찰 → SD카드 접촉 불량으로 인한 boot 실패 패턴으로 추정
- SD카드의 boot 파티션(FAT32)은 노트북에서 읽히지만, 이건 OS가 있는 루트 파티션(ext4, 노트북에서 못 읽음)이 멀쩡하다는 뜻은 아님 — boot는 되지만 그 다음 단계에서 막히는 것과 완전히 다른 문제
- **결론: 잠정 보류.** TV에 직접 연결해서 화면으로 부팅 진행 상황을 봐야 확실해짐 (오늘은 시간 관계상 보류, 다음 세션 과제)
- 참고: `network-config`(netplan) 내용 — `eth0: dhcp4 true, optional true` / `wlan0: SK_0600_5G에 자동연결, optional true`. `user-data`(cloud-init) — 계정 `capstone`, `enable_ssh: true`, `ssh_pwauth: true`, `avahi-daemon` 설치됨. **비밀번호는 해시(yescrypt)라 SD카드에서 평문 확인 불가** — 호중에게 "capstone 계정" 비밀번호로 다시 확인 필요

### 데이터 수집 계속 확대 (834행 → 1254행) — 진짜 다중 station 경합으로 label 3 대폭 확보
- 어제 세션 마지막에 발견한 문제("노트북이 발신 허브라 진짜 다중 station 경합이 아니었다")를 해결: 폰을 2대(`192.168.8.103`, 나중엔 새 폰 `192.168.8.191`) 동시에 사용해서 각각 독립적으로 100~120Mbps UDP 부하를 걸어 진짜 2-station 경합 재현
- `high_load` 단일기기로 10분 추가 수집(244행), 2폰 동시 부하로 `stress_load`에 2회 추가 수집(총 523행까지), `medium_load`도 폰 하나로 보충(187행) → 최종 **1254행**
- 라벨 재계산 결과 **label 3(심각) 4개 → 13개**로 대폭 증가 (채널 100% 포화 순간이 훨씬 자주 잡힘). val/test에도 각각 2개씩 포함되어 처음으로 통계적으로 유의미한 평가가 가능해짐
- 재학습 후 confusion matrix 확인: **모델이 이제 4개 클래스를 전부 예측함**(전엔 label 2/3을 아예 안 찍었음). label 3 정확도 0%→50%(2개 중 1개), val balanced accuracy 54.4%→69.4%
- **새로 드러난 부작용**: label 3에 준 클래스 가중치가 너무 세서(train 9개뿐이라 가중치 ~20배) label 2(혼잡)를 label 3으로 오버슈팅하는 경향 생김(label 2 정확도 19%로 하락, 58개 중 33개를 3으로 오분류). 다음에 가중치를 완화(역빈도 대신 제곱근 역빈도 등)하면 다듬을 수 있을 것으로 보임
- 폰 iperf3 서버가 화면 꺼짐으로 중간에 두 번 죽음(`termux-wake-lock` 안 걸어둔 폰들) → 재시작 대기 중 세션 일시 중단, 여기서 기록 저장

## 다음 할 일 (2026-08-23 밤 세션 기준 갱신)
- [x] 파이를 TV(HDMI)에 연결해서 부팅 화면 확인 — 정상 부팅, `capstone@CapsTone` 프롬프트 뜸
- [x] SSH 키 인증 설정 완료 (`ssh capstone@192.168.45.31` 비밀번호 없이 접속 가능)
- [x] `project/deploy/raspberry_pi_ap/` 번들로 8개 조합 Pi 실측 완료 (단, 구버전 588행 데이터 기준 — 위 섹션 참고)
- [x] 와이파이를 Opal AP로 전환, 폰 3대(103/191/221) 붙여서 label 3 추가 수집 시도 — 12행 확보(label 3 1개 포함)했지만 AP가 4번 크래시함
- [x] congestion_score 가중치 재조정(occupancy/jitter 상향, throughput 하향) + 기존 데이터 재라벨링으로 label 3 21→33개 확보 (AP 재수집 없이)
- [x] class weight power 재실험 및 확정 — **power=1.0**(순수 역빈도)을 기본값으로 채택
- [x] AP 재시도(폰 1대 단일 스트림) — "다중 station이 크래시 원인" 가설 재확인(2대 이상은 재연결 유무와 무관하게 크래시), 100Mbps로 5분 완주해서 label 3 4개 확보. 단, 휴식 없이 반복하면 단일 station도 결국 크래시함(3번째 연속 시도 2분 41초 실패)
- [x] 신규 수집분 반영 — 재라벨링(검증됨, 변화 없음) + `ap_metrics_v2` 재변환(train 903/val 194/test 191, label 3 27/6/5) + 재학습. 결과: 전체 정확도 73.3%, Label 2 70.8%(개선), **Label 3 20.0%(하락, test 5개뿐이라 노이즈로 추정)**
- [ ] **최우선(다음 세션)**: AP를 몇 시간 이상 충분히 쉬게 한 뒤, 폰 1대 단일 스트림으로 재개하되 **세션 사이에 의도적으로 휴식 시간을 넣을 것**(오늘처럼 연속으로 몰아붙이지 말 것). label 3 test가 아직 5개뿐이라 recall 수치가 세션마다 크게 흔들림 — 표본을 10개 이상으로 늘리는 게 목표
- [ ] **`ap_cleaned_strict`(production, 588행)를 새 congestion_score 가중치로 재라벨링할지 팀 논의 필요** — 지금 `ap_metrics_v2`과 `ap_cleaned_strict`가 서로 다른 라벨 기준(가중치)을 쓰고 있어서 두 데이터셋을 같은 기준으로 비교하면 안 됨
- [ ] `192.168.8.103` 폰이 오프라인됐던 원인 확인 (Termux 강제종료 추정) — 배터리 최적화 예외 설정 확인 권장, 복귀하면 `collect_metrics.py`의 `SERVER_IP`를 다시 `103`으로 되돌릴지 `191` 유지할지는 팀 편의대로
- [ ] `ap_metrics_v2`(1338행) 기준 ONNX export + Pi 배포 번들은 아직 없음
- [ ] `ap_metrics_v2`(1253행) 기준 ONNX export + Pi 배포 번들은 아직 없음 — `ap_cleaned_strict`용 `export_onnx_ap.py`/`prepare_pi_bundle_ap.py`를 새 데이터 경로로 재사용할지, 별도 스크립트를 만들지 결정 필요
- [ ] 호중에게 "capstone 계정" 비밀번호 재확인은 더 이상 불필요 (SSH 키 인증으로 해결됨), 대신 이 사실 공유
- [ ] label 1/2 경계(congestion_score 0.50 부근) 재설계 논의 — threshold 재조정 또는 feature 추가 필요할 수 있음
- [ ] 스케일러 불일치 발견 사항을 예나·팀에 공유 (원본 `ap_cleaned_strict`의 latency_ms/rssi_dbm 측정 방식 재검토 필요)
- [ ] 장기적으로 이 실측 방식으로 모델을 새로 학습/파인튜닝해서 `ap_cleaned_strict`를 대체할지, 팀 논의 필요
- [ ] (여유 시) AP strict용 실시간 추론 파이프라인 설계 착수 — 현재 어느 브랜치에도 코드 없음


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
- `prepare_ap_metrics_dataset.py --input project/scripts/metrics_v2.csv --out-dir project/data/ap_metrics_v2`로 윈도우 변환 (41 샘플: train 28/val 7/test 6) — **기존 `ap_metrics_cleaned_strict` 폴더는 건드리지 않고 별도 폴더로 생성**
- **중요 발견**: 원본 학습 스케일러(`ap_metrics_cleaned_strict/scaler_params.json`)와 우리 실측 데이터의 실제 범위가 완전히 다름
  - `latency_ms`: 원본 0.047~0.163 vs 실측 2~841 (원본이 ms 단위가 맞는지 의심스러움)
  - `tx_retries_delta`: 원본 최대 23 vs 실측 최대 20만대
  - `rssi_dbm`: 원본 -30~-17(매우 근접 측정) vs 실측 -67~-53.5
  - → 1학기/AP strict 원본 학습 데이터의 측정 방식 자체에 단위 버그가 있거나, 완전히 다른 물리적 실험 조건(매우 가까운 거리)에서 수집됐을 가능성. **예나·팀에 공유 필요**
- `evaluate_ap_early_exit.py`로 `ap_early_exit_lstm_best.pth` 평가 (자체 스케일러 사용, `project/results/yongsang/ap_v2_mismatched_scaler_diagnostic.txt`에 저장): 전체 정확도 50%(단, test 샘플 6개뿐이라 통계적으로 거의 무의미), Label 0/1(정상/경고)은 100% 정확했지만 **Label 2/3(혼잡/심각)은 0%** — 사전학습된 모델이 이 새로운 측정 환경에 일반화되지 않음을 시사

## 완료된 작업 (2026-08-22 밤 세션, Claude Code와 진행)

### station 재연결 스파이크 버그 실제 수정 (이전 세션엔 "미수정"으로 남아있던 것)
- 원인: `parse_station_info`가 연결된 모든 station의 누적 rx/tx bytes·재시도 카운터를 그냥 합산해서 반환 → 어떤 station이 station dump에서 잠깐 빠졌다 재등장하면 그 station의 전체 누적값이 "한 폴링 주기 증가분"으로 잘못 계산되어 throughput이 수천 Mbps로 튀는 버그였음
- 수정: station을 MAC 주소별로 개별 추적하도록 변경(`parse_station_info`가 dict 반환), `calculate_station_deltas` 신설 — 직전 폴링에 없던(방금 나타난) station은 이번 폴링 델타를 0으로 스킵. 시뮬레이션 테스트 + 실측(약 45분 연속 수집) 양쪽에서 재현 안 됨 확인 (`project/scripts/collect_metrics.py`)
- 커밋: `126c782`

### 데이터 재수집 (67행 → 636행 → 834행, 3단계)
1. 버그 수정 스크립트로 5개 시나리오 짧게(각 60~90초) 재수집 → 67행, 커밋 `d328115`
2. "샘플이 너무 적다"는 판단 하에 5개 시나리오를 각 9~10분씩 재수집 → 636행. `connected_clients`가 9분 내내 안정적으로 유지되고 스파이크 없음을 재검증
3. 2대 동시 부하(아래 항목)로 stress_load에 197행 추가 → 최종 **834행**

### congestion_score 계산식 재보정 (`JITTER_MAX_MS`, `RETRY_FAILED_MAX`)
- 기존 `JITTER_MAX_MS=1.0`, `RETRY_FAILED_MAX=100.0`은 시뮬레이터 데이터 기준값이라 실측 AP 데이터(jitter 수백ms, retry 수천~수만)에서 `jitter_score`/`retry_failed_score`가 거의 항상 1.0으로 clamp됨 → label 2(혼잡)로 66% 쏠리는 문제 확인
- 실측 분포 p90 근처로 재보정(`JITTER_MAX_MS=300.0`, `RETRY_FAILED_MAX=25000.0`) → saturation 문제는 해결됐으나, congestion_score 자체가 0.25~0.55 구간에 몰려있어 label 3(≥0.75) 문턱을 못 넘는 문제가 새로 드러남 → **채널 100% 포화 같은 진짜 극단 조건이 있어야 label 3이 실제로 나온다**는 결론

### 2대 동시 부하로 진짜 다중 station 혼잡 재현 시도
- 처음엔 노트북 하나로 iperf3 `-P 4`(다중 논리 스트림)를 시도했으나 폰 서버 프로세스가 병목이 되어 오히려 약해짐 → **물리적으로 다른 기기**가 필요하다는 결론
- 공기계(Termux+iperf3, `192.168.8.235`) 추가 확보 → 노트북에서 두 폰으로 각각 150Mbps UDP 동시 발사 → **AP(Opal)가 58초만에 크래시, WiFi SSID 자체가 완전히 사라짐**(재부팅 필요)
- 재부팅 후 100M×2로 재시도 성공 — 560초 전부 완료, AP 생존, `channel_occupancy_percent=100.0`(완전 포화) 순간을 포착해 **진짜 label 3(심각) 샘플 3개 신규 확보**(기존 1개 포함 총 4개)
- 이후 새 공기계(`192.168.8.109`, "CapsTone")로 교체 진행. **중요 인사이트**: 지금까지 구성은 "노트북 1대가 발신 허브로 두 폰에 동시 전송"이라 실제로는 노트북의 단일 업링크가 병목일 수 있음 — 진짜 독립적인 다중 station 경합을 만들려면 폰↔폰 직접 전송이나 노트북을 수신측(iperf3 -s)으로 추가하는 식으로 트래픽 발신원 자체를 분산해야 함 (다음 세션 과제)

### 모델 학습 파이프라인의 근본 버그 발견 및 수정: class imbalance로 인한 완전한 클래스 붕괴
- 체크포인트 불일치(스케일러 다른 모델로 새 데이터 평가) 때문에 정확도가 낮다는 가설을 직접 검증: 새 데이터로 처음부터 재학습 → 39.3% → 79.8%로 확인, 가설 맞음
- 그런데 재학습해도 label 2(혼잡)가 여전히 0%로 나와서 confusion matrix를 직접 뽑아봄 → **모델이 label 2/3을 단 한 번도 예측하지 않는 완전한 class collapse** 확인 (`actual 2: [0, 30, 0, 0]`)
- 원인 1: `multi_exit_loss`(`project/models/early_exit_lstm.py`)가 클래스 비율을 전혀 반영 안 하는 순수 `F.cross_entropy` → `class_weights` 파라미터 추가(옵션, 기본값 None이라 다른 호출부(`train_early_exit.py`, `train_ap_sdn.py`)는 영향 없음)
- 원인 2 (더 결정적): `train_ap_early_exit.py`의 체크포인트 저장 기준이 raw val accuracy였음 — val set이 label 0+1로 74% 쏠려있어서, class-weighted loss로 학습해도 "다수 클래스만 찍어서 raw acc가 우연히 높은 에폭"이 선택되고 있었음 → **balanced accuracy(클래스별 recall 평균) 기준으로 체크포인트 선택하도록 변경**
- 결과: label 2 정확도 0% → 60.0%로 개선 (전체 정확도는 69.5%→57.6%로 하락했지만, 이는 "다수 클래스 찍기로 만든 가짜 높은 점수"가 없어진 것이라 더 정직한 수치). label 3은 train 2개/test 1개뿐이라 가중치를 줘도 여전히 학습 불가 — 데이터 자체가 부족한 문제라 알고리즘으로 해결 안 됨
- 검증용 체크포인트: `project/checkpoints/ap_v2/` (프로덕션 `ap_cleaned_strict` 체크포인트는 안 건드림)

## 주요 파일
- `project/scripts/collect_metrics.py` — AP 라이브 측정 스크립트. station 재연결 스파이크 버그 수정 완료, congestion_score 임계값(`JITTER_MAX_MS`, `RETRY_FAILED_MAX`) 재보정 완료
- `project/scripts/metrics_v2.csv` — 실측 데이터 누적본 (834행, 5개 시나리오, 2대 동시 부하 포함)
- `project/data/ap_metrics_v2/` — 새 실측 데이터 기반 windowed train/val/test (자체 스케일러, `ap_metrics_cleaned_strict`와 별개, train 548/val 118/test 118)
- `project/results/yongsang/ap_v2_mismatched_scaler_diagnostic.txt` — 기존(스케일러 다른) 체크포인트로 새 데이터 평가한 리포트
- `project/results/yongsang/ap_v2_eval_report.txt` — 새 데이터로 처음부터 학습한 체크포인트 평가 리포트
- `project/checkpoints/ap_v2/` — 새 데이터 전용 검증용 체크포인트 (class-weighted + balanced-accuracy 선택 적용)
- `project/models/early_exit_lstm.py` — `multi_exit_loss`에 옵션 `class_weights` 파라미터 추가
- `project/scripts/train_ap_early_exit.py` — inverse-frequency 클래스 가중치 계산(`compute_class_weights`) + balanced accuracy 기준 체크포인트 선택 추가
- `~/.ssh/config`, `~/.ssh/id_rsa_ap*` — 이 노트북 로컬 SSH 키/설정 (git에는 없음, 이 기기에서만 유효). AP(`root@192.168.8.1`) 비밀번호 없이 접속 가능
- `C:\Users\dkssu\anaconda3\envs\capstone` — 이 노트북에서 torch(CPU)+pandas+numpy 설치된 conda 환경 (base는 DLL 로딩 실패)
- `project/README_AP_STRICT.md` — AP strict 파이프라인 전체 기준 문서
- `docs/hochung/ap_traffic_measurement_guide.md` — AP 라이브 트래픽 측정 방법 정리 문서
- `project/deploy/raspberry_pi_ap/README.md` — Pi ONNX 8개 조합 재측정 명령어 (아직 미착수, Pi 로그인 대기 중)

## 특이사항 / 결정 사항
- **AP(Opal)가 과도한 부하에서 완전히 크래시될 수 있음**: 150Mbps×2(합계 300M) 동시 부하에서 58초 만에 WiFi 자체가 완전히 죽음(SSID 방송 중단), 물리적 재부팅 필요했음. 100M×2(합계 200M)는 9분 넘게 안정적으로 버팀 — 이 AP로 극한 테스트할 땐 200M대에서 시작해서 조심스럽게 올릴 것
- **iperf3 UDP `-b` 타겟은 실제 전달량과 다름**: 단일 스트림이든 2개 스트림이든, 이 AP의 실제 물리 채널 용량은 대략 35~50Mbps대에서 포화되는 것으로 보임(타겟을 100M로 걸든 150M로 걸든 실제 전달량은 비슷). 부하를 더 세게 걸고 싶으면 타겟 숫자보다 "몇 대가 동시에 붙어있는지"가 더 중요함
- **폰 iperf3 서버 화면 꺼짐 대응**: `termux-wake-lock` 먼저 실행해두고 `iperf3 -s` 띄우는 걸 권장 (화면 꺼지면 서버 죽을 위험)
- **Opal 포트 구분 중요**: 집 인터넷은 반드시 Opal의 **WAN 포트**에 꽂아야 함. LAN 포트에 꽂으면 Opal이 브릿지 모드처럼 동작해서 자기 관리 IP(`192.168.8.1`)가 WiFi 클라이언트에서 안 열림
- **AP 비번 ≠ Pi 비번, Pi 계정명은 `capstone`**: 호중이 알려준 비밀번호는 `root@192.168.8.1`(AP)엔 맞지만 Pi엔 안 맞음. SD카드 cloud-init 설정 확인 결과 **Pi 계정명은 `pi`가 아니라 `capstone`**(hostname `CapsTone`) — 로그인 시도할 땐 `capstone@<Pi IP>`로 해야 함
- **주의**: 어젯밤(8/22) "새 공기계"로 착각하고 극한 부하 테스트에 썼던 `192.168.8.109`(hostname "CapsTone")는 사실 폰이 아니라 **라즈베리파이 그 자체**였음(MAC `d8:3a:dd:48:55:97` → Raspberry Pi Trading Ltd 벤더 조회로 확인). 그 세션에선 파이가 이더넷으로 정상 동작했었음 — 즉 하드웨어 자체는 멀쩡했던 적이 있으므로, 지금 안 잡히는 건 케이블/SD카드 접촉 문제일 가능성이 하드웨어 고장보다 높음
- AP 모델은 GL.iNet Opal(GL-SFT1200), 관리 IP `192.168.8.1`(WAN 포트로 인터넷 연결 시 정상적으로 이 IP로 접속 가능)
- AP dropbear SSH가 오래돼서 `ssh-rsa`/RSA 키만 지원함 (ed25519 거부됨, `HostKeyAlgorithms`/`PubkeyAcceptedAlgorithms` 옵션 필수)
- AP strict `test.csv`(`ap_metrics_cleaned_strict`)는 82개 샘플, 이번에 새로 만든 `ap_metrics_v2`은 41개 샘플 — 서로 다른 데이터셋이니 항상 경로 확인할 것
- `torch`가 시스템 기본 python(anaconda base)에서 DLL 로딩 실패하는 문제는 노트북이 바뀌어도(이전 DESKTOP-5A9LEGQ, 이번 DESKTOP-29GLQJF) 계속 재현됨 — 항상 별도 conda 환경에 torch 설치할 것
- Fixed/Dynamic은 같은 체크포인트(`ap_early_exit_lstm_best.pth`)에서 threshold 정책만 다르게 평가하는 것이고, SDN은 반드시 독립 학습해야 공정 비교가 됨
