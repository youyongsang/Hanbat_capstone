# 발표용 시각 자료 (모델 비교 결과)

`project/results/yongsang/ap_model_comparison_redesign2.{txt,csv}` +
`ap_v2_redesign2_pi_latency_comparison.txt`의 수치를 그래프로 만든 것.
발표 슬라이드·PDF에 바로 넣어 쓰라고 이미지로 저장했다.

| 파일 | 내용 | 발표에서 쓸 곳 |
|---|---|---|
| `01_accuracy_vs_latency` | 정확도 vs Pi INT8 지연 산점도 (5시드 평균, ±1σ) | "정확도는 동급, Proposed가 더 빠르다" 한 장 |
| `02_accuracy_and_f1` | 전체 정확도 · Label 3(심각) F1 막대 (±표준편차) | "정확도 근소차 / SDN은 희소클래스에서 불안정(±8.1)" |
| `03_pi_latency` | Pi INT8 평균 추론 지연 막대 (+1ms 목표선) | "목표2(<1ms) 달성, EE가 Baseline −22%" |
| `04_exit_distribution` | Early Exit 종료 지점 분포 (가로 스택) | "Proposed는 대부분 exit 1~2에서 끝난다 = 속도 이득의 실체" |

## 형식

- **`.svg` (권장)** — 벡터. 아무리 확대해도 안 깨진다. PowerPoint 2016+ / Google 슬라이드 / Keynote / 한컴오피스 전부 "그림 삽입"으로 넣을 수 있고, 슬라이드를 PDF로 내보내도 벡터로 유지된다. 색·글자 수정도 가능(그림 편집).
- **`.png`** — 2배 해상도(1320×760 내외) 래스터. SVG가 안 먹는 도구용 fallback.

## 수치가 바뀌면

`docs/yongsang/model_results.html`가 같은 데이터의 인터랙티브 버전이고, 이 이미지들은
같은 폴더의 `build_figures.js`로 생성했다. 수치 정본이 바뀌면 그 스크립트의 `MODELS`
배열만 고쳐 재생성한다. (Dynamic θ는 배포 체크포인트 값 — 5시드 특성화 대상 아님.)

```bash
node docs/yongsang/figures/build_figures.js docs/yongsang/figures   # .svg 재생성
# PNG는 headless Chrome로: chrome --headless --screenshot=out.png --window-size=W,H fig.svg
```

기준: `class-weight-power=0.0` · 5시드 평균 · Pi INT8 실측(`capstone@192.168.8.109`, test 310창, 2026-08-30).
