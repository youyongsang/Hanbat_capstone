# ONNX Early Exit 배포 재설계 — staged(세션 3개) → 단일 그래프(If 노드)

2026-08-28 밤 세션. `ap_metrics_v2_redesign2`(6-feature, 본수집 2115행) 모델을 처음 Raspberry Pi에 배포하면서 겪은 문제와 해결 과정을 기록한다.

> **(업데이트 — 2026-08-29 확인, 1차)** 이 문서의 수치는 균등 exit-loss 가중치(0.3/0.3/0.4)로 학습한 체크포인트 기준이다. 2026-08-29 세션에서 `ap_v2_redesign2`의 기본 EE 체크포인트를 SDN 스타일 가중치(0.15/0.30/0.55)로 재학습한 버전으로 교체했다 — **staged→unified→INT8이라는 이 문서의 방법론적 결론(그래프 구조가 원인, INT8 stage별 양자화 후 재조립)은 그대로 유효**하지만, 실제 latency 숫자는 바뀌었다(같은 세션 실측 기준 Fixed 0.595ms/Dynamic 0.591ms, baseline 아키텍처 0.747ms·SDN-style 아키텍처 0.615ms보다도 빠름 — 정확도까지 함께 개선됨). 아래 수치는 그 시점 기록으로 보존한다.
>
> **(업데이트 — 2026-08-29 확인, 2차, 1차 정정)** 같은 날 후속 세션에서 1차 업데이트의 SDN 가중치 승격 자체가 다중 시드 검증 결과 노이즈였을 가능성이 높다고 확인됨(단일 실행 비교였음 — 이 파이프라인엔 그때까지 랜덤 시드 고정이 아예 없었다) → Proposed는 균등 가중치로 되돌림. 대신 같은 세션에 `sta_tx_bitrate_mean`을 7번째 입력 feature로 승격(다중 시드로 검증된 진짜 신호)하면서 Baseline·SDN·Proposed 전부 재학습·재수출·Pi 재측정함 — 최신 Pi 실측은 Baseline 0.756ms/SDN 0.636ms/Proposed Fixed 0.641ms/Proposed Dynamic 0.645ms로, 이번엔 SDN-style이 속도·Label3 F1에서 근소 우위(Proposed는 recall이 근소 우위) — "Proposed가 전면 우위"라는 1차 업데이트의 결론은 성립하지 않는다. 최신 수치·전체 경위는 `.work-log/current.md`의 2026-08-29 항목과 `project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt`를 따른다. 아래 본문 수치는 여전히 최초(균등 가중치, 6-feature) 시점 기록으로 보존한다.
>
> **(업데이트 — 2026-08-30, class-weight-power=0.0 승격 + SDN 논문 충실 재구현)** **이 문서의 방법론적 결론(staged가 세션 호출 오버헤드로 느림 → unified If 노드 → INT8은 stage별 양자화 후 재조립)은 그대로 유효.** 그 뒤 바뀐 것: ①`--class-weight-power` 기본값 1.0→0.0 (재스윕에서 정확도·Label3 F1 둘 다 최고), 세 모델 재학습·5시드 특성화, Baseline 배포 seed0→seed3 교체. ②SDN 비교모델을 Kaya et al.(ICML 2019) 논문대로 재구현(pooling IC + 램프 depth-weighted loss + val 캘리브레이션 T, base 백본만 공유). ③Pi INT8 실측 (test 310창, ⚠ window 10): Baseline 0.746ms / SDN 0.572ms / Proposed Fixed 0.540ms / Dynamic 0.555ms — 전부 <1ms. ④**2026-09-01 window 10→12 승격** (EE 정확도 90.7→91.9~92.0%, Label3 F1 분산 전 모델 절반 이하). ONNX 6개 스크립트 `WINDOW_SIZE` 파라미터화 → `[1,12,7]` 재수출. 로컬 parity(test 309창): EE unified fp32 = PyTorch 309/309, INT8 v2 = 308/309, Baseline INT8 309/309, SDN INT8 305/309. **Pi latency 재측정만 대기(Pi 오프라인)**. 최신 경위: `.work-log/current.md` 15차, `docs/yongsang/model_results.html`.

## 배경

Early Exit LSTM은 학습 시 `EarlyExitLSTM.forward()`가 3개 exit(lstm1/2/3 각각 뒤에 분류기)의 로짓을 전부 반환하고, 추론 시엔 `infer_batch_stepwise()`가 앞 exit의 entropy가 임계값(θ) 아래면 뒤 레이어를 계산하지 않고 즉시 반환한다. 이 "레이어를 실제로 건너뛴다"는 동작을 파이 실기기에서도 재현하려면 ONNX로 내보낼 때 그 조건부 실행을 어떻게 표현할지가 문제였다.

## 1차 시도: staged export (세션 3개로 분리)

1학기(`yongsang` 브랜치, 4-feature) 때부터 써온 방식을 그대로 재사용했다(`project/scripts/export_onnx_ap.py`, `git show yongsang:project/scripts/export_onnx_ap.py`로 원본 참고해서 6-feature용으로 재작성). 모델을 stage1(lstm1+classifier1) / stage2(lstm2+classifier2) / stage3(lstm3+classifier3) 세 개의 독립된 ONNX 그래프로 쪼갠다. 파이 쪽 추론 스크립트(`project/deploy/raspberry_pi_ap_v2/inference_pi_ap.py`)가 stage1을 먼저 돌리고, entropy가 θ₁보다 크면 stage2를, θ₂보다 크면 stage3를 이어서 돌린다 — 뒤 stage를 아예 실행 안 하는 방식으로 "skip"을 구현한다.

파이(Raspberry Pi, `capstone@192.168.8.109`, onnxruntime 1.26.0 CPU)에서 실측한 결과:

| 방법 | avg 지연 | exit1/2/3 비율 |
|---|---:|---|
| Baseline(전체 그래프 1회, 매번 3개 exit 다 계산) | **1.966ms** | 0/0/100% |
| Fixed θ (staged, 세션 최대 3개 순차 호출) | 2.337ms (**+19%**) | 29.7/11.6/58.7% |
| Dynamic θ (staged) | 2.189ms (**+11%**) | 30.0/22.3/47.7% |

**Early Exit이 baseline보다 느렸다.** exit point별로 쪼개보면 원인이 드러난다: exit1(29.7%)은 baseline보다 51% 빠르다(0.97ms) — Early Exit 자체의 이득은 진짜 있다. 그런데 exit3(58.7%, 이 test set은 라벨이 어려워서 얕은 exit로 안 끝나는 샘플이 많다)는 baseline보다 오히려 57% 느리다(3.08ms). staged 방식은 `ort.InferenceSession.run()`을 최대 3번 순차 호출하는데, 이 모델은 `hidden_size=128`로 작아서 LSTM 레이어 자체의 연산량보다 **세션 호출 1번당 붙는 고정 오버헤드**(스레드 동기화, Python-C++ 경계, 텐서 메모리 할당)가 더 크다. exit3까지 가는 샘플은 그 오버헤드를 3번 다 물어야 해서 밑지는 장사가 된다.

### 1학기 자료와 교차검증

이 현상이 오늘 데이터의 우연인지 확인하려고 1학기(4-feature) 실측(`project/results/hojung/`, 실제 Pi 하드웨어)을 다시 봤다. **같은 패턴이 이미 있었다:**

| 방법 | 1학기(4-feature) avg | 오늘(6-feature) avg |
|---|---:|---:|
| Baseline | 1.530ms | 1.966ms |
| Fixed θ (staged) | 2.089ms (+37%) | 2.337ms (+19%) |
| Dynamic θ (staged) | 1.989ms (+30%) | 2.189ms (+11%) |

모델 세대·feature 개수와 무관하게 방향이 똑같다 — staged 구조 자체가 이 Pi + ONNX Runtime + 이 정도로 작은 LSTM 조합에서 구조적으로 불리하다는 뜻이다. (1학기의 별도 `comparison_summary.txt`는 반대 결론을 냈는데, 그건 PC 타이밍으로 보이는 수치라 실제 Pi 실측끼리 직접 대조한 적은 이번이 처음이었다.)

## 원인 재정리: LSTM 재계산이 아니라 세션 호출 오버헤드

"LSTM은 레이어마다 다시 계산하니까 작은 모델일수록 불리한 거 아니냐"는 질문이 나왔는데, 정확히는 이렇다:

- staged 방식은 **레이어를 중복 계산하지 않는다** — stage2는 stage1이 만든 hidden state를 입력으로 받아 이어서 계산할 뿐이다.
- 진짜 비용은 **`InferenceSession.run()`을 여러 번 호출하는 고정비**다. 이 오버헤드는 모델 크기와 거의 무관하게 일정한데, skip해서 아끼는 실제 연산량(작은 LSTM 한 층)은 원래도 작다. "아끼는 양"은 작고 "치르는 세금"은 고정이니, 모델이 작을수록 손해가 두드러진다 — 라는 직관이 정확히 맞았다.

## 2차 시도: 단일 그래프 + ONNX If 노드

세션 호출을 여러 번 하지 않고, **조건부 실행 자체를 그래프 안으로** 넣으면 세금 없이 이득만 챙길 수 있다. `torch.onnx.export`의 레거시(TorchScript) 경로는 파이썬 `if`문을 그대로 트레이싱하면 데이터 의존적 분기를 못 잡지만(트레이싱은 한 번 실행한 경로만 굳혀버린다), **`torch.jit.script`로 스크립팅하면** entropy 비교 `if` 문이 실제 제어 흐름으로 컴파일되고, 그 스크립트를 ONNX로 export하면 `If` 연산자로 나온다.

`project/scripts/export_onnx_ap_unified.py` (신규):

```python
class UnifiedEarlyExitFixed(nn.Module):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out1, _ = self.lstm1(x)
        logits1 = self.classifier1(out1[:, -1, :])
        e1 = self._entropy(logits1)[0]
        if float(e1.item()) < self.theta_1:
            return logits1, torch.tensor(1, dtype=torch.int64)
        out2, _ = self.lstm2(out1)
        ...  # theta_2 조건도 동일하게 if로
```

`torch.jit.script(module)` → `torch.onnx.export(scripted, ...)` 순서로 내보내면 그래프에 `If` 노드가 실제로 생긴다(`onnx.load()`로 top-level op 목록을 세보면 `If` 1개가 확인된다 — 두 번째 `if`는 첫 If의 서브그래프 안에 중첩되어 있다). Dynamic θ 버전(`UnifiedEarlyExitDynamic`)은 최근 occupancy 변화량으로 임계값을 조정하는 로직(`compute_dynamic_threshold`와 동일 공식)까지 같은 그래프 안에 스크립팅했다.

제약: **batch_size=1 전제**다 — 조건 분기가 "이 한 샘플의 entropy"를 보고 결정되므로, 배치 안의 샘플마다 다른 exit을 타는 배치 추론에는 안 맞는다. 실시간 엣지 추론(윈도우 하나씩 순차 처리)이 목표라 문제없다.

### 검증

PyTorch의 `infer_batch_stepwise()`(참조 구현) 대비 ONNX 출력을 test 310창 전체에서 비교 — **예측 라벨·exit_point 100% 일치**(fixed·dynamic 둘 다 미스매치 0건).

### 재측정 결과

| 방법 | avg 지연 | vs baseline |
|---|---:|---:|
| Baseline | 1.966ms | — |
| Staged Fixed θ | 2.337ms | +19% |
| Staged Dynamic θ | 2.189ms | +11% |
| **통합 Fixed θ (If 노드)** | **1.183ms** | **-40%** |
| **통합 Dynamic θ (If 노드)** | **1.190ms** | **-39%** |

통합 fixed θ의 exit별 지연: exit1 0.373ms / exit2 1.033ms / exit3 1.621ms — **exit3(모든 레이어 계산)조차 baseline(1.966ms)보다 빠르다.** baseline은 매번 3개 분류기 출력을 전부 계산하는데, 통합 그래프는 If 분기 안에서 실제로 필요한 만큼만 계산하기 때문이다. 재실행해도 1.183ms/1.187ms로 재현된다(오차 범위 내).

## 결론

"Early Exit이 이 정도로 작은 LSTM에서는 latency를 못 줄인다"는 처음 결론은 **staged(세션 분리) 배포 아키텍처에 국한된 이야기**였다. 원인(세션 호출 오버헤드)을 없애자 Early Exit의 latency 이득이 실측으로 확인됐다 — 문제는 모델이나 Early Exit 개념이 아니라 배포 방식이었다.

논문·보고서에는 이 **단일 그래프(If 노드) 결과를 "Proposed"** 로 쓴다. staged 결과는 삭제하지 않고 "왜 단순히 세션을 나누는 방식이 아니라 단일 그래프 설계가 필요했는가"를 보여주는 동기/반례로 남긴다(데이터 정직성 원칙 — 실패한 시도도 기록에서 지우지 않는다).

## 후속 1차: INT8 양자화 시도 — "속도 이득 없음" (오판)

단일 그래프가 baseline보다 40% 빠르다는 걸 확인한 김에, INT8 동적 양자화(`onnxruntime.quantization.quantize_dynamic`, `LSTM`·`MatMul`·`Gemm` 포함)까지 시도했다(`project/scripts/export_onnx_ap_unified_int8.py`).

- **정확도**: fp32 unified 대비 test 310창 전체 **0 mismatch** — 손실 없음.
- **속도**: fixed 1.204ms / dynamic 1.193ms — fp32 unified(1.183ms/1.190ms)와 **오차 범위 내 동일, 이득 없음**.

"이 모델 크기(`hidden_size=128`)에서는 이론상 더 가벼운 연산이 고정 오버헤드에 묻힌다"는 staged→unified 때와 같은 교훈이 반복된 것으로 보고 **여기서 결론을 냈었다.** — 틀린 결론이었다.

## 후속 2차: "1학기 땐 됐었는데?" — 재확인 질문에서 원인을 다시 찾음

사용자가 1학기(4-feature) INT8 결과를 근거로 이 결론에 의문을 제기했다: 1학기 Pi 실측은 baseline 1.530ms → INT8 0.914ms(**-40%**), staged fixed 2.089ms → INT8 1.297ms(**-38%**)로 **실제로 크게 빨라졌었다.** 같은 도구(`quantize_dynamic`)를 쓰는데 왜 이번엔 이득이 없었을까?

ONNX 그래프를 재귀적으로 순회해서 op 종류를 직접 세어봤다:

```python
def walk(graph, counter, depth=0):
    for n in graph.node:
        counter[(depth, n.op_type)] += 1
        for attr in n.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                walk(attr.g, counter, depth + 1)  # If 노드의 서브그래프까지 재귀
```

결과: **unified(If 포함) 그래프를 양자화하면 LSTM 3개가 전부 원래 `LSTM`(float) 그대로였다** — 제일 작은 `classifier1`의 `Gemm` 하나만 `MatMulInteger`로 바뀌어 있었다. 반면 `ap_early_exit_fixed_stage1.onnx`(제어 흐름이 없는 flat 그래프) 하나만 따로 양자화하니 LSTM이 정상적으로 `DynamicQuantizeLSTM`(진짜 int8 연산)으로 바뀌었다.

**진짜 원인**: "모델이 작아서 양자화가 안 먹힌다"가 아니라, **ONNX Runtime의 동적 양자화 도구가 `If`(제어 흐름) 노드가 있는 그래프에서는 LSTM 변환을 조용히 건너뛴다**는 도구상의 한계였다. 1학기 때 실제로 빨라졌던 이유는 그때 그래프에 애초에 `If` 노드가 없었기 때문(staged 방식 = 단순 분리 = 매 stage가 flat 그래프)이다.

## 후속 3차: 양자화된 조각을 손수 재조립 — 두 마리 토끼를 다 잡음

세션 호출 오버헤드(1차 문제)를 피하려고 만든 단일 그래프가, 이번엔 양자화 도구의 발목을 잡은 셈이다. 해법은 순서를 바꾸는 것 — **양자화가 먹히는 flat 상태로 먼저 각 stage를 양자화한 뒤, 그 결과물을 If 노드로 손수 감싼다.**

`project/scripts/export_onnx_ap_unified_int8_v2.py` (신규):

1. `ap_early_exit_{fixed,dynamic}_stage{1,2,3}.onnx`(staged, flat) 각각을 독립적으로 양자화 → LSTM 3개 다 `DynamicQuantizeLSTM`으로 정상 변환됨(확인 완료).
2. `onnx.helper`로 3개 조각을 직접 조립 — entropy/threshold/If 배선(`Softmax→+eps→Log→Mul→ReduceSum→Neg→Gather(0)→Less(theta)→Cast→If`)을 fp32 unified 그래프와 동일하게 재현. 각 stage 내부 텐서 이름은 이름 충돌 방지를 위해 prefix를 붙이되, 경계 텐서(`input`/`hidden1`/`hidden2`/`exit1`/`exit2`/`exit3`)는 stage export 관례상 이미 이름이 일치해서 그대로 이어붙임. Dynamic θ는 occupancy 변화량 기반 임계값 조정 로직(`Slice→Gather→Sub→Abs→Greater→Where`)도 같은 방식으로 그래프에 재현.

### 검증 및 결과

PyTorch 참조 구현 대비 test 310창 중 **309개 정확히 일치**(fixed·dynamic 각각 1개만 경계값 근처 엔트로피 오차 — int8 양자화 노이즈 수준, 정확도는 fp32와 동일).

| 방법 | avg 지연 | vs baseline | vs unified fp32 |
|---|---:|---:|---:|
| Baseline | 1.966ms | — | |
| 통합 fp32 Fixed θ | 1.183ms | -40% | |
| 통합 int8 1차(LSTM 미양자화) | 1.204ms | -39% | +2%(이득 없음) |
| **통합 int8 v2 Fixed θ (LSTM 진짜 양자화)** | **0.641ms** | **-67%** | **-46%** |
| **통합 int8 v2 Dynamic θ (LSTM 진짜 양자화)** | **0.679ms** | **-65%** | **-43%** |

exit별 지연(fixed θ v2): exit1 0.288ms(baseline 대비 -85%) / exit2 0.569ms(-71%) / exit3 0.832ms(-58%) — 모든 exit point에서 baseline은 물론 fp32 unified보다도 크게 빠르다.

**최종 결론**: 1학기 자료가 실제로 보여줬던 int8 이득이 이번에도 재현 가능했다 — 그래프 구조(If 노드) 때문에 막혀 있었을 뿐이다. **최종 배포 구성은 "단일 그래프 + INT8(stage별 양자화 후 재조립)"**로 확정한다. fp32 unified(1.18ms)는 안전한 대안으로, 1차 int8 시도(1.20ms, LSTM 미양자화)는 실패 사례로 기록에 남긴다 — 사용자가 1학기 결과를 근거로 재확인 질문을 하지 않았다면 이 오판을 그대로 최종 결론으로 남길 뻔했다.

## 관련 파일

- `project/scripts/export_onnx_ap.py` — staged export (참고/역사 기록용으로 유지, 배포엔 unified 사용 권장)
- `project/scripts/export_onnx_ap_unified.py` — 단일 그래프 export, fp32(안전한 대안)
- `project/scripts/export_onnx_ap_unified_int8.py` — INT8 1차 시도(unified 그래프를 그대로 양자화 — LSTM이 안 바뀌는 실패 사례로 기록, 채택 안 함)
- `project/scripts/export_onnx_ap_unified_int8_v2.py` — **INT8 최종(권장, 최종 배포)**: staged로 먼저 양자화 후 손수 재조립
- `project/checkpoints/ap_v2_redesign2/ap_early_exit_{fixed,dynamic}_unified.onnx`
- `project/deploy/raspberry_pi_ap_v2/inference_pi_ap.py` — staged 추론 러너
- `project/deploy/raspberry_pi_ap_v2/bench_unified.py` — 단일 그래프 벤치마크 러너
- `project/results/yongsang/ap_v2_redesign2_pi_latency_comparison.txt` — 전체 실측 원본 기록
- `.work-log/current.md` (2026-08-28 밤 섹션) — 세션 타임라인
