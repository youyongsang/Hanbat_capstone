/* Standalone SVG figure generator for the model-comparison charts.
   One source of truth: MODELS + geometry. Emits light-theme standalone SVGs. */
const fs = require("fs");
const path = require("path");

const OUT = process.argv[2];
if (!OUT) { console.error("usage: node build_figures.js <out-dir>"); process.exit(1); }
fs.mkdirSync(OUT, { recursive: true });

const C = {
  ink: "#12151b", ink2: "#4b5566", mute: "#7c8494",
  grid: "#e1e0d9", axis: "#c3c2b7", surface: "#ffffff",
  blue: "#2a78d6", orange: "#eb6834", gray: "#8a9099",
  exit1: "#86b6ef", exit2: "#2a78d6", exit3: "#104281",
  target: "#d03b3b",
};
const FONT = '-apple-system, "Segoe UI", "Malgun Gothic", "Noto Sans KR", system-ui, sans-serif';

// window 12, 2551-row data (2026-09-01 collection). acc/F1 = 5-seed test mean +-std (test 366).
// exits = deploy checkpoint. pi = Pi INT8 (2026-09-01, test 366).
const MODELS = [
  { key:"baseline", name:"Baseline (EE 없음)", short:"Baseline", acc:92.9, accSd:1.2, f1:81.3, f1Sd:3.1, pi:0.851, exits:[0,0,100], fam:"gray", scatter:true },
  { key:"sdn", name:"SDN (논문 충실)", short:"SDN", acc:93.3, accSd:0.9, f1:84.2, f1Sd:1.5, pi:0.516, exits:[56,34,11], fam:"orange", scatter:true },
  { key:"eef", name:"Proposed EE · Fixed θ", short:"EE Fixed θ", acc:92.1, accSd:0.6, f1:82.9, f1Sd:2.0, pi:0.662, exits:[28,41,31], fam:"blue", scatter:true },
  { key:"eed", name:"Proposed EE · Dynamic θ", short:"EE Dynamic θ", acc:92.6, accSd:0.7, f1:85.2, f1Sd:1.8, pi:0.658, exits:[30,53,16], fam:"blue", scatter:false },
];
const FAMC = { gray:C.gray, orange:C.orange, blue:C.blue };

// ---- tiny SVG builder ----
function el(tag, attrs, kids) {
  const a = Object.entries(attrs || {})
    .map(([k, v]) => ` ${k}="${String(v).replace(/"/g, "&quot;")}"`).join("");
  const inner = (kids || []).join("");
  return `<${tag}${a}>${inner}</${tag}>`;
}
function esc(s) { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
function T(x, y, s, o) {
  o = o || {};
  return el("text", {
    x, y, "text-anchor": o.anchor || "start", "font-size": o.size || 12,
    "font-weight": o.weight || 400, fill: o.fill || C.mute,
    "font-family": FONT, transform: o.transform || undefined,
  }, [esc(s)]);
}
function line(x1, y1, x2, y2, stroke, w) {
  return el("line", { x1, y1, x2, y2, stroke, "stroke-width": w || 1 });
}

function wrap(W, H, title, sub, body) {
  const head = [
    el("rect", { x: 0, y: 0, width: W, height: H, fill: C.surface }),
    T(20, 30, title, { size: 15, weight: 700, fill: C.ink }),
    sub ? T(20, 48, sub, { size: 11.5, fill: C.mute }) : "",
  ].join("");
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family='${FONT}'>
${head}
${body}
</svg>
`;
}

// ============ 1. scatter: accuracy vs latency ============
function figScatter() {
  const W = 660, H = 380, L = 58, R = 24, TOP = 66, B = 52;
  const x0 = 0.45, x1 = 0.90, y0 = 88, y1 = 96;
  const px = v => L + (v - x0) / (x1 - x0) * (W - L - R);
  const py = v => TOP + (1 - (v - y0) / (y1 - y0)) * (H - TOP - B);
  const pts = MODELS.filter(m => m.accSd != null && m.scatter);
  const g = [];
  for (let yv = 88; yv <= 96; yv += 2) {
    g.push(line(L, py(yv), W - R, py(yv), C.grid, 1));
    g.push(T(L - 10, py(yv) + 4, yv + "%", { anchor: "end", size: 11 }));
  }
  [0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9].forEach(xv =>
    g.push(T(px(xv), H - B + 18, xv.toFixed(2), { anchor: "middle", size: 11 })));
  g.push(line(L, H - B, W - R, H - B, C.axis, 1));
  g.push(T((L + W - R) / 2, H - 8, "Pi INT8 지연 (ms / sample) — 낮을수록 좋음", { anchor: "middle", size: 11.5 }));
  g.push(T(18, (TOP + H - B) / 2, "정확도 (%)", { anchor: "middle", size: 11.5, transform: `rotate(-90 18 ${(TOP + H - B) / 2})` }));
  // target 95
  g.push(line(L, py(95), W - R, py(95), C.target, 1.5));
  g.push(T(W - R, py(95) - 6, "목표 1 — 95%", { anchor: "end", size: 11, fill: C.target, weight: 600 }));
  const LP = {
    baseline: { nameDy: -12, valDy: 2, dx: -14, anch: "end" },
    eef: { nameDy: 24, valDy: 37, dx: 0, anch: "middle" },
    sdn: { nameDy: -12, valDy: 2, dx: 13, anch: "start" },
  };
  pts.forEach(m => {
    const cx = px(m.pi), cy = py(m.acc), col = FAMC[m.fam], p = LP[m.key];
    g.push(line(cx, py(m.acc - m.accSd), cx, py(m.acc + m.accSd), C.ink2, 1.5));
    g.push(line(cx - 4, py(m.acc - m.accSd), cx + 4, py(m.acc - m.accSd), C.ink2, 1.5));
    g.push(line(cx - 4, py(m.acc + m.accSd), cx + 4, py(m.acc + m.accSd), C.ink2, 1.5));
    g.push(el("circle", { cx, cy, r: 7, fill: C.surface }));
    g.push(el("circle", { cx, cy, r: 5.5, fill: col }));
    g.push(T(cx + p.dx, cy + p.nameDy, m.short, { anchor: p.anch, size: 12, weight: 600, fill: C.ink }));
    g.push(T(cx + p.dx, cy + p.valDy, `${m.acc.toFixed(1)}% · ${m.pi.toFixed(3)} ms`, { anchor: p.anch, size: 10.5 }));
  });
  return wrap(W, H, "정확도 vs Pi INT8 지연",
    "5시드 ±1σ (window 12 + 라벨 게이트) · Pi INT8 실측 · 가로축 0.45 ms · EE Dynamic(92.6%·0.658ms)은 Fixed와 겹쳐 생략",
    g.join("\n"));
}

// ============ 2. grouped bar: accuracy + Label3 F1 ============
function figAcc() {
  const W = 660, H = 384, L = 50, R = 20, TOP = 78, B = 70;
  const y0 = 45, y1 = 95;
  const rows = MODELS.filter(m => m.accSd != null);
  const n = rows.length, band = (W - L - R) / n, bw = 24, gap = 8;
  const py = v => TOP + (1 - (v - y0) / (y1 - y0)) * (H - TOP - B);
  const g = [];
  for (let yv = 45; yv <= 95; yv += 10) {
    g.push(line(L, py(yv), W - R, py(yv), C.grid, 1));
    g.push(T(L - 10, py(yv) + 4, yv + "%", { anchor: "end", size: 11 }));
  }
  g.push(line(L, py(y0), W - R, py(y0), C.axis, 1));
  g.push(line(L, py(95), W - R, py(95), C.target, 1.5));
  g.push(T(W - R, py(95) - 6, "목표 1 — 95%", { anchor: "end", size: 11, fill: C.target, weight: 600 }));
  rows.forEach((m, i) => {
    const cx = L + band * i + band / 2;
    [{ v: m.acc, sd: m.accSd, c: C.blue }, { v: m.f1, sd: m.f1Sd, c: C.orange }].forEach((s, j) => {
      const bx = cx - (bw + gap / 2) + j * (bw + gap);
      const top = py(s.v);
      g.push(el("rect", { x: bx, y: top, width: bw, height: Math.max(0, py(y0) - top), rx: 4, fill: s.c }));
      g.push(el("rect", { x: bx, y: py(y0) - 3, width: bw, height: 3, fill: s.c }));
      const wx = bx + bw / 2;
      g.push(line(wx, py(s.v - s.sd), wx, py(s.v + s.sd), C.ink2, 1.5));
      g.push(line(wx - 4, py(s.v + s.sd), wx + 4, py(s.v + s.sd), C.ink2, 1.5));
      g.push(line(wx - 4, py(s.v - s.sd), wx + 4, py(s.v - s.sd), C.ink2, 1.5));
      g.push(T(wx, py(s.v + s.sd) - 7, s.v.toFixed(1), { anchor: "middle", size: 10.5, weight: 600, fill: C.ink }));
    });
    g.push(T(cx, H - B + 20, m.short, { anchor: "middle", size: 11.5 }));
  });
  // legend
  const ly = H - 20;
  g.push(el("rect", { x: L, y: ly - 9, width: 12, height: 12, rx: 3, fill: C.blue }));
  g.push(T(L + 18, ly + 1, "전체 정확도", { size: 11, fill: C.ink2 }));
  g.push(el("rect", { x: L + 110, y: ly - 9, width: 12, height: 12, rx: 3, fill: C.orange }));
  g.push(T(L + 128, ly + 1, "Label 3 (심각) F1", { size: 11, fill: C.ink2 }));
  return wrap(W, H, "정확도 · Label 3 F1 — 5시드 평균 ±표준편차 (window 12 + 라벨 게이트)",
    "정확도 근소차(SDN 근소 선두) · Label 3 F1 81~85%로 붙음 · 라벨 지속성 게이트로 전 모델 +1.5~2.4pt",
    g.join("\n"));
}

// ============ 3. latency bar ============
function figLat() {
  const W = 660, H = 344, L = 50, R = 20, TOP = 66, B = 52;
  const y1 = 1.1;
  const n = MODELS.length, band = (W - L - R) / n, bw = 46;
  const py = v => TOP + (1 - v / y1) * (H - TOP - B);
  const g = [];
  for (let yv = 0; yv <= 1.0; yv += 0.25) {
    g.push(line(L, py(yv), W - R, py(yv), C.grid, 1));
    g.push(T(L - 10, py(yv) + 4, yv.toFixed(2), { anchor: "end", size: 11 }));
  }
  g.push(line(L, py(0), W - R, py(0), C.axis, 1));
  g.push(line(L, py(1.0), W - R, py(1.0), C.target, 1.5));
  g.push(T(W - R, py(1.0) - 6, "목표 2 — 1.0 ms", { anchor: "end", size: 11, fill: C.target, weight: 600 }));
  g.push(T(16, (TOP + H - B) / 2, "ms / sample", { anchor: "middle", size: 11.5, transform: `rotate(-90 16 ${(TOP + H - B) / 2})` }));
  MODELS.forEach((m, i) => {
    const cx = L + band * i + band / 2, bx = cx - bw / 2, top = py(m.pi);
    g.push(el("rect", { x: bx, y: top, width: bw, height: py(0) - top, rx: 4, fill: FAMC[m.fam] }));
    g.push(el("rect", { x: bx, y: py(0) - 3, width: bw, height: 3, fill: FAMC[m.fam] }));
    g.push(T(cx, top - 8, m.pi.toFixed(3), { anchor: "middle", size: 12, weight: 600, fill: C.ink }));
    g.push(T(cx, H - B + 20, m.short, { anchor: "middle", size: 11 }));
  });
  return wrap(W, H, "Pi INT8 평균 추론 지연 (ms / sample) — window 12",
    "test 365창 · 5회 반복 평균 · 2026-09-02 · 전부 avg <1 ms · EE Fixed = Baseline -22% · SDN 저지연은 T=0.70의 exit1 front-load",
    g.join("\n"));
}

// ============ 4. exit distribution (horizontal stacked) ============
function figExit() {
  const W = 660, H = 300, L = 150, R = 44, TOP = 62, B = 46;
  const rowH = 34;
  const gap = (H - TOP - B - rowH * MODELS.length) / (MODELS.length - 1);
  const cols = [C.exit1, C.exit2, C.exit3], names = ["exit 1", "exit 2", "exit 3"];
  const px = v => L + v / 100 * (W - L - R);
  const g = [];
  [0, 25, 50, 75, 100].forEach(xv => {
    g.push(line(px(xv), TOP - 6, px(xv), TOP + rowH * MODELS.length + gap * (MODELS.length - 1), C.grid, 1));
    g.push(T(px(xv), H - B + 16, xv + "%", { anchor: "middle", size: 10.5 }));
  });
  MODELS.forEach((m, i) => {
    const y = TOP + i * (rowH + gap);
    g.push(T(L - 12, y + rowH / 2 + 4, m.short, { anchor: "end", size: 11.5, weight: 600, fill: C.ink }));
    let acc = 0;
    m.exits.forEach((v, j) => {
      if (v > 0) {
        const xs = px(acc), xe = px(acc + v);
        g.push(el("rect", { x: xs, y, width: Math.max(0, xe - xs - 2), height: rowH, fill: cols[j], rx: (j === 0 || j === 2) ? 3 : 0 }));
        if (v >= 10) g.push(T(xs + (xe - xs) / 2, y + rowH / 2 + 4, v + "%", { anchor: "middle", size: 11, weight: 600, fill: j === 0 ? C.ink : "#ffffff" }));
      }
      acc += v;
    });
  });
  // legend
  const ly = H - 18;
  cols.forEach((c, j) => {
    const lx = L + j * 150;
    g.push(el("rect", { x: lx, y: ly - 9, width: 12, height: 12, rx: 3, fill: c }));
    g.push(T(lx + 18, ly + 1, `${names[j]} (LSTM ${j + 1}층${j === 2 ? ", 최종" : ""})`, { size: 10.5, fill: C.ink2 }));
  });
  return wrap(W, H, "샘플이 어느 exit에서 종료됐나 (%)",
    "window 12 · 2551+게이트 배포 체크포인트 · Baseline은 항상 exit 3 · Proposed·SDN은 대부분 exit 1~2에서 종료",
    g.join("\n"));
}

const figs = {
  "01_accuracy_vs_latency.svg": figScatter(),
  "02_accuracy_and_f1.svg": figAcc(),
  "03_pi_latency.svg": figLat(),
  "04_exit_distribution.svg": figExit(),
};
for (const [name, svg] of Object.entries(figs)) {
  fs.writeFileSync(path.join(OUT, name), svg);
  console.log("wrote", name, `(${svg.length} B)`);
}
