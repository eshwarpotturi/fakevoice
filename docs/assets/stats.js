"use strict";
// Results page: a scroll story. One question per section, one big number, one short line, one chart.
const $ = (id) => document.getElementById(id);
const pct = (v) => (v == null ? "n/a" : Math.round(v * 100) + "%");
const f2 = (v) => (v == null ? "n/a" : v.toFixed(2));

(function theme() {
  const saved = (() => { try { return localStorage.getItem("theme"); } catch (e) { return null; } })();
  if (saved) document.documentElement.dataset.theme = saved;
  const set = () => { $("theme").textContent = document.documentElement.dataset.theme === "dark" ? "Light mode" : "Dark mode"; };
  set();
  $("theme").onclick = () => {
    const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("theme", t); } catch (e) {}
    set();
  };
})();
const tip = $("tip");
const showTip = (e, h) => { tip.innerHTML = h; tip.style.display = "block"; tip.style.left = Math.min(e.clientX + 14, innerWidth - 340) + "px"; tip.style.top = e.clientY + 14 + "px"; };
const hideTip = () => { tip.style.display = "none"; };
const openDialog = (t, h) => { $("dlg-t").textContent = t; $("dlg-b").innerHTML = h; $("dlg").showModal(); };
$("dlg-x").onclick = () => $("dlg").close();

const NS = "http://www.w3.org/2000/svg";
function el(name, attrs, parent) {
  const n = document.createElementNS(NS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(n);
  return n;
}
// Bar with a rounded data end, square at the baseline.
function barPath(x, y, w, h, r) {
  r = Math.min(r, w, h / 2);
  return `M${x},${y} H${x + w - r} Q${x + w},${y} ${x + w},${y + r} V${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} H${x} Z`;
}
function wrap(textEl, str, y, maxPx) {
  const words = str.split(" "); let line = "", lines = [];
  words.forEach((w) => { if ((line + " " + w).trim().length * 7.4 > maxPx) { lines.push(line); line = w; } else line = (line + " " + w).trim(); });
  lines.push(line);
  const y0 = y - (lines.length - 1) * 8;
  lines.forEach((l, i) => { const ts = el("tspan", { x: textEl.getAttribute("x"), y: y0 + i * 16 }, textEl); ts.textContent = l; });
}

let D;
fetch("data/results.json", { cache: "no-cache" }).then((r) => r.json()).then((d) => { D = d; render(); });

const det = (id) => D.detectors.find((d) => d.id === id);

// ---------- charts ----------
// Grouped bars: clean audio (blue) and phone-quality audio (orange), as "right on X%".
function vizAccuracy(host, rows) {
  host.innerHTML = "";
  const W = 780, left = 290, right = 70, top = 8, rowH = 74, H = top + rows.length * rowH + 40;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "How often each detector was right, on clean and phone-quality audio" }, host);
  const x = (v) => left + v * (W - left - right);
  [0, 0.5, 1].forEach((v) => {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: H - 32, stroke: "var(--line)", "stroke-width": v === 0.5 ? 1.6 : 1, "stroke-dasharray": v === 0.5 ? "5 5" : "" }, svg);
    const t = el("text", { x: x(v), y: H - 12, "text-anchor": "middle" }, svg); t.textContent = v === 0.5 ? "50% coin toss" : Math.round(v * 100) + "%";
  });
  rows.forEach((d, i) => {
    const y0 = top + i * rowH;
    const lab = el("text", { x: left - 12, y: y0 + 30, "text-anchor": "end", style: "font-size:14px" }, svg);
    wrap(lab, d.name, y0 + 30, left - 16);
    [["clean", "var(--series-1)"], ["phone", "var(--series-2)"]].forEach(([k, col], j) => {
      const m = d[k]; if (!m) return;
      const y = y0 + 6 + j * 28, w = Math.max(2, x(m.accuracy) - x(0));
      el("path", { d: barPath(x(0), y, w, 24, 5), fill: col }, svg);
      const v = el("text", { x: x(m.accuracy) + 8, y: y + 17, class: "val", style: "font-size:15px" }, svg); v.textContent = pct(m.accuracy);
      const hit = el("rect", { x: x(0), y: y - 2, width: W - left - right, height: 28, fill: "transparent", class: "bar-hit", tabindex: 0 }, svg);
      const html = `<b>${d.name}</b><br>${k === "clean" ? "Clean audio" : "Phone-quality audio"}: right on ${pct(m.accuracy)}`;
      hit.addEventListener("mousemove", (e) => showTip(e, html)); hit.addEventListener("mouseleave", hideTip);
      const open = () => detailDialog(d);
      hit.addEventListener("click", open); hit.addEventListener("keydown", (e) => e.key === "Enter" && open());
    });
  });
}
function detailDialog(d) {
  openDialog(d.name, `<p class="muted">${d.note || ""}</p>
    <table class="data"><tr><th></th><th>Clean</th><th>Phone-quality</th></tr>
    <tr><td>Right on</td><td>${pct(d.clean.accuracy)}</td><td>${pct(d.phone && d.phone.accuracy)}</td></tr>
    <tr><td>Score (0.5 = coin toss, 1 = perfect)</td><td>${f2(d.clean.auc)}</td><td>${f2(d.phone && d.phone.auc)}</td></tr>
    <tr><td>Fakes caught at 5% false alarms</td><td>${pct(d.clean.fakes_caught_at_5pct_false_alarms)}</td><td>${pct(d.phone && d.phone.fakes_caught_at_5pct_false_alarms)}</td></tr>
    <tr><td>Likely range of the score (95%)</td><td>${f2(d.clean.auc_95ci[0])} to ${f2(d.clean.auc_95ci[1])}</td><td>${d.phone ? f2(d.phone.auc_95ci[0]) + " to " + f2(d.phone.auc_95ci[1]) : "n/a"}</td></tr></table>`);
}
function tableAccuracy(host, rows) {
  host.innerHTML = `<table class="data"><tr><th>Detector</th><th>Right, clean</th><th>Right, phone</th><th>Score, clean</th><th>Score, phone</th></tr>${rows.map((d) => `<tr><td>${d.name}</td><td>${pct(d.clean.accuracy)}</td><td>${pct(d.phone && d.phone.accuracy)}</td><td>${f2(d.clean.auc)}</td><td>${f2(d.phone && d.phone.auc)}</td></tr>`).join("")}</table>`;
}

// 100 squares: how many of 100 clips it got right.
function vizWaffle(host, right) {
  host.innerHTML = "";
  const n = Math.round(right * 100), S = 34, G = 6, W = 10 * (S + G), H = 10 * (S + G);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", style: "max-width:520px;display:block;margin:0 auto", role: "img", "aria-label": `${n} of 100 clips right` }, host);
  for (let i = 0; i < 100; i++) {
    const r = Math.floor(i / 10), c = i % 10, ok = i < n;
    el("rect", { x: c * (S + G), y: r * (S + G), width: S, height: S, rx: 6, fill: ok ? "var(--series-1)" : "none", stroke: ok ? "none" : "var(--series-2)", "stroke-width": ok ? 0 : 3 }, svg);
  }
  const p = document.createElement("p"); p.className = "small"; p.style.textAlign = "center"; p.style.marginTop = ".6rem";
  p.textContent = `Every square is one clip. Filled: right (${n}). Outlined: wrong (${100 - n}).`;
  host.appendChild(p);
}

function personRows() {
  const d = det("wavlm_both");
  return Object.entries(d.per_speaker_clean).map(([n, v]) => ({ n, ...v })).sort((a, b) => a.auc - b.auc);
}
function vizPerson(host) {
  host.innerHTML = "";
  const rows = personRows();
  const W = 780, left = 210, right = 60, top = 6, rowH = 30, H = top + rows.length * rowH + 30;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Score for each test person" }, host);
  const x = (v) => left + v * (W - left - right);
  [0, 0.5, 1].forEach((v) => {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: H - 26, stroke: "var(--line)", "stroke-dasharray": v === 0.5 ? "5 5" : "" }, svg);
    const t = el("text", { x: x(v), y: H - 8, "text-anchor": "middle" }, svg); t.textContent = v === 0.5 ? "0.5 coin toss" : v;
  });
  rows.forEach((r, i) => {
    const y = top + i * rowH + 4;
    const t = el("text", { x: left - 10, y: y + 15, "text-anchor": "end", style: "font-size:14px" }, svg); t.textContent = r.n;
    el("path", { d: barPath(x(0), y, Math.max(2, x(r.auc) - x(0)), 20, 4), fill: "var(--series-1)" }, svg);
    const v = el("text", { x: x(r.auc) + 8, y: y + 15, class: "val", style: "font-size:14px" }, svg); v.textContent = f2(r.auc);
    const hit = el("rect", { x: x(0), y: y - 4, width: W - left - right, height: 28, fill: "transparent", class: "bar-hit", tabindex: 0 }, svg);
    const html = `<b>${r.n}</b><br>Score ${f2(r.auc)} · ${r.real} real and ${r.fake} fake clips`;
    hit.addEventListener("mousemove", (e) => showTip(e, html)); hit.addEventListener("mouseleave", hideTip);
    const open = () => openDialog(r.n, `<p>Score for this person: <b>${f2(r.auc)}</b>, from ${r.real} real and ${r.fake} fake clips.</p><p class="muted">With so few clips, small differences between people are not reliable. The fake clips were collected from public sources by the dataset's researchers. We made none.</p>`);
    hit.addEventListener("click", open); hit.addEventListener("keydown", (e) => e.key === "Enter" && open());
  });
}
function tablePerson(host) {
  host.innerHTML = `<table class="data"><tr><th>Person</th><th>Score</th><th>Real clips</th><th>Fake clips</th></tr>${personRows().map((r) => `<tr><td>${r.n}</td><td>${f2(r.auc)}</td><td>${r.real}</td><td>${r.fake}</td></tr>`).join("")}</table>`;
}

// Who-is-it accuracy with no training: 12 people (blue) and 49 people (orange), against guessing.
function vizFingerprint(host) {
  host.innerHTML = "";
  const rows = D.fingerprint, W = 780, left = 230, right = 70, top = 8, rowH = 74, H = top + rows.length * rowH + 40;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "How often each embedding named the right person" }, host);
  const x = (v) => left + v * (W - left - right);
  [0, 0.5, 1].forEach((v) => {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: H - 32, stroke: "var(--line)", "stroke-dasharray": v === 0.5 ? "5 5" : "" }, svg);
    const t = el("text", { x: x(v), y: H - 12, "text-anchor": "middle" }, svg); t.textContent = Math.round(v * 100) + "%";
  });
  rows.forEach((d, i) => {
    const y0 = top + i * rowH;
    const lab = el("text", { x: left - 12, y: y0 + 34, "text-anchor": "end", style: "font-size:14px" }, svg); lab.textContent = d.name;
    [[d.who12, "var(--series-1)", `${d.people12} people`], [d.who49, "var(--series-2)", `${d.people49} people`]].forEach(([v, col, txt], j) => {
      const y = y0 + 6 + j * 28;
      el("path", { d: barPath(x(0), y, Math.max(2, x(v) - x(0)), 24, 5), fill: col }, svg);
      const t = el("text", { x: x(v) + 8, y: y + 17, class: "val", style: "font-size:15px" }, svg); t.textContent = pct(v);
      const hit = el("rect", { x: x(0), y: y - 2, width: W - left - right, height: 28, fill: "transparent", class: "bar-hit" }, svg);
      const html = `<b>${d.name}</b><br>Named the right person ${pct(v)} of the time among ${txt}<br>Guessing: ${pct(1 / (txt.startsWith("12") ? d.people12 : d.people49))}`;
      hit.addEventListener("mousemove", (e) => showTip(e, html)); hit.addEventListener("mouseleave", hideTip);
    });
  });
}
function tableFingerprint(host) {
  host.innerHTML = `<table class="data"><tr><th>Embedding</th><th>Right, 12 people</th><th>Right, 49 people</th><th>Same person? wrong</th></tr>${D.fingerprint.map((d) => `<tr><td>${d.name}</td><td>${pct(d.who12)} (${d.clips12} clips)</td><td>${pct(d.who49)} (${d.clips49} clips)</td><td>${pct(d.same_eer)}</td></tr>`).join("")}</table>`;
}

// What the detector said (tick) against what the clips really were (bar).
function vizCal(host) {
  host.innerHTML = "";
  const bins = D.calibration.test_bins;
  const W = 780, left = 190, right = 80, top = 6, rowH = 46, H = top + bins.length * rowH + 62;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Share of clips that were really fake, by what the detector said" }, host);
  const x = (v) => left + v * (W - left - right);
  [0, 0.5, 1].forEach((v) => {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: H - 56, stroke: "var(--line)", "stroke-dasharray": v === 0.5 ? "5 5" : "" }, svg);
    const t = el("text", { x: x(v), y: H - 40, "text-anchor": "middle" }, svg); t.textContent = Math.round(v * 100) + "%";
  });
  bins.forEach((b, i) => {
    const y = top + i * rowH + 6;
    const t = el("text", { x: left - 12, y: y + 20, "text-anchor": "end", style: "font-size:14px" }, svg); t.textContent = `Said ${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}% AI`;
    el("path", { d: barPath(x(0), y, Math.max(3, x(b.observed_fake) - x(0)), 28, 5), fill: "var(--series-2)" }, svg);
    el("rect", { x: x(b.predicted) - 2, y: y - 4, width: 4, height: 36, fill: "var(--text-primary)" }, svg);
    const v = el("text", { x: Math.max(x(b.observed_fake), x(b.predicted)) + 10, y: y + 20, class: "val", style: "font-size:15px" }, svg); v.textContent = pct(b.observed_fake);
    const hit = el("rect", { x: x(0), y: y - 6, width: W - left - right, height: 40, fill: "transparent", class: "bar-hit", tabindex: 0 }, svg);
    const html = `<b>${b.n} clips</b> where the detector said ${Math.round(b.lo * 100)}–${Math.round(b.hi * 100)}% AI<br>Really fake: ${pct(b.observed_fake)}`;
    hit.addEventListener("mousemove", (e) => showTip(e, html)); hit.addEventListener("mouseleave", hideTip);
  });
  const lg = el("text", { x: left, y: H - 8, style: "font-size:13px" }, svg);
  lg.textContent = "Orange bar: share that were really fake. Black tick: what the detector claimed on average.";
}
function tableCal(host) {
  host.innerHTML = `<table class="data"><tr><th>When the detector said</th><th>Clips</th><th>Really fake</th></tr>${D.calibration.test_bins.map((b) => `<tr><td>${pct(b.lo)} to ${pct(b.hi)} AI</td><td>${b.n}</td><td>${pct(b.observed_fake)}</td></tr>`).join("")}</table>`;
}


// Two bars per row: the first version (orange) against the retrained detector (blue).
function vizPairs(host, p) {
  host.innerHTML = "";
  const rows = p.rows, W = 780, left = 290, right = 70, top = 8, rowH = 74, H = top + rows.length * rowH + 40;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": p.aria }, host);
  const x = (v) => left + v * (W - left - right);
  [0, 0.5, 1].forEach((v) => {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: H - 32, stroke: "var(--line)", "stroke-dasharray": v === 0.5 ? "5 5" : "" }, svg);
    const t = el("text", { x: x(v), y: H - 12, "text-anchor": "middle" }, svg); t.textContent = Math.round(v * 100) + "%";
  });
  rows.forEach((r, i) => {
    const y0 = top + i * rowH;
    const lab = el("text", { x: left - 12, y: y0 + 30, "text-anchor": "end", style: "font-size:14px" }, svg);
    wrap(lab, r.name, y0 + 30, left - 16);
    [[r.a, "var(--series-2)", p.la], [r.b, "var(--series-1)", p.lb]].forEach(([v, col, who], j) => {
      const y = y0 + 6 + j * 28;
      el("path", { d: barPath(x(0), y, Math.max(2, x(v) - x(0)), 24, 5), fill: col }, svg);
      const t = el("text", { x: x(v) + 8, y: y + 17, class: "val", style: "font-size:15px" }, svg); t.textContent = pct(v);
      const hit = el("rect", { x: x(0), y: y - 2, width: W - left - right, height: 28, fill: "transparent", class: "bar-hit" }, svg);
      const html = `<b>${r.name}</b><br>${who}: ${pct(v)}${r.n ? `<br>${r.n}` : ""}`;
      hit.addEventListener("mousemove", (e) => showTip(e, html)); hit.addEventListener("mouseleave", hideTip);
    });
  });
}
function tablePairs(host, p) {
  host.innerHTML = p.html || `<table class="data"><tr><th></th><th>${p.la}</th><th>${p.lb}</th></tr>${p.rows.map((r) => `<tr><td>${r.name}</td><td>${pct(r.a)}</td><td>${pct(r.b)}</td></tr>`).join("")}</table>`;
}

// ---------- the story ----------
function render() {
  const used = det("wavlm_both"), cl = det("wavlm_clean"), w2 = det("w2v2_clean"), gm = det("gemini_clean");
  const rd = D.detectors.find((d) => d.kind === "off-the-shelf");
  const n = D.counts, bins = D.calibration.test_bins, lo = bins[0], hi = bins[bins.length - 1];
  $("headline").textContent = "Can you tell a cloned voice from a real one?";
  $("subhead").textContent = "A caller sounds just like your customer. We tested whether a detector can still tell it is a fake, on voices and generators it had never seen, even over a phone line.";

  const old = [
    { q: "Can it catch fakes of people it has never heard?", hero: pct(used.clean.accuracy),
      cap: `Right on ${n.test} clips from ${n.test_speakers} people held out of training. Half real, half fake.`,
      viz: (h) => vizWaffle(h, used.clean.accuracy), title: "Out of every 100 clips" },
    { q: "Does a phone line break it?", hero: `<span class="from">${pct(cl.phone.accuracy)}</span><span class="arrow">→</span><span class="to">${pct(used.phone.accuracy)}</span>`,
      cap: `A detector trained on clean audio fell to ${pct(cl.phone.accuracy)} over a simulated phone line. Trained on phone-quality copies too, it held at ${pct(used.phone.accuracy)}.`,
      note: `Not only the line: wav2vec 2.0, trained on clean audio, lost less (${pct(w2.clean.accuracy)} to ${pct(w2.phone.accuracy)}).`,
      rows: [cl, used, w2], title: "Right on, by training" },
    { q: "What about detectors that already exist?", hero: pct(rd.clean.accuracy), heroClass: "bad",
      cap: "A ready-made detector from a public model hub. Worse than a coin toss.",
      note: `Over the phone line it rose to ${pct(rd.phone.accuracy)}. We do not know why.`,
      rows: [rd, used], title: "Ready-made against ours" },
    { q: "Do Gemini's embeddings hear it too?", hero: pct(gm.clean.accuracy),
      cap: `Partly. Far above guessing, below the speech models (${pct(cl.clean.accuracy)} to ${pct(w2.clean.accuracy)}). Over a phone line: ${pct(gm.phone.accuracy)}.`,
      rows: [gm, w2, cl], title: "Three ways to listen" },
    { q: "Does it work for everyone?", hero: `${n.test_speakers} of ${n.test_speakers}`,
      cap: `Within each person, real and fake separate almost perfectly. Across everyone together: ${pct(used.clean.accuracy)}.`,
      note: "Our reading: the cut-off between real and fake shifts from person to person.",
      person: true, title: "Score for each person" },
    { q: "Can you trust the percentage?", hero: pct(lo.observed_fake), heroClass: "bad",
      cap: `of clips it was sure were human (under 5% AI) were fake. Of those it was sure were fake (over 95%): ${hi.observed_fake >= 0.995 ? "every one" : pct(hi.observed_fake)}.`,
      note: "So the demo never shows less than 5% AI.", cal: true, title: "What it said, against what was true" },
    { q: "One more question: can embeddings fingerprint a voice?", hero: pct(D.fingerprint[0].who12),
      cap: `Gemini's embeddings named the right person among ${D.fingerprint[0].people12} people, with no training. Among ${D.fingerprint[0].people49}, ${pct(D.fingerprint[0].who49)}.`,
      note: `Rough, not precise: for "same person?", it was wrong ${pct(D.fingerprint[0].same_eer)} of the time. Clips of one person may share a recording source, which flatters these numbers.`,
      fp: true, title: "Named the right person, no training" },
  ];

  const V = D.v3, F = V.families, P = V.pooled, RS = V.real_speech;
  const LA = "First version", LB = "Retrained";
  const genHtml = `<table class="data"><tr><th>Held-out generator</th><th>Clips</th><th>First version</th><th>Retrained</th><th>Phone, first</th><th>Phone, retrained</th></tr>${V.generators.map((g) => `<tr><td>${g.family}: ${g.name}</td><td>${g.n}</td><td>${g.current_clean}%</td><td>${g.v3_clean}%</td><td>${g.current_phone}%</td><td>${g.v3_phone}%</td></tr>`).join("")}</table>`;
  const fresh = [
    { q: "Can it catch fakes from voice generators it has never seen?", hero: `<span class="from">${P.clean.current.caught}%</span><span class="arrow">→</span><span class="to">${P.clean.v3.caught}%</span>`,
      cap: `Fakes caught on ${P.n_fake} clips from ${P.generators} generators held out of training. Over a simulated phone line: ${P.phone.v3.caught}%.`,
      note: "Across all fakes, the score for telling fake from real rose from 0.59 to 0.94 (0.5 is a coin toss). Click Show table for each generator.",
      pairs: { rows: F.map((f) => ({ name: f.name, a: f.current.caught / 100, b: f.v3.caught / 100, n: `${f.current.n_fake} fake clips` })), la: LA, lb: LB, aria: "Share of fakes caught, first version against retrained", html: genHtml }, title: "Fakes caught, by dataset" },
    { q: "Does it call real voices fake?", hero: `<span class="from">${P.clean.current.flagged}%</span><span class="arrow">→</span><span class="to">${P.clean.v3.flagged}%</span>`,
      cap: `Real clips wrongly flagged as AI, on the same held-out test sets (${P.n_real} real clips).`,
      note: `Not everywhere: on the 12 people from the first test, wrongly flagged real clips rose from ${F[0].current.flagged}% to ${F[0].v3.flagged}%.`,
      pairs: { rows: F.map((f) => ({ name: f.name, a: f.current.flagged / 100, b: f.v3.flagged / 100, n: `${f.current.n_real} real clips` })), la: LA, lb: LB, aria: "Share of real clips wrongly flagged, first version against retrained" }, title: "Real clips wrongly flagged as AI" },
    { q: "What about real people on ordinary microphones?", hero: `<span class="from">${RS[0].current}%</span><span class="arrow">→</span><span class="to">${RS[0].v3}%</span>`,
      cap: `Real Indian-accented English wrongly flagged (${RS[0].n} clips, speakers never seen in training). The first version flagged almost half. Meeting headsets, a source never seen in training: ${RS[1].current}% to ${RS[1].v3}%.`,
      note: `Noisy audiobook recordings are still hard: ${RS[3].v3}% wrongly flagged.`,
      pairs: { rows: RS.map((r) => ({ name: r.name, a: r.current / 100, b: r.v3 / 100, n: `${r.n} real clips` })), la: LA, lb: LB, aria: "Share of real speech wrongly flagged, first version against retrained" }, title: "Real speech wrongly flagged as AI" },
    { q: "Does it catch a clone of an Indian-accented voice?", hero: "2 of 3", heroClass: "bad",
      cap: `Three clones of one person's voice, made with an open model: 2 of 3 caught on clean audio, 1 of 3 over a simulated phone line. The first version caught 3 of 3 on clean audio. The real recording is called human (${V.own.real.v3}% AI).`,
      note: "Our reading: no fake Indian-accent voices were in training, so the accent may count as a sign of a real voice. Three clips cannot show this. The next step is to train on accent-matched fakes.",
      pairs: { rows: V.own.clones.map((c, i) => ({ name: `Clone ${i + 1}: called AI, clean audio`, a: c.current / 100, b: c.v3 / 100 })), la: LA, lb: LB, aria: "How AI each clone looked, first version against retrained" }, title: "How AI each clone looked" },
    { q: "Can you trust the percentage?", hero: pct(bins[0].observed_fake), heroClass: "bad",
      cap: `of clips it was sure were human (under 5% AI) were fake. Of those it was sure were fake (over 95%): ${pct(hi.observed_fake)}.`,
      note: "So the demo never shows less than 5% AI.", cal: true, title: "What it said, against what was true" },
  ];
  const chapters = [...fresh, ...old.filter((c) => !c.cal).map((c) => ({ ...c, q: "First test: " + c.q }))];

  const host = $("chapters"); host.innerHTML = "";
  chapters.forEach((c) => {
    const sec = document.createElement("section"); sec.className = "chapter";
    sec.innerHTML = `<div><h3 class="q">${c.q}</h3><div class="hero ${c.heroClass || ""}">${c.hero}</div><p class="cap">${c.cap}</p>${c.note ? `<p class="note">${c.note}</p>` : ""}</div>
      <div class="card"><div class="vizhead"><span>${c.title}</span>${c.viz && !c.rows ? "" : '<button class="btn tb">Show table</button>'}</div>${c.rows ? '<div class="legend"><span><i style="background:var(--series-1)"></i>Clean audio</span><span><i style="background:var(--series-2)"></i>Phone-quality audio</span></div>' : ""}${c.pairs ? `<div class="legend"><span><i style="background:var(--series-2)"></i>${c.pairs.la}</span><span><i style="background:var(--series-1)"></i>${c.pairs.lb}</span></div>` : ""}${c.fp ? '<div class="legend"><span><i style="background:var(--series-1)"></i>12 people</span><span><i style="background:var(--series-2)"></i>49 people</span></div>' : ""}<div class="viz"></div></div>`;
    host.appendChild(sec);
    const viz = sec.querySelector(".viz"), btn = sec.querySelector(".tb");
    let chart, table;
    if (c.viz) { chart = () => c.viz(viz); }
    if (c.rows) { chart = () => vizAccuracy(viz, c.rows); table = () => tableAccuracy(viz, c.rows); }
    if (c.person) { chart = () => vizPerson(viz); table = () => tablePerson(viz); }
    if (c.pairs) { chart = () => vizPairs(viz, c.pairs); table = () => tablePairs(viz, c.pairs); }
    if (c.cal) { chart = () => vizCal(viz); table = () => tableCal(viz); }
    if (c.fp) { chart = () => vizFingerprint(viz); table = () => tableFingerprint(viz); }
    chart();
    if (btn && table) { let t = false; btn.onclick = () => { t = !t; btn.textContent = t ? "Show chart" : "Show table"; t ? table() : chart(); }; }
  });
  $("caveats").innerHTML = D.caveats_html;
  $("method").innerHTML = D.method_html;
  $("closing").textContent = "A voice that sounds right is not proof, and a detector's answer is not proof either. Today it is one signal in a longer set of checks.";
}
