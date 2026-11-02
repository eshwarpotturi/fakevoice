"use strict";
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

let D;
fetch("data/results.json").then((r) => r.json()).then((d) => { D = d; render(); });

function render() {
  const best = D.detectors.find((d) => d.id === D.best);
  $("headline").textContent = D.headline;
  $("subhead").textContent = D.subhead;

  // tiles
  const tiles = [
    { big: pct(best.clean.accuracy), label: "Right on clean audio", detail: `<p>The best detector (${best.name}) was right on <b>${pct(best.clean.accuracy)}</b> of ${D.counts.test} test clips from ${D.counts.test_speakers} speakers it had never heard.</p><p class="muted">Half the clips were real and half were AI-generated, so guessing would score about 50%.</p>` },
    { big: best.phone ? pct(best.phone.accuracy) : "n/a", label: "Right over a simulated phone line", detail: best.phone ? `<p>The same detector, on the same clips after a phone-line effect (narrow band, 8 kHz, compressed), was right on <b>${pct(best.phone.accuracy)}</b>.</p><p class="muted">This is the setting a bank cares about, because calls arrive over phone lines.</p>` : "<p>Not tested.</p>" },
    { big: pct(best.clean.fakes_caught_at_5pct_false_alarms), label: "Fakes caught when 5% of real voices are wrongly flagged", detail: `<p>If a bank accepts wrongly flagging <b>5 in 100</b> genuine callers, this detector catches <b>${pct(best.clean.fakes_caught_at_5pct_false_alarms)}</b> of fakes on clean audio${best.phone ? ` and <b>${pct(best.phone.fakes_caught_at_5pct_false_alarms)}</b> over a phone line` : ""}.</p>` },
  ];
  const t = $("tiles"); t.innerHTML = "";
  tiles.forEach((x) => {
    const d = document.createElement("div"); d.className = "card tile"; d.tabIndex = 0;
    d.innerHTML = `<div class="big">${x.big}</div><div class="label">${x.label}</div>`;
    d.onclick = () => openDialog(x.label, x.detail); d.onkeydown = (e) => e.key === "Enter" && d.onclick();
    t.appendChild(d);
  });

  chart1(); chart2();
  $("method").innerHTML = D.method_html;
  $("cal-sub").textContent = "On the test people only. Clips are grouped by the percentage the detector gave, then compared with what they really were.";
  $("cal").innerHTML = `<table class="data"><tr><th>When the detector said</th><th>Clips</th><th>Actually AI-generated</th></tr>${D.calibration.test_bins.map((b) => `<tr><td>${pct(b.lo)} to ${pct(b.hi)} AI</td><td>${b.n}</td><td>${pct(b.observed_fake)}</td></tr>`).join("")}</table><p class="small" style="margin-top:.5rem">${D.calibration.note}</p>`;
  $("caveats").innerHTML = D.caveats_html;
  $("t1").onclick = () => toggle("chart1", "t1", chart1, table1);
  $("t2").onclick = () => toggle("chart2", "t2", chart2, table2);
}
const shown = {};
function toggle(id, btn, chartFn, tableFn) {
  shown[id] = !shown[id];
  $(btn).textContent = shown[id] ? "Show chart" : "Show table";
  shown[id] ? tableFn() : chartFn();
}

function chart1() {
  const host = $("chart1"); host.innerHTML = "";
  const rows = D.detectors;
  const W = 760, left = 270, right = 60, top = 8, rowH = 70, H = top + rows.length * rowH + 34;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Detector scores on clean and phone-quality audio" }, host);
  const x = (v) => left + v * (W - left - right);
  for (const v of [0, 0.5, 1]) {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: H - 28, stroke: "var(--line)", "stroke-width": v === 0.5 ? 1.5 : 1, "stroke-dasharray": v === 0.5 ? "4 4" : "" }, svg);
    const tx = el("text", { x: x(v), y: H - 10, "text-anchor": "middle" }, svg); tx.textContent = v === 0.5 ? "0.5 coin toss" : v;
  }
  rows.forEach((d, i) => {
    const y0 = top + i * rowH;
    const lab = el("text", { x: left - 10, y: y0 + 26, "text-anchor": "end" }, svg);
    wrap(lab, d.name, y0 + 24, left - 14);
    [["clean", "var(--series-1)"], ["phone", "var(--series-2)"]].forEach(([k, col], j) => {
      const m = d[k]; if (!m) return;
      const y = y0 + 6 + j * 24, w = Math.max(2, x(m.auc) - x(0));
      const p = el("path", { d: barPath(x(0), y, w, 20, 4), fill: col }, svg);
      const v = el("text", { x: x(m.auc) + 6, y: y + 15, class: "val" }, svg); v.textContent = f2(m.auc);
      const hit = el("rect", { x: x(0), y: y - 2, width: W - left - right, height: 24, fill: "transparent", class: "bar-hit", tabindex: 0 }, svg);
      const html = `<b>${d.name}</b><br>${k === "clean" ? "Clean audio" : "Phone-quality audio"}: score ${f2(m.auc)}<br>Right on ${pct(m.accuracy)} of clips`;
      hit.addEventListener("mousemove", (e) => showTip(e, html)); hit.addEventListener("mouseleave", hideTip);
      const open = () => openDialog(d.name, `<p class="muted">${d.kind === "off-the-shelf" ? "A ready-made detector from the Hugging Face model hub. We did not train it." : "Trained by us on a separate set of speakers."}</p>
        <table class="data"><tr><th></th><th>Clean</th><th>Phone-quality</th></tr>
        <tr><td>Score (0.5 = coin toss)</td><td>${f2(d.clean.auc)}</td><td>${f2(d.phone && d.phone.auc)}</td></tr>
        <tr><td>Right on</td><td>${pct(d.clean.accuracy)}</td><td>${pct(d.phone && d.phone.accuracy)}</td></tr>
        <tr><td>Equal error rate</td><td>${pct(d.clean.eer)}</td><td>${pct(d.phone && d.phone.eer)}</td></tr>
        <tr><td>Fakes caught at 5% false alarms</td><td>${pct(d.clean.fakes_caught_at_5pct_false_alarms)}</td><td>${pct(d.phone && d.phone.fakes_caught_at_5pct_false_alarms)}</td></tr>
        <tr><td>Likely range of the score (95%)</td><td>${f2(d.clean.auc_95ci[0])} to ${f2(d.clean.auc_95ci[1])}</td><td>${d.phone ? f2(d.phone.auc_95ci[0]) + " to " + f2(d.phone.auc_95ci[1]) : "n/a"}</td></tr></table>
        <p class="small">${d.note || ""}</p>`);
      hit.addEventListener("click", open); hit.addEventListener("keydown", (e) => e.key === "Enter" && open());
    });
  });
}
function wrap(textEl, str, y, maxPx) {
  const words = str.split(" "); let line = "", lines = [];
  words.forEach((w) => { if ((line + " " + w).trim().length * 7.2 > maxPx) { lines.push(line); line = w; } else line = (line + " " + w).trim(); });
  lines.push(line);
  const y0 = y - (lines.length - 1) * 8;
  lines.forEach((l, i) => { const ts = el("tspan", { x: textEl.getAttribute("x"), y: y0 + i * 16 }, textEl); ts.textContent = l; });
}
function table1() {
  $("chart1").innerHTML = `<table class="data"><tr><th>Detector</th><th>Clean score</th><th>Clean right</th><th>Phone score</th><th>Phone right</th></tr>${D.detectors.map((d) => `<tr><td>${d.name}</td><td>${f2(d.clean.auc)}</td><td>${pct(d.clean.accuracy)}</td><td>${f2(d.phone && d.phone.auc)}</td><td>${pct(d.phone && d.phone.accuracy)}</td></tr>`).join("")}</table>`;
}

function spkRows() {
  const best = D.detectors.find((d) => d.id === D.best);
  return Object.entries(best.per_speaker_clean).map(([n, v]) => ({ n, clean: v, phone: best.per_speaker_phone && best.per_speaker_phone[n] })).sort((a, b) => a.clean.auc - b.clean.auc);
}
function chart2() {
  const best = D.detectors.find((d) => d.id === D.best);
  $("h-spk").textContent = "Speaker by speaker: " + best.name;
  $("spk-sub").textContent = "Each score compares one person's real clips with fake clips of the same person, on clean audio. None of these people was used in training. Click a bar for the clip counts.";
  const host = $("chart2"); host.innerHTML = "";
  const rows = spkRows();
  const W = 760, left = 190, right = 56, top = 6, rowH = 30, H = top + rows.length * rowH + 30;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img", "aria-label": "Score for each test speaker" }, host);
  const x = (v) => left + v * (W - left - right);
  for (const v of [0, 0.5, 1]) {
    el("line", { x1: x(v), x2: x(v), y1: top, y2: H - 26, stroke: "var(--line)", "stroke-dasharray": v === 0.5 ? "4 4" : "" }, svg);
    const tx = el("text", { x: x(v), y: H - 8, "text-anchor": "middle" }, svg); tx.textContent = v;
  }
  rows.forEach((r, i) => {
    const y = top + i * rowH + 4;
    const t = el("text", { x: left - 8, y: y + 14, "text-anchor": "end" }, svg); t.textContent = r.n;
    el("path", { d: barPath(x(0), y, Math.max(2, x(r.clean.auc) - x(0)), 18, 4), fill: "var(--series-1)" }, svg);
    const v = el("text", { x: x(r.clean.auc) + 6, y: y + 14, class: "val" }, svg); v.textContent = f2(r.clean.auc);
    const hit = el("rect", { x: x(0), y: y - 4, width: W - left - right, height: 26, fill: "transparent", class: "bar-hit", tabindex: 0 }, svg);
    const html = `<b>${r.n}</b><br>Score ${f2(r.clean.auc)} · ${r.clean.real} real and ${r.clean.fake} fake clips`;
    hit.addEventListener("mousemove", (e) => showTip(e, html)); hit.addEventListener("mouseleave", hideTip);
    const open = () => openDialog(r.n, `<p>Score on clean audio: <b>${f2(r.clean.auc)}</b>${r.phone ? `; over a phone line: <b>${f2(r.phone.auc)}</b>` : ""}.</p><p class="muted">Based on ${r.clean.real} real and ${r.clean.fake} fake clips. With so few clips per person, small differences between people are not reliable.</p><p class="small">The fake clips were collected from public sources by the dataset's researchers. We did not make any.</p>`);
    hit.addEventListener("click", open); hit.addEventListener("keydown", (e) => e.key === "Enter" && open());
  });
}
function table2() {
  $("chart2").innerHTML = `<table class="data"><tr><th>Speaker</th><th>Clean score</th><th>Phone score</th><th>Real clips</th><th>Fake clips</th></tr>${spkRows().map((r) => `<tr><td>${r.n}</td><td>${f2(r.clean.auc)}</td><td>${f2(r.phone && r.phone.auc)}</td><td>${r.clean.real}</td><td>${r.clean.fake}</td></tr>`).join("")}</table>`;
}
