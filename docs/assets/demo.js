"use strict";
// Fake Voice Check - runs entirely in the browser. Nothing is uploaded.
const SR = 16000;
const $ = (id) => document.getElementById(id);
let cfg, session, samples = [];
let busy = false;

// ---------- theme ----------
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

// ---------- tooltip and dialog ----------
const tip = $("tip");
function showTip(e, html) { tip.innerHTML = html; tip.style.display = "block"; tip.style.left = Math.min(e.clientX + 14, innerWidth - 340) + "px"; tip.style.top = e.clientY + 14 + "px"; }
function hideTip() { tip.style.display = "none"; }
function openDialog(title, html) { $("dlg-t").textContent = title; $("dlg-b").innerHTML = html; $("dlg").showModal(); }
$("dlg-x").onclick = () => $("dlg").close();

// ---------- audio ----------
async function decodeToMono16k(arrayBuffer) {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  const buf = await ctx.decodeAudioData(arrayBuffer.slice(0));
  ctx.close();
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(buf.duration * SR)), SR);
  const src = off.createBufferSource();
  src.buffer = buf; src.connect(off.destination); src.start();
  return (await off.startRendering()).getChannelData(0).slice();
}

// Phone line: 300-3400 Hz band, 8 kHz sampling, mu-law compression, back to 16 kHz.
async function phoneLine(x) {
  const off = new OfflineAudioContext(1, x.length, SR);
  const b = off.createBuffer(1, x.length, SR); b.copyToChannel(x, 0);
  const s = off.createBufferSource(); s.buffer = b;
  const hp = off.createBiquadFilter(); hp.type = "highpass"; hp.frequency.value = 300;
  const lp1 = off.createBiquadFilter(); lp1.type = "lowpass"; lp1.frequency.value = 3400;
  const lp2 = off.createBiquadFilter(); lp2.type = "lowpass"; lp2.frequency.value = 3400;
  s.connect(hp); hp.connect(lp1); lp1.connect(lp2); lp2.connect(off.destination); s.start();
  const f = (await off.startRendering()).getChannelData(0);
  const n8 = Math.floor(f.length / 2), y8 = new Float32Array(n8), mu = 255;
  for (let i = 0; i < n8; i++) {
    const v = Math.max(-1, Math.min(1, f[2 * i]));
    const c = Math.sign(v) * Math.log1p(mu * Math.abs(v)) / Math.log1p(mu);
    const q = Math.round((c + 1) / 2 * 255) / 255 * 2 - 1;
    y8[i] = Math.sign(q) * ((1 + mu) ** Math.abs(q) - 1) / mu;
  }
  const out = new Float32Array(x.length);
  for (let i = 0; i < out.length; i++) {
    const p = i / 2, i0 = Math.min(n8 - 1, Math.floor(p)), i1 = Math.min(n8 - 1, i0 + 1), t = p - i0;
    out[i] = y8[i0] * (1 - t) + y8[i1] * t;
  }
  return out;
}

function toWavBlob(x) {
  const buf = new ArrayBuffer(44 + x.length * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + x.length * 2, true); w(8, "WAVEfmt "); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, SR, true); v.setUint32(28, SR * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, x.length * 2, true);
  for (let i = 0; i < x.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 32767, true);
  return new Blob([buf], { type: "audio/wav" });
}

// ---------- model ----------
let loadingModel = null;
async function loadModel() {
  const r = await fetch("model/detector.onnx");
  if (!r.ok) throw new Error("model file not found (" + r.status + ")");
  const total = +r.headers.get("content-length") || 0, reader = r.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    $("status").textContent = `Loading the detector… ${(got / 1e6).toFixed(0)} of ${total ? (total / 1e6).toFixed(0) : "about 58"} MB`;
  }
  const buf = new Uint8Array(got); let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href; ort.env.wasm.numThreads = 1;
  return ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
}
async function ensureSession() {
  if (session) return session;
  if (!loadingModel) loadingModel = loadModel().then((s) => (session = s)).finally(() => { loadingModel = null; });
  return loadingModel;
}
const sigmoid = (z) => 1 / (1 + Math.exp(-z));
async function scoreWindow(x) {
  const t = new ort.Tensor("float32", x, [1, x.length]);
  const out = await session.run({ audio: t });
  return out.logit.data[0];
}
function windows(x) {
  const W = cfg.window_sec * SR, out = [];
  for (let s = 0; s + W <= x.length && out.length < cfg.max_windows; s += W) out.push([s, s + W]);
  const rem = x.length - out.length * W;
  if (out.length < cfg.max_windows && rem >= 2 * SR) out.push([out.length * W, x.length]);
  if (!out.length && x.length >= 1 * SR) out.push([0, x.length]);
  return out;
}
const rms = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);

async function analyse(x, truth, label) {
  if (busy) return; busy = true;
  try {
    try { await ensureSession(); }
    catch (e) { console.error(e); $("status").textContent = "The detector could not start in this browser (" + (e && e.message ? e.message : e) + "). Please reload the page, or try another browser."; return; }
    $("status").textContent = "Checking…";
    if ($("phone").checked) { $("status").textContent = "Simulating a phone line…"; x = await phoneLine(x); }
    $("player").src = URL.createObjectURL(toWavBlob(x));
    const wins = windows(x);
    if (!wins.length) { $("status").textContent = "That recording is too short. Please use at least 2 seconds."; return; }
    const res = [];
    for (let i = 0; i < wins.length; i++) {
      const seg = x.slice(wins[i][0], wins[i][1]);
      if (rms(seg) < 1e-4) continue;   // silence
      $("status").textContent = `Checking slice ${i + 1} of ${wins.length}…`;
      await new Promise((r) => setTimeout(r, 0));
      const logit = await scoreWindow(seg);
      res.push({ i, start: wins[i][0] / SR, end: wins[i][1] / SR, logit, p: Math.min(cfg.ceil, Math.max(cfg.floor, sigmoid(cfg.platt.coef * (logit - cfg.platt.threshold) + cfg.platt.intercept))) });
    }
    if (!res.length) { $("status").textContent = "The recording is silent."; return; }
    const meanLogit = res.reduce((s, r) => s + r.logit, 0) / res.length;
    const p = Math.min(cfg.ceil, Math.max(cfg.floor, sigmoid(cfg.platt.coef * (meanLogit - cfg.platt.threshold) + cfg.platt.intercept)));
    show(p, res, truth, label, x.length / SR);
    $("status").textContent = "Done.";
  } catch (e) {
    console.error(e);
    $("status").textContent = "The check failed (" + (e && e.message ? e.message : e) + "). Please try again or use a different file.";
  } finally { busy = false; }
}

function show(p, res, truth, label, secs) {
  $("empty").hidden = true; $("result").hidden = false;
  const ai = Math.round(p * 100), human = 100 - ai;
  const unsure = p > 0.35 && p < 0.65;
  const v = $("verdict");
  v.className = "verdict " + (p >= 0.5 ? "ai" : "human");
  v.textContent = (p >= 0.5 ? ai : human) + "% " + (p >= 0.5 ? "AI-generated" : "human");
  $("verdict-sub").textContent = unsure ? "Not sure. Treat this as a coin toss." : (p >= 0.5 ? "Likely a synthetic voice" : "Likely a real voice");
  $("bar").innerHTML = `<div class="h" style="flex:${Math.max(human, 1)}">${human >= 12 ? human + "% human" : ""}</div><div class="a" style="flex:${Math.max(ai, 1)}">${ai >= 12 ? ai + "% AI" : ""}</div>`;
  const tl = $("timeline"); tl.innerHTML = "";
  res.forEach((r) => {
    const d = document.createElement("div");
    d.className = "w"; d.style.height = Math.max(3, r.p * 100) + "%"; d.tabIndex = 0;
    d.setAttribute("aria-label", `Seconds ${r.start.toFixed(0)} to ${r.end.toFixed(0)}: ${Math.round(r.p * 100)} percent AI`);
    const html = `<b>${r.start.toFixed(0)}–${r.end.toFixed(0)} s</b><br>${Math.round(r.p * 100)}% AI-generated`;
    d.onmousemove = (e) => showTip(e, html); d.onmouseleave = hideTip;
    d.onclick = () => openDialog(`Slice ${r.i + 1}: ${r.start.toFixed(0)}–${r.end.toFixed(0)} seconds`,
      `<p><b>${Math.round(r.p * 100)}%</b> AI-generated for this slice.</p><p class="muted">The overall result is the average of all slices, so one odd slice does not decide it.</p>`);
    d.onkeydown = (e) => { if (e.key === "Enter") d.onclick(); };
    tl.appendChild(d);
  });
  const t = $("truth");
  if (truth) {
    const right = (truth === "fake") === (p >= 0.5);
    t.innerHTML = `Known answer for <b>${label}</b>: <b>${truth === "fake" ? "AI-generated" : "genuine"}</b>. The detector was ${unsure ? "unsure" : right ? "right" : "wrong"}${$("phone").checked ? " (after the phone line)" : ""}.`;
  } else t.textContent = `Length checked: ${Math.min(secs, cfg.max_windows * cfg.window_sec).toFixed(0)} seconds.`;
}

// ---------- inputs ----------
async function fromFile(file) {
  $("status").textContent = "Reading the file…";
  try { const x = await decodeToMono16k(await file.arrayBuffer()); await analyse(x, null, file.name); }
  catch (e) { console.error(e); $("status").textContent = "Sorry, that file could not be read. Try a WAV or MP3 file, or a short MP4 video."; }
}
$("pick").onclick = () => $("file").click();
$("file").onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) fromFile(f); };
const drop = $("drop");
["dragenter", "dragover"].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((n) => drop.addEventListener(n, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && fromFile(e.dataTransfer.files[0]));

let recorder = null, chunks = [];
$("rec").onclick = async () => {
  if (recorder) { recorder.stop(); return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    chunks = []; recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop()); recorder = null; $("rec").textContent = "Record with microphone";
      const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
      try { const x = await decodeToMono16k(await blob.arrayBuffer()); await analyse(x, null, "your recording"); }
      catch (e) { $("status").textContent = "Sorry, the recording could not be read."; }
    };
    recorder.start(); $("rec").textContent = "Stop and check"; $("status").textContent = "Recording… speak for at least 5 seconds.";
  } catch (e) { $("status").textContent = "The microphone is not available. Allow access, or choose a file instead."; }
};

async function loadSample(s) {
  $("status").textContent = "Loading sample…";
  const r = await fetch(s.file); const x = await decodeToMono16k(await r.arrayBuffer());
  await analyse(x, s.truth, s.name);
}

// ---------- start ----------
(async function init() {
  try {
    cfg = await (await fetch("model/config.json")).json();
    samples = await (await fetch("data/samples.json")).json();
    const box = $("samples");
    samples.forEach((s, i) => {
      const b = document.createElement("button"); b.className = "btn";
      b.innerHTML = `<span class="rf ${s.truth === "real" ? "r" : "f"}" title="${s.truth === "real" ? "Real recording" : "AI-generated fake"}">${s.truth === "real" ? "R" : "F"}</span> ${i + 1}. ${s.name}`;
      b.onclick = () => loadSample(s); box.appendChild(b);
    });
    $("reliability").innerHTML = cfg.reliability_html;
    await ensureSession();
    $("status").textContent = "Ready. Choose a recording.";
  } catch (e) {
    console.error(e);
    $("status").textContent = "The detector has not loaded yet (" + (e && e.message ? e.message : e) + "). It will try again when you choose a recording.";
  }
})();
