const $ = (id) => document.getElementById(id);
let fileId = null, duration = 0, peaks = [], stepTimer = null;

function show(id, on = true) { $(id).hidden = !on; }
function showError(msg) { $("error").textContent = msg; show("error", !!msg); }
function fmtTime(s) { const m = Math.floor(s / 60); return `${String(m).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }

// ---------- upload (UC01) ----------
async function uploadFile(file) {
  if ($("browse").disabled) return;
  $("browse").disabled = true;
  $("browse").textContent = "Uploading…";
  fileId = null;
  showError(""); show("file-card", false);
  const form = new FormData(); form.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.id) { showError(data.error || `Upload failed (HTTP ${res.status}). Please retry.`); return; }
    fileId = data.id; duration = data.duration;
    $("file-name").textContent = data.name;
    $("file-info").textContent = `Duration ${fmtTime(data.duration)}   Size ${(data.size / 1048576).toFixed(1)} MB   File accepted`;
    show("file-card");
  } catch (e) {
    showError("The upload connection was interrupted. Please retry.");
  } finally {
    $("browse").disabled = false;
    $("browse").textContent = "Browse files";
    $("file-input").value = "";
  }
}
$("browse").onclick = () => $("file-input").click();
$("file-input").onchange = (e) => e.target.files[0] && uploadFile(e.target.files[0]);
const drop = $("drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); e.dataTransfer.files[0] && uploadFile(e.dataTransfer.files[0]); };
$("remove").onclick = reset;
$("again").onclick = reset;

async function reset() {
  if (fileId) fetch(`/api/file/${fileId}`, { method: "DELETE" });
  fileId = null; $("file-input").value = "";
  ["file-card", "progress-section", "result-section", "explore-section"].forEach((i) => show(i, false));
  show("upload-section"); showError(""); $("player").pause();
}

// ---------- analysis (UC02): the progress steps are only indicative, the server does one request ----------
function startProgress() {
  const items = [...$("steps").children]; let i = 0;
  items.forEach((li) => li.classList.remove("done")); $("bar").value = 5;
  stepTimer = setInterval(() => { if (i < items.length - 1) { items[i++].classList.add("done"); $("bar").value = (i / items.length) * 90; } }, 700);
}
function stopProgress() { clearInterval(stepTimer); [...$("steps").children].forEach((li) => li.classList.add("done")); $("bar").value = 100; }

$("analyze").onclick = async () => {
  if (!fileId) return;
  showError(""); $("progress-name").textContent = $("file-name").textContent;
  show("upload-section", false); show("progress-section"); startProgress();
  try {
    const res = await fetch(`/api/analyze/${fileId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
      signal: AbortSignal.timeout(90000),
    });
    const data = await res.json();
    stopProgress(); show("progress-section", false);
    if (!res.ok) { show("upload-section"); showError(data.error || `Analysis failed (HTTP ${res.status}). Please retry.`); return; }
    showResult(data);
  } catch (e) { stopProgress(); show("progress-section", false); show("upload-section"); showError(e.name === "TimeoutError" ? "Analysis timed out. Please retry with a shorter clip." : "The analysis connection was interrupted. Please upload the file again."); }
};

// ---------- result (UC03) ----------
const TEXT = { real: "Likely real", synthetic: "Likely synthetic", uncertain: "Uncertain" };
function showResult(r) {
  peaks = r.peaks;
  $("result-label").textContent = TEXT[r.label];
  $("result-score").textContent = `Score: ${r.score.toFixed(2)}  (0 = real, 1 = synthetic)`;
  if (r.sampled) $("result-score").textContent += ` · Sampled ${r.n_segments} windows across the recording`;
  $("marker").style.left = `${r.score * 100}%`;
  $("export").href = `/api/export/${fileId}`;
  show("result-section"); show("explore-section");
  if ($("cues").checked) new Audio(`/api/cue/${r.label}.wav`).play().catch(() => {});
  setupExplore();
}

// ---------- waveform, spectrogram, playback (UC04, UC05) ----------
function drawWave() {
  const c = $("wave"); c.width = c.clientWidth * (window.devicePixelRatio || 1); c.height = 130 * (window.devicePixelRatio || 1);
  const g = c.getContext("2d"), mid = c.height / 2; g.clearRect(0, 0, c.width, c.height); g.fillStyle = "#000";
  peaks.forEach(([lo, hi], i) => { const x = (i / peaks.length) * c.width; g.fillRect(x, mid - hi * mid, Math.max(1, c.width / peaks.length), Math.max(1, (hi - lo) * mid)); });
}
function setupExplore() {
  $("spec").src = `/api/spectrogram/${fileId}.png`;
  $("player").src = `/api/audio/${fileId}?band=off`;
  document.querySelectorAll("#filters button").forEach((b) => b.classList.toggle("active", b.dataset.band === "off"));
  show("custom", false); $("zoom").value = 1; $("inner").style.width = "100%";
  requestAnimationFrame(drawWave);
}
$("zoom").oninput = (e) => { $("inner").style.width = `${e.target.value * 100}%`; drawWave(); };
window.onresize = () => peaks.length && drawWave();

function movePlayhead() { const p = $("player"); if (duration) $("playhead").style.left = `${(p.currentTime / duration) * 100}%`; }
$("player").ontimeupdate = movePlayhead;
$("player").onplay = function tick() { movePlayhead(); if (!$("player").paused) requestAnimationFrame(tick); };
$("wave").onclick = (e) => { const r = $("wave").getBoundingClientRect(); $("player").currentTime = ((e.clientX - r.left) / r.width) * duration; };

function setBand(url, button) {            // switch between original and filtered audio, keep position
  const p = $("player"), t = p.currentTime, wasPlaying = !p.paused;
  p.src = url; p.onloadedmetadata = () => { p.currentTime = t; if (wasPlaying) p.play(); };
  document.querySelectorAll("#filters button").forEach((b) => b.classList.toggle("active", b === button));
}
document.querySelectorAll("#filters button").forEach((b) => b.onclick = () => {
  if (b.dataset.band === "custom") { show("custom"); document.querySelectorAll("#filters button").forEach((x) => x.classList.toggle("active", x === b)); return; }
  show("custom", false); setBand(`/api/audio/${fileId}?band=${b.dataset.band}`, b);
});
$("apply").onclick = () => setBand(`/api/audio/${fileId}?band=custom&low=${$("low").value}&high=${$("high").value}`, document.querySelector('#filters [data-band="custom"]'));
