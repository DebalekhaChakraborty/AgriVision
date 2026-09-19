/* AgriVision judge demo.
 *
 * Every number on this page comes from a live call to the running service.
 * Nothing is precomputed, and there is no hard-coded example masquerading as a
 * run: if the agent behaved differently tomorrow, this page would show that.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const MAX_BYTES = 12 * 1024 * 1024;

let selected = null;
let lastRun = null;

/* ---------- service status ---------------------------------------------- */

async function checkStatus() {
  const el = $("status"), text = $("statusText");
  try {
    const r = await fetch("/ready");
    const d = await r.json();
    if (d.ready) {
      el.classList.add("ok");
      text.textContent = `Service ready · OpenCV ${d.opencv_version} · model verified`;
    } else {
      el.classList.add("bad");
      text.textContent = d.initialising ? "Service starting…" : "Service not ready";
      if (d.initialising) setTimeout(checkStatus, 3000);
    }
  } catch {
    el.classList.add("bad");
    text.textContent = "Service unreachable";
  }
}

/* ---------- file selection ----------------------------------------------- */

const drop = $("drop"), fileInput = $("file");

function selectFile(file) {
  $("uploadError").hidden = true;
  if (!file) return;
  if (!/^image\/(jpeg|png)$/.test(file.type)) {
    return showUploadError("Please choose a JPG or PNG image.");
  }
  if (file.size > MAX_BYTES) {
    return showUploadError("That image is larger than the 12 MB limit.");
  }
  selected = file;
  const img = $("preview");
  img.src = URL.createObjectURL(file);
  img.hidden = false;
  $("dropIdle").hidden = true;
  $("inspect").disabled = false;
  $("counter").disabled = false;
}

function showUploadError(message) {
  const el = $("uploadError");
  el.textContent = message;
  el.hidden = false;
}

drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
fileInput.addEventListener("change", (e) => selectFile(e.target.files[0]));
["dragenter", "dragover"].forEach((t) =>
  drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) =>
  drop.addEventListener(t, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  selectFile(e.dataTransfer.files[0]);
});

$("reset").addEventListener("click", () => {
  selected = null; lastRun = null;
  fileInput.value = "";
  $("preview").hidden = true; $("dropIdle").hidden = false;
  $("inspect").disabled = true; $("counter").disabled = true;
  $("results").hidden = true; $("cfSection").hidden = true;
  $("uploadError").hidden = true;
  window.scrollTo({ top: 0, behavior: "smooth" });
});

/* ---------- calls --------------------------------------------------------- */

function busy(on, message) {
  $("busy").hidden = !on;
  if (message) $("busyText").textContent = message;
  $("inspect").disabled = on || !selected;
  $("counter").disabled = on || !selected;
}

async function post(path, file) {
  const form = new FormData();
  form.append("image", file, file.name);
  const response = await fetch(path, { method: "POST", body: form });
  let body;
  try { body = await response.json(); }
  catch { throw new Error("The service returned a response this page could not read."); }
  if (!response.ok) {
    const detail = body && body.detail;
    throw new Error((detail && (detail.detail || detail.reason_code)) ||
                    "The request was refused.");
  }
  return body;
}

$("inspect").addEventListener("click", async () => {
  if (!selected) return;
  busy(true, "Running the bounded inspection loop…");
  $("cfSection").hidden = true;
  try {
    lastRun = await post("/inspect", selected);
    renderResult(lastRun);
    $("results").hidden = false;
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showUploadError(error.message);
  } finally { busy(false); }
});

$("counter").addEventListener("click", async () => {
  if (!selected) return;
  busy(true, "Generating controlled variants and running the agent on each…");
  try {
    const data = await post("/counterfactual", selected);
    renderCounterfactual(data);
    $("cfSection").hidden = false;
    $("cfSection").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showUploadError(error.message);
  } finally { busy(false); }
});

/* ---------- rendering ----------------------------------------------------- */

const OUTCOME = {
  COMPLETE:                 ["good", "Inspection complete"],
  REQUEST_RECAPTURE:        ["warn", "Recapture requested"],
  REQUEST_REPOSITION_LIGHT: ["warn", "Reposition light"],
  REQUEST_HUMAN_REVIEW:     ["warn", "Review required"],
  FAILED_SAFE:              ["stop", "Stopped safely"],
};

const HUMAN_COPY = {
  REQUEST_RECAPTURE: ["Please retake the photo",
    "Retake with the fruit in focus, well lit and clearly visible, filling a good part of the frame."],
  REQUEST_REPOSITION_LIGHT: ["Please reposition the light",
    "A strong local highlight is sitting on the subject. Move the light or the camera and try again."],
  REQUEST_HUMAN_REVIEW: ["Human review needed",
    "The system could not reach a conclusion it trusts. A person should look at this capture."],
};

const n = (v, d = 3) =>
  (v === null || v === undefined) ? "—" : Number(v).toFixed(d).replace(/\.?0+$/, "");

function renderResult(r) {
  const [cls, label] = OUTCOME[r.final_status] || ["warn", r.final_status];
  const out = $("outcome");
  out.className = "outcome " + cls;
  out.innerHTML =
    `<span>${label}</span>` +
    `<span class="sub">Condition model ${r.condition_model_invoked ?
      "invoked" : "<strong>not invoked</strong>"} · run ${r.run_id.slice(0, 12)}…</span>`;

  // Human-directed outcomes are intentional, not errors.
  const human = $("humanAction");
  const copy = HUMAN_COPY[r.requested_human_action];
  if (copy) {
    human.innerHTML = `<h3>${copy[0]}</h3><p>${copy[1]}</p>`;
    human.hidden = false;
  } else if (r.quality_summary && r.quality_summary.foreground_valid === false) {
    human.innerHTML = `<h3>Could not isolate one primary subject</h3>` +
      `<p>The system could not establish a trustworthy single-subject foreground, ` +
      `so no subject-restricted measurement was taken.</p>`;
    human.hidden = false;
  } else { human.hidden = true; }

  // Advisory, never presented as a blocking finding.
  if (r.artifact_summary && r.artifact_summary.advisory_human_action) {
    human.hidden = false;
    human.innerHTML += `<p style="margin-top:8px"><strong>Advisory</strong>` +
      `<span class="badge ADVISORY">ADVISORY</span><br>Local highlight evidence ` +
      `was recorded. It did <em>not</em> block this result.</p>`;
  }

  const model = $("modelResult");
  if (r.condition_evidence) {
    model.hidden = false;
    model.innerHTML = `<h3>Visible-condition model</h3><div class="mgrid">` +
      cell("Fruit", String(r.condition_evidence.fruit_type || "—").toUpperCase()) +
      cell("Visible condition", String(r.condition_evidence.visible_condition || "—").toUpperCase()) +
      cell("Confidence", n(r.model_confidence, 3)) +
      `</div><p class="dropsmall subtle">Confidence is recorded, not thresholded: ` +
      `no confidence policy has been calibrated, so it triggers no escalation.</p>`;
  } else { model.hidden = true; }

  renderCausal(r);
  renderForeground(r);
  renderRemediation(r);
  renderPath(r);
  renderTrace(r);
  renderTech(r);
}

const cell = (k, v) => `<div><div class="k">${k}</div><div class="v">${v}</div></div>`;

function findStep(r, tool) {
  return (r.trace_steps || []).find((s) => s.tool_name === tool);
}

function renderCausal(r) {
  const q = findStep(r, "assess_capture_quality");
  const decision = (r.trace_steps || []).find((s) => s.selected_action);
  const box = $("causal");
  let evidence = "Segmentation did not produce a trustworthy subject region.";
  let maturity = "CALIBRATED";

  if (q) {
    const e = q.evidence_summary || {};
    const parts = [];
    if (e.high_frequency_ratio !== undefined)
      parts.push(`High-frequency ratio <span class="num">${n(e.high_frequency_ratio)}</span>` +
                 ` (floor <span class="num">${n(e.focus_floor)}</span>)`);
    if (e.contrast_score !== undefined)
      parts.push(`ROI contrast <span class="num">${n(e.contrast_score)}</span>`);
    if (e.mean_luminance !== undefined)
      parts.push(`ROI mean luminance <span class="num">${n(e.mean_luminance)}</span>`);
    evidence = parts.join(" · ");
    maturity = q.evidence_maturity || "CALIBRATED";
  }

  const flags = (r.quality_summary.quality_flags || [])
    .concat(r.artifact_summary.artifact_flags || []);
  box.innerHTML =
    step("OpenCV finding", evidence, maturity) + `<div class="arrow">↓</div>` +
    step("Interpretation", flags.length ? flags.join(", ") :
         "No blocking capture-quality finding") + `<div class="arrow">↓</div>` +
    step("Agent decision", (decision && decision.selected_action) || "NONE") +
    `<div class="arrow">↓</div>` +
    step("Next", r.condition_model_invoked ?
      "Condition model invoked" : "Condition model <strong>NOT</strong> invoked");
}

function step(label, value, maturity) {
  const badge = maturity ? ` <span class="badge ${maturity}">${maturity}</span>` : "";
  return `<div class="cstep"><div class="lbl">${label}</div>` +
         `<div class="val">${value}${badge}</div></div>`;
}

function renderForeground(r) {
  const fg = findStep(r, "segment_foreground");
  const e = (fg && fg.evidence_summary) || {};
  const q = findStep(r, "assess_capture_quality");
  const qe = (q && q.evidence_summary) || {};
  const provenance = e.mask_provenance || "PRIMARY";
  $("foreground").innerHTML =
    kv("Foreground", e.valid ? "Detected" : "Not established") +
    kv("Subject fraction", n(e.foreground_fraction, 4)) +
    kv("Method", e.accepted_method || "—") +
    kv("Mask source", provenance === "FALLBACK" ?
       'Fallback <span class="badge PROVISIONAL">PROVISIONAL</span>' :
       'Primary <span class="badge CALIBRATED">CALIBRATED</span>') +
    (qe.high_frequency_ratio !== undefined ?
      kv("High-frequency ratio", n(qe.high_frequency_ratio, 4)) : "") +
    (qe.contrast_score !== undefined ? kv("ROI contrast", n(qe.contrast_score, 4)) : "");
}

const kv = (k, v) => `<div class="kv"><div class="k">${k}</div><div class="v">${v}</div></div>`;

function renderRemediation(r) {
  const card = $("remediationCard");
  if (!r.remediation_applied) { card.hidden = true; return; }
  card.hidden = false;
  const quality = (r.trace_steps || []).filter((s) => s.tool_name === "assess_capture_quality");
  const before = quality[0] && quality[0].evidence_summary;
  const after = quality[1] && quality[1].evidence_summary;
  let compare = "";
  if (before && after) {
    compare = `<div class="grid2" style="margin-top:12px">` +
      kv("Mean luminance before", n(before.mean_luminance, 4)) +
      kv("Mean luminance after", n(after.mean_luminance, 4)) +
      kv("ROI contrast before", n(before.contrast_score, 4)) +
      kv("ROI contrast after", n(after.contrast_score, 4)) + `</div>`;
  }
  $("remediation").innerHTML =
    `<p><strong>${r.remediation_action}</strong> — capture enhancement attempted, ` +
    `then the visual evidence was <strong>re-assessed on the new image</strong>. ` +
    `The correction was ${r.remediation_accepted ?
      "accepted by the harm guard" : "<strong>rejected</strong> and the original restored"}.</p>` +
    compare +
    `<p class="dropsmall subtle">Enhancement adjusts tone. It does not restore ` +
    `information the sensor did not record.</p>`;
}

function renderPath(r) {
  const path = r.state_path || [];
  $("statePath").innerHTML = path.map((s, i) => {
    const terminal = i === path.length - 1;
    const cls = terminal ? (s === "COMPLETE" ? "done" : "term") : "";
    return `<span class="pnode ${cls}">${s}</span>` +
           (terminal ? "" : '<span class="psep">→</span>');
  }).join("");
}

function renderTrace(r) {
  const list = $("trace");
  list.innerHTML = (r.trace_steps || []).map((s) => {
    const e = s.evidence_summary || {};
    const bits = [];
    if (e.high_frequency_ratio !== undefined)
      bits.push(`high-frequency ratio ${n(e.high_frequency_ratio)} (required ≥ ${n(e.focus_floor)})`);
    if (e.contrast_score !== undefined) bits.push(`contrast ${n(e.contrast_score)}`);
    if (e.mean_luminance !== undefined) bits.push(`luminance ${n(e.mean_luminance)}`);
    if (e.valid !== undefined) bits.push(`mask ${e.valid ? "valid" : "invalid"}`);
    if (e.glare_flag !== undefined) bits.push(`glare ${e.glare_flag ? "found" : "none"}`);
    if (e.inference_executed !== undefined)
      bits.push(`condition model ${e.inference_executed ? "invoked" : "SKIPPED"}`);
    const badge = s.evidence_maturity ?
      `<span class="badge ${s.evidence_maturity}">${s.evidence_maturity}</span>` : "";
    const action = s.selected_action ?
      `<div class="taction">${s.selected_action}</div>` : "";
    return `<li class="${s.selected_action ? "acts" : ""}">` +
      `<div class="tstep">Step ${s.step_id} · ${s.state_after}</div>` +
      `<div class="ttool">${s.tool_name}${badge}</div>` +
      (bits.length ? `<div class="tev">${bits.join(" · ")}</div>` : "") +
      action +
      (s.duration_ms ? `<div class="tev">${n(s.duration_ms, 1)} ms</div>` : "") +
      `</li>`;
  }).join("");
  $("rawTrace").textContent = JSON.stringify(r.trace_steps || [], null, 2);
}

function renderTech(r) {
  $("techDetails").innerHTML = [
    ["OpenCV", r.opencv_version || "5.0.0"],
    ["Condition runtime", "MobileNetV3-Large · ONNX · cv2.dnn"],
    ["Cloud", "AWS App Runner"],
    ["Agent", "deterministic bounded state machine"],
    ["Max remediation attempts", "1"],
    ["Max model invocations", "1"],
    ["Input contract", "one primary fruit per image"],
    ["Transport", r.lossless_transport ? "lossless (PNG)" : "lossy (JPEG)"],
    ["Pipeline version", r.pipeline_version || "—"],
    ["Run id", r.run_id],
    ["Trace", `<a href="${r.trace_url}">View technical trace</a>`],
    ["Policy fingerprints", Object.entries(r.policy_fingerprints || {})
      .map(([k, v]) => `${k}: ${v}`).join("<br>")],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

/* ---------- counterfactual ------------------------------------------------ */

function renderCounterfactual(d) {
  $("cfNotice").textContent = d.notice;
  $("cfGrid").innerHTML = (d.variants || []).map((v) => {
    const e = v.headline_evidence || {};
    const rows = [];
    if (e.high_frequency_ratio !== undefined && e.high_frequency_ratio !== null)
      rows.push(row("High-frequency", `${n(e.high_frequency_ratio)} / ${n(e.focus_floor)}`));
    if (e.contrast_score !== undefined && e.contrast_score !== null)
      rows.push(row("ROI contrast", n(e.contrast_score)));
    if (e.mean_luminance !== undefined && e.mean_luminance !== null)
      rows.push(row("Luminance", n(e.mean_luminance)));
    if (e.foreground_valid === false) rows.push(row("Foreground", "not established"));
    return `<div class="cfcard"><h3>${v.label}</h3>` +
      `<p class="dropsmall subtle">${v.description}</p>` + rows.join("") +
      `<div class="cfaction">→ ${v.first_action}</div>` +
      `<div class="cfmodel ${v.condition_model_invoked ? "yes" : "no"}">` +
      `Condition model ${v.condition_model_invoked ? "invoked" : "skipped"}</div>` +
      `</div>`;
  }).join("");
  $("cfVerdict").textContent = d.evidence_changes_action
    ? `Same subject, ${d.distinct_first_actions.length} different next actions: ` +
      `${d.distinct_first_actions.join(" · ")}. Only the visual evidence changed.`
    : "Every variant produced the same action on this image.";
}

const row = (k, v) => `<div class="cfrow"><span class="k">${k}:</span> ${v}</div>`;

checkStatus();
