/* DeepHybrid front end — talks to the FastAPI endpoints in main.py */

const $ = (s) => document.querySelector(s);
const api = async (path, opts) => {
  const r = await fetch(path, opts);
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed (${r.status})`);
  }
  return r.json();
};

/* ---------- navigation ---------- */

function goTo(id) {
  document.querySelectorAll(".page").forEach((p) => p.classList.remove("on"));
  $("#" + id).classList.add("on");
  document.querySelectorAll("nav button").forEach((x) => {
    if (x.dataset.go === id) x.setAttribute("aria-current", "page");
    else x.removeAttribute("aria-current");
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Any element carrying data-go navigates — nav tabs and in-page buttons alike.
document.querySelectorAll("[data-go]").forEach((b) => {
  b.onclick = () => goTo(b.dataset.go);
});

/* ---------- overview ---------- */

let DIST = null;

async function boot() {
  const p = await api("/api/parents");

  const fmt = (n) => n.toLocaleString();
  const pctGrown = Math.round((100 * p.n_observed) / p.n_possible);
  $("#ov-stats").innerHTML = [
    [fmt(p.n_possible), "possible crosses",
     `${p.females.length} × ${p.males.length} parents`],
    [fmt(p.n_observed), "already grown", `${pctGrown}% of the grid`],
    [fmt(p.n_untested), "never tested", "what the model is for"],
    ["0.73", "predictive ability", "Pearson r, both parents known"],
  ].map(([v, k, s]) =>
    `<div class="card stat"><div class="v num">${v}</div>
     <div class="k mono">${k}</div><div class="s">${s}</div></div>`).join("");

  const opts = (a) => a.map((x) => `<option>${x}</option>`).join("");
  $("#fem").innerHTML = opts(p.females);
  $("#mal").innerHTML = opts(p.males);
  $("#x-fem").innerHTML =
    '<option value="">All females</option>' + opts(p.females);

  DIST = await api("/api/distribution?bins=48");
  drawOverviewRidge();
  loadTopCrosses();
  loadMethodology();
  loadScannerStatus();
}

function drawOverviewRidge() {
  const { counts, centres, median } = DIST;
  const max = Math.max(...counts);
  $("#ov-ridge").innerHTML = counts.map((c, i) =>
    `<i style="height:${Math.max(3, (100 * c) / max)}%"
      title="${centres[i]} Mg/ha"></i>`).join("");
  $("#ov-lo").textContent = `${centres[0].toFixed(2)} Mg/ha`;
  $("#ov-med").textContent = `median ${median.toFixed(2)}`;
  $("#ov-hi").textContent = `${centres[centres.length - 1].toFixed(2)} Mg/ha`;
}

async function loadTopCrosses() {
  const d = await api("/api/rank?limit=5&untested_only=true");
  $("#ov-top").innerHTML =
    `<thead><tr><th>#</th><th>Female</th><th>Male</th>
     <th>Predicted yield</th></tr></thead><tbody>` +
    d.crosses.map((r, i) =>
      `<tr><td class="mono">${i + 1}</td><td>${r.female}</td>
       <td>${r.male}</td>
       <td class="num">${r.yield_pred.toFixed(3)} Mg/ha</td></tr>`)
      .join("") + "</tbody>";
}

/* ---------- predictor ---------- */

$("#run").onclick = async () => {
  const btn = $("#run");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';
  $("#err").textContent = "";

  try {
    const r = await api("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ female: $("#fem").value, male: $("#mal").value }),
    });
    showResult(r);
  } catch (e) {
    $("#err").textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict cross";
  }
};

function showResult(r) {
  // Three decimals: observed hybrids have SD 0.35 Mg/ha, so two would hide
  // differences that matter to a breeder.
  $("#r-yield").innerHTML = `${r.yield_pred.toFixed(3)}<small>Mg/ha</small>`;
  const sign = r.delta_vs_median >= 0 ? "+" : "";
  $("#r-delta").textContent =
    `${sign}${r.delta_vs_median.toFixed(3)} vs median of all crosses`;

  $("#r-pct").textContent =
    r.percentile === null ? "—" : `${Math.round(r.percentile)}${ordinal(r.percentile)}`;
  $("#r-rank").textContent = r.rank_label;

  let cls = "note", body;
  if (r.observed) {
    body = `<b>This cross has been grown.</b> Measured yield
      <b>${r.observed_value.toFixed(3)} Mg/ha</b>. Trust that over the estimate —
      the estimate is shown so you can see how the model behaves on a known cross.`;
  } else if (r.scenario === "T2") {
    cls = "note good";
    body = `<b>Novel cross between two known parents.</b> Scenario T2,
      r ≈ ${r.r_value.toFixed(2)} — the model's strongest case, and the reason
      this tool exists.`;
  } else {
    body = `<b>Scenario ${r.scenario}.</b> ${r.note}`;
  }
  $("#r-note").innerHTML = `<div class="${cls}">${body}</div>`;

  drawRidge(r.yield_pred);
  LAST_YIELD = r.yield_pred;
  updateEconomics();
  $("#res").classList.add("on");
}

/* ---------- economics ---------- */

let LAST_YIELD = null;

async function updateEconomics() {
  if (LAST_YIELD === null) return;
  const price = parseFloat($("#ec-price").value) || 0;
  const ha = parseFloat($("#ec-ha").value) || 0;
  if (price <= 0 || ha <= 0) return;

  const e = await api("/api/economics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      yield_pred: LAST_YIELD, price_per_tonne: price, hectares: ha }),
  });

  const eur = (v) => "€" + Math.round(v).toLocaleString();
  const sign = e.vs_median >= 0 ? "+" : "−";
  $("#ec-out").innerHTML = `
    <div class="grid" style="grid-template-columns:1fr 1fr;gap:22px">
      <div>
        <div class="mono">Revenue over ${ha} ha</div>
        <div class="big num" style="font-size:32px">${eur(e.revenue)}</div>
        <div class="delta">${eur(e.revenue_per_ha)} per hectare</div>
      </div>
      <div>
        <div class="mono">Against the median cross</div>
        <div class="big num" style="font-size:32px">${sign}${eur(Math.abs(e.vs_median))}</div>
        <div class="delta">median yields ${e.median_yield} Mg/ha</div>
      </div>
    </div>`;
}

["#ec-price", "#ec-ha"].forEach((sel) => {
  const el = $(sel);
  if (el) el.oninput = () => updateEconomics();
});

const ordinal = (n) => {
  const i = Math.round(n) % 100;
  if (i >= 11 && i <= 13) return "th";
  return { 1: "st", 2: "nd", 3: "rd" }[i % 10] || "th";
};

function drawRidge(value) {
  const { counts, centres } = DIST;
  const max = Math.max(...counts);
  let hit = 0;
  centres.forEach((c, i) => {
    if (Math.abs(c - value) < Math.abs(centres[hit] - value)) hit = i;
  });

  $("#ridge").innerHTML = counts
    .map((c, i) => `<i class="${i === hit ? "hit" : ""}"
       style="height:${Math.max(3, (100 * c) / max)}%"
       title="${centres[i]} Mg/ha"></i>`).join("");
  $("#ax-lo").textContent = `${centres[0].toFixed(2)} Mg/ha`;
  $("#ax-hi").textContent = `${centres[centres.length - 1].toFixed(2)} Mg/ha`;
  // Restate the percentile next to the highlighted bar so the chart is
  // readable without cross-referencing the metric above it.
  const pctHere = (100 * counts.slice(0, hit).reduce((a, c) => a + c, 0)
                   / counts.reduce((a, c) => a + c, 0));
  $("#ax-lo").title = `${pctHere.toFixed(0)}% of crosses score lower`;
}

/* ---------- explorer ---------- */

let ROWS = [];

$("#x-run").onclick = async () => {
  const btn = $("#x-run");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>';

  try {
    const q = new URLSearchParams({
      limit: $("#x-n").value,
      untested_only: $("#x-untested").checked,
    });
    if ($("#x-fem").value) q.set("female", $("#x-fem").value);

    const d = await api("/api/rank?" + q);
    ROWS = d.crosses;

    if (!ROWS.length) {
      $("#x-wrap").style.display = "none";
      return;
    }
    $("#x-count").textContent = `${d.count} crosses, ranked by predicted yield`;
    $("#x-table").innerHTML =
      `<thead><tr><th>#</th><th>Female</th><th>Male</th>
       <th>Predicted yield</th><th>Status</th></tr></thead><tbody>` +
      ROWS.map((r, i) =>
        `<tr><td class="mono">${i + 1}</td><td>${r.female}</td>
         <td>${r.male}</td><td class="num">${r.yield_pred.toFixed(3)}</td>
         <td><span class="tag ${r.status === "grown" ? "grey" : ""}">
         ${r.status}</span></td></tr>`).join("") + "</tbody>";
    $("#x-wrap").style.display = "block";
  } finally {
    btn.disabled = false;
    btn.textContent = "Rank crosses";
  }
};

$("#x-csv").onclick = () => {
  const csv = "female,male,predicted_yield,status\n" +
    ROWS.map((r) => `${r.female},${r.male},${r.yield_pred},${r.status}`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = "ranked_crosses.csv";
  a.click();
};

/* ---------- scanner ---------- */

let SC_FILE = null;
let SC_SAMPLES = [];

async function loadScannerStatus() {
  const s = await api("/api/scanner/status");
  $("#sc-eyebrow").textContent = s.available ? "model connected" : "model unavailable";
  $("#sc-banner").innerHTML = s.available
    ? `<div class="note good">${s.message} Reported test accuracy: mIoU ${s.reported_test_metrics.mIoU},
       pixel accuracy ${s.reported_test_metrics.pixel_accuracy}.</div>`
    : `<div class="note"><b>Model unavailable.</b> ${s.message}</div>`;
  $("#sc-run").disabled = true;
  SC_SAMPLES = s.sample_patches_available || [];
  renderSampleButtons();
}

function renderSampleButtons() {
  const box = $("#sc-samples");
  if (!box) return;
  if (!SC_SAMPLES.length) {
    box.innerHTML = '<div class="note">No demo patches bundled.</div>';
    return;
  }
  box.innerHTML = SC_SAMPLES.map((name, i) =>
    `<button class="btn ghost sm sc-sample-btn" data-name="${name}">Sample ${i + 1}</button>`
  ).join(" ");
  box.querySelectorAll(".sc-sample-btn").forEach(btn => {
    btn.onclick = () => runScan({ sampleName: btn.dataset.name });
  });
}

const drop = $("#drop"), fileIn = $("#sc-file");
if (drop) {
  drop.onclick = () => fileIn.click();
  drop.ondragover = (e) => {
    e.preventDefault();
    drop.style.borderColor = "var(--ink-2)";
    drop.style.background = "var(--ok-bg)";
  };
  drop.ondragleave = () => {
    drop.style.borderColor = "var(--line)";
    drop.style.background = "transparent";
  };
  drop.ondrop = (e) => {
    e.preventDefault();
    drop.ondragleave();
    if (e.dataTransfer.files[0]) takeFile(e.dataTransfer.files[0]);
  };
  fileIn.onchange = () => fileIn.files[0] && takeFile(fileIn.files[0]);
}

function takeFile(f) {
  const name = f.name.toLowerCase();
  if (!(name.endsWith(".tif") || name.endsWith(".tiff"))) {
    $("#sc-preview").innerHTML =
      '<div class="note"><b>That won\'t work.</b> This model reads 9-band multispectral ' +
      'GeoTIFF patches from UAV capture — not JPG/PNG photos. Try one of the sample ' +
      'patches above, or upload a .tif with the required bands.</div>';
    SC_FILE = null;
    $("#sc-run").disabled = true;
    return;
  }
  SC_FILE = f;
  $("#sc-preview").innerHTML =
    `<div class="mono">${f.name} · ${(f.size / 1e6).toFixed(1)} MB</div>
     <div style="font-size:13px;color:var(--muted);margin-top:4px">
     GeoTIFF preview isn't rendered in-browser — the result map will appear after analysis.</div>`;
  $("#sc-run").disabled = false;
}

function renderScanResult(r) {
  const pct = r.class_pixel_pct;
  const colors = { Low: "#d62728", Medium: "#ff7f0e", High: "#2ca02c" };
  const bars = Object.entries(pct).map(([label, v]) => `
    <div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;font-size:13px">
        <span>${label}</span><span class="mono">${v}%</span>
      </div>
      <div style="background:var(--line);border-radius:6px;height:8px;overflow:hidden">
        <div style="width:${v}%;height:100%;background:${colors[label]}"></div>
      </div>
    </div>`).join("");

  $("#sc-result").innerHTML = `
    <img src="data:image/png;base64,${r.map_png_base64}" alt="crop-vigor map"
         style="width:100%;border-radius:10px;border:1px solid var(--line);margin-bottom:14px;image-rendering:pixelated">
    <div class="mono" style="margin-bottom:6px">Dominant class</div>
    <div class="big" style="font-size:26px;margin-bottom:14px">${r.dominant_class}</div>
    ${bars}
    <div class="note" style="margin-top:14px">${r.note}</div>`;
}

async function runScan({ sampleName } = {}) {
  const scRun = $("#sc-run");
  const busyEl = sampleName ? null : scRun;
  if (busyEl) { busyEl.disabled = true; busyEl.innerHTML = '<span class="spinner"></span> Analysing…'; }
  $("#sc-result").innerHTML = '<div class="note">Analysing…</div>';
  try {
    let r;
    if (sampleName) {
      r = await api(`/api/scanner/predict-sample/${encodeURIComponent(sampleName)}`, { method: "POST" });
    } else {
      if (!SC_FILE) return;
      const fd = new FormData();
      fd.append("file", SC_FILE);
      r = await api("/api/scanner/predict", { method: "POST", body: fd });
    }
    renderScanResult(r);
  } catch (e) {
    $("#sc-result").innerHTML = `<div class="note"><b>No prediction available.</b> ${e.message}</div>`;
  } finally {
    if (busyEl) { busyEl.disabled = false; busyEl.textContent = "Analyse patch"; }
  }
}

const scRun = $("#sc-run");
if (scRun) scRun.onclick = () => runScan();

/* ---------- methodology ---------- */

async function loadMethodology() {
  const m = await api("/api/methodology");

  $("#m-bands").innerHTML =
    `<thead><tr><th>Scenario</th><th>Pearson r</th><th>Confidence</th>
     <th>Meaning</th></tr></thead><tbody>` +
    Object.entries(m.bands).map(([k, v]) =>
      `<tr><td class="mono">${k}</td><td class="num">${v.r.toFixed(3)}</td>
       <td>${v.label}</td><td>${v.note}</td></tr>`).join("") + "</tbody>";

  $("#m-repro").innerHTML =
    `<thead><tr><th>Quantity</th><th>This work</th><th>Published</th>
     <th>Source</th></tr></thead><tbody>` +
    m.reproduction.map((r) =>
      `<tr><td>${r.quantity}</td><td class="num">${r.this_work}</td>
       <td class="num">${r.published}</td>
       <td class="mono">${r.source}</td></tr>`).join("") + "</tbody>";

  $("#m-not").innerHTML = m.not_modelled.map((t) => {
    const [head, ...rest] = t.split("—");
    return `<div class="card"><h3 style="font-size:16px">${head.trim()}</h3>
            <p>${rest.join("—").trim()}</p></div>`;
  }).join("");
}

boot().catch((e) => {
  document.querySelector("main").innerHTML =
    `<div class="note"><b>Could not reach the API.</b> ${e.message}</div>`;
});
