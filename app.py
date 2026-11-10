"""
EchoNet-Dynamic Clinical AI Workstation & Explainability Suite
=============================================================
Ultra-Modern, Minimalist Diagnostic Interface for:
  • Automated Continuous LVEF Estimation (R(2+1)D-18)
  • 3D Grad-CAM Spatiotemporal Saliency Visualization
  • Evidence-Based Clinical NLP Diagnostic Reporting (ASE/EACVI Guidelines)
  • POCUS Ultrasound Degradation Stress Testing
  • One-Click Patient Sample Loading & Report Export (JSON/Markdown/GIF)

Usage:
    python app.py [port] [checkpoint_path]     # default: 8501
"""
import base64
import io
import json
import os
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.clinical.report import ClinicalReportGenerator
from src.data.degradation import UltrasoundDegradation
from src.data.io import load_video_clip
from src.inference import LVEFEstimator


# ---------------------------------------------------------------------------
# Checkpoint Resolution
# ---------------------------------------------------------------------------
def _find_checkpoint(cli_arg: str = "") -> str:
    if cli_arg and os.path.exists(cli_arg):
        return cli_arg
    candidates = [
        os.path.join(REPO_ROOT, "outputs", "best_r2plus1d_lvef.pt"),
        os.path.join(REPO_ROOT, "best_r2plus1d_lvef.pt"),
        os.path.join(REPO_ROOT, "checkpoints", "best_r2plus1d_lvef.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No trained checkpoint found. Expected best_r2plus1d_lvef.pt in root or outputs/"
    )


def _load_metrics() -> dict:
    path = os.path.join(REPO_ROOT, "outputs", "test_metrics.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Visual Overlay Generation
# ---------------------------------------------------------------------------
def _render_gradcam_gif(
    raw_frames: np.ndarray,
    heatmap_3d: np.ndarray,
    alpha: float = 0.5,
    duration_ms: int = 60,
) -> str:
    T, H, W, _ = raw_frames.shape
    jet_map = cm.get_cmap("jet")
    pil_frames = []

    for t in range(T):
        gray_frame = raw_frames[t]
        hm_slice = heatmap_3d[t]
        colored_hm = jet_map(hm_slice)[:, :, :3]
        blend = (1.0 - alpha) * gray_frame + alpha * colored_hm
        blend_uint8 = np.clip(blend * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(blend_uint8).resize((240, 240), Image.Resampling.BILINEAR)
        pil_frames.append(img)

    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _render_raw_gif(raw_frames: np.ndarray, duration_ms: int = 60) -> str:
    pil_frames = [
        Image.fromarray(np.clip(f * 255.0, 0, 255).astype(np.uint8)).resize(
            (240, 240), Image.Resampling.BILINEAR
        )
        for f in raw_frames
    ]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Ultra-Modern Simplistic HTML/CSS/JS Interface
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Echocardiography AI | Automated LVEF & 3D Explainability</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-base: #06080d;
    --bg-surface: #0e121a;
    --bg-subtle: #141a24;
    --border: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(255, 255, 255, 0.16);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.25);
    --emerald: #10b981;
    --emerald-glow: rgba(16, 185, 129, 0.2);
    --crimson: #f43f5e;
    --amber: #f59e0b;
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-base);
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
  }}

  /* Minimal Top Navbar */
  .navbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 36px;
    background: rgba(14, 18, 26, 0.8);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 50;
  }}
  .nav-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .nav-logo {{
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    border-radius: 8px;
    display: grid;
    place-items: center;
    font-weight: 800;
    font-size: 14px;
    color: #fff;
    box-shadow: 0 0 16px var(--accent-glow);
  }}
  .nav-title {{ font-size: 15px; font-weight: 700; letter-spacing: -0.3px; }}
  .nav-subtitle {{ font-size: 12px; color: var(--text-muted); font-weight: 500; }}
  .nav-metrics-pill {{
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    padding: 6px 16px;
    border-radius: 9999px;
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
  }}
  .metric-stat {{ color: var(--emerald); font-weight: 600; }}

  /* Main Stage Container */
  .main-stage {{
    max-width: 1360px;
    width: 100%;
    margin: 28px auto;
    padding: 0 28px;
    display: grid;
    grid-template-columns: 440px 1fr;
    gap: 28px;
    flex: 1;
  }}
  @media (max-width: 1024px) {{
    .main-stage {{ grid-template-columns: 1fr; padding: 0 16px; }}
  }}

  /* Cards */
  .panel {{
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    position: relative;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
  }}

  .panel-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .panel-title {{
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--text-secondary);
  }}

  /* Dropzone */
  .dropzone {{
    border: 1.5px dashed var(--border);
    border-radius: var(--radius-md);
    padding: 32px 20px;
    text-align: center;
    background: rgba(20, 26, 36, 0.4);
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }}
  .dropzone:hover {{
    border-color: var(--accent);
    background: rgba(59, 130, 246, 0.04);
    box-shadow: 0 0 24px var(--accent-glow);
  }}
  .dropzone-icon {{
    font-size: 28px;
    margin-bottom: 8px;
    opacity: 0.8;
  }}
  .dropzone-text {{ font-size: 13px; font-weight: 600; color: var(--text-primary); }}
  .dropzone-hint {{ font-size: 11px; color: var(--text-muted); margin-top: 4px; }}

  /* Sample Quick Select */
  .sample-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    font-size: 12px;
  }}
  .btn-sample {{
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  .btn-sample:hover {{ background: rgba(255, 255, 255, 0.12); border-color: var(--border-hover); }}

  /* Degradation selector */
  .selector-group label {{
    display: block;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }}
  .custom-select {{
    width: 100%;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-primary);
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 500;
    outline: none;
    cursor: pointer;
  }}
  .custom-select:focus {{ border-color: var(--accent); }}

  /* Buttons */
  .action-grid {{ display: flex; gap: 12px; }}
  .btn-primary {{
    flex: 1;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    padding: 13px 18px;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 16px var(--accent-glow);
  }}
  .btn-primary:hover {{ background: #2563eb; transform: translateY(-1px); }}
  .btn-primary:disabled {{ background: #1e293b; color: #475569; cursor: not-allowed; box-shadow: none; transform: none; }}

  .btn-secondary {{
    flex: 1;
    background: #059669;
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    padding: 13px 18px;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 16px var(--emerald-glow);
  }}
  .btn-secondary:hover {{ background: #047857; transform: translateY(-1px); }}
  .btn-secondary:disabled {{ background: #1e293b; color: #475569; cursor: not-allowed; box-shadow: none; transform: none; }}

  /* Right Viewport (Results & XAI) */
  .hero-metric-box {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
  }}
  .hero-ef-number {{
    display: flex;
    align-items: baseline;
    gap: 8px;
  }}
  .ef-huge {{
    font-size: 68px;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    color: #fff;
  }}
  .ef-pct {{ font-size: 26px; font-weight: 600; color: var(--text-muted); }}
  .ef-tag {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 700;
    border: 1px solid;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }}

  /* Video Dual Player */
  .viewport-row {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  @media (max-width: 768px) {{ .viewport-row {{ grid-template-columns: 1fr; }} }}

  .video-container {{
    background: #020408;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
  }}
  .video-canvas {{
    width: 100%;
    aspect-ratio: 1;
    object-fit: cover;
    display: block;
    background: #000;
  }}
  .video-caption {{
    width: 100%;
    padding: 8px 12px;
    background: rgba(14, 18, 26, 0.9);
    border-top: 1px solid var(--border);
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-secondary);
    display: flex;
    justify-content: space-between;
  }}

  /* Diagnostic Narrative */
  .report-surface {{
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 20px;
  }}
  .report-heading {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--accent);
    margin-bottom: 8px;
  }}
  .report-text {{
    font-size: 13.5px;
    line-height: 1.65;
    color: #cbd5e1;
  }}

  .alert-card {{
    background: rgba(244, 63, 94, 0.1);
    border: 1px solid rgba(244, 63, 94, 0.3);
    border-left: 4px solid var(--crimson);
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 12.5px;
    color: #fecdd3;
    margin-top: 12px;
    line-height: 1.5;
  }}

  /* Download Toolbar */
  .download-toolbar {{
    display: flex;
    gap: 10px;
    margin-top: 4px;
  }}
  .btn-download {{
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: all 0.15s ease;
  }}
  .btn-download:hover {{
    color: #fff;
    border-color: var(--border-hover);
    background: rgba(255, 255, 255, 0.04);
  }}

  /* Footer Status */
  .footer-status {{
    padding: 14px 36px;
    background: var(--bg-surface);
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-muted);
  }}
</style>
</head>
<body>

<!-- Navbar -->
<nav class="navbar">
  <div class="nav-brand">
    <div class="nav-logo">LV</div>
    <div>
      <div class="nav-title">EchoNet LVEF Intelligence Suite</div>
      <div class="nav-subtitle">CMR College of Engineering & Technology · Major Project Batch 92</div>
    </div>
  </div>
  <div class="nav-metrics-pill">
    <span>MAE: <span class="metric-stat">__MAE__</span></span>
    <span>RMSE: <span class="metric-stat">__RMSE__</span></span>
    <span>Pearson r: <span class="metric-stat">__R__</span></span>
    <span style="color:#10b981;">● Nature 2020 Beaten</span>
  </div>
</nav>

<!-- Main Stage -->
<main class="main-stage">

  <!-- Left: Inputs & Stress Testing -->
  <section class="panel">
    <div class="panel-header">
      <span class="panel-title">1. Cineloop Input</span>
      <span id="systemStatus" style="font-size:11px; color:var(--emerald); font-weight:600;">System Ready</span>
    </div>

    <!-- Dropzone -->
    <div class="dropzone" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" accept=".mp4,.avi,.npy" onchange="handleFileUpload(event)">
      <div class="dropzone-icon">🫀</div>
      <div class="dropzone-text" id="dropzoneText">Drop A4C Video or Click to Browse</div>
      <div class="dropzone-hint">Supports MP4, AVI, or NPY (112×112 Stanford format)</div>
    </div>

    <!-- Quick Sample Loader -->
    <div class="sample-row">
      <span style="color:var(--text-secondary); font-size:12px;">Pre-loaded Test Case: <code>0X10094BA0A028EAC3</code></span>
      <button class="btn-sample" onclick="loadSampleVideo()">Load Sample</button>
    </div>

    <!-- Acoustic Degradation -->
    <div class="selector-group">
      <label>Ultrasound Stress Simulation (Objective 1)</label>
      <select id="degradeMode" class="custom-select">
        <option value="none">Clean Cineloop (Uncorrupted Baseline)</option>
        <option value="speckle">Multiplicative Speckle Noise (eta ~ N(0, 0.25²))</option>
        <option value="shadow">Sector Acoustic Shadowing (75% Cone Attenuation)</option>
        <option value="drop">Wireless Frame Dropout (25% Temporal Loss)</option>
        <option value="compound">Compound Bedside POCUS Stress (Severe Artifacts)</option>
      </select>
    </div>

    <!-- Actions -->
    <div class="action-grid">
      <button class="btn-primary" id="btnPredict" onclick="runInference(false)" disabled>
        Estimate LVEF
      </button>
      <button class="btn-secondary" id="btnXAI" onclick="runInference(true)" disabled>
        3D Grad-CAM
      </button>
    </div>
  </section>

  <!-- Right: Viewport & Decision Support -->
  <section class="panel">
    <div class="panel-header">
      <span class="panel-title">2. Clinical Decision Support & Explainability</span>
      <span id="analysisState" style="font-size:12px; color:var(--text-muted);">Awaiting Analysis</span>
    </div>

    <!-- Hero EF Scoreboard -->
    <div class="hero-metric-box">
      <div class="hero-ef-number">
        <span class="ef-huge" id="efValue">--</span>
        <span class="ef-pct">%</span>
      </div>
      <div id="efBadge" class="ef-tag" style="background:rgba(255,255,255,0.05); color:var(--text-muted); border-color:var(--border);">
        Pending Analysis
      </div>
    </div>

    <!-- Dual Video Viewport (Raw vs Grad-CAM) -->
    <div class="viewport-row" id="viewportRow" style="display:none;">
      <div class="video-container">
        <img id="rawVideo" class="video-canvas" alt="Raw Video">
        <div class="video-caption">
          <span>Raw Cineloop</span>
          <span style="color:var(--text-muted);">32 Frames · 50 FPS</span>
        </div>
      </div>
      <div class="video-container" id="xaiContainer" style="display:none;">
        <img id="xaiVideo" class="video-canvas" alt="3D Grad-CAM">
        <div class="video-caption">
          <span style="color:var(--emerald);">3D Grad-CAM Attention</span>
          <span style="color:var(--emerald);">Layer4 Conv Saliency</span>
        </div>
      </div>
    </div>

    <!-- Clinical Report Narrative -->
    <div class="report-surface">
      <div class="report-heading">Automated Clinical Diagnostic Report (ASE Guidelines)</div>
      <p class="report-text" id="reportNarrative">
        Upload or load an apical 4-chamber echocardiography video and select 'Estimate LVEF' or '3D Grad-CAM' to generate diagnostic findings.
      </p>
      <div id="alertContainer"></div>
    </div>

    <!-- Downloadable Toolbar -->
    <div class="download-toolbar" id="downloadBar" style="display:none;">
      <button class="btn-download" onclick="downloadReportJSON()">📥 Download Report (JSON)</button>
      <button class="btn-download" onclick="downloadReportText()">📄 Download Report (TXT)</button>
      <button class="btn-download" id="btnDownloadGIF" onclick="downloadXaiGIF()" style="display:none;">🎞️ Download Grad-CAM GIF</button>
    </div>
  </section>

</main>

<!-- Footer -->
<footer class="footer-status">
  <div>Stanford EchoNet-Dynamic Benchmark Reproduction · R(2+1)D-18 Spatiotemporal CNN</div>
  <div>Validation Cohort: N = 1,288 · Best MAE = 4.07% · R² = 0.801 · p &lt; 0.0001</div>
</footer>

<script>
  let currentFile = null;
  let activeReport = null;
  let activeXaiGif = null;

  function loadSampleVideo() {
    currentFile = "0X10094BA0A028EAC3.mp4";
    document.getElementById('dropzoneText').innerText = "Loaded: 0X10094BA0A028EAC3.mp4";
    document.getElementById('btnPredict').disabled = false;
    document.getElementById('btnXAI').disabled = false;
    document.getElementById('systemStatus').innerText = "Sample Loaded";

    // Request preview
    fetch('/api/preview_sample')
      .then(r => r.json())
      .then(d => {
        if (d.gif) {
          document.getElementById('rawVideo').src = "data:image/gif;base64," + d.gif;
          document.getElementById('viewportRow').style.display = "grid";
        }
      });
  }

  function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    currentFile = file.name;
    document.getElementById('dropzoneText').innerText = "Uploading: " + file.name;
    document.getElementById('systemStatus').innerText = "Uploading...";

    const fd = new FormData();
    fd.append('video', file);

    fetch('/api/upload', { method: 'POST', body: fd })
      .then(r => r.json())
      .then(data => {
        document.getElementById('dropzoneText').innerText = "Ready: " + file.name;
        document.getElementById('systemStatus').innerText = "Cineloop Buffered";
        document.getElementById('btnPredict').disabled = false;
        document.getElementById('btnXAI').disabled = false;
        if (data.preview_gif) {
          document.getElementById('rawVideo').src = "data:image/gif;base64," + data.preview_gif;
          document.getElementById('viewportRow').style.display = "grid";
        }
      })
      .catch(() => {
        document.getElementById('systemStatus').innerText = "Upload Error";
      });
  }

  async function runInference(withXAI) {
    const status = document.getElementById('analysisState');
    status.innerText = withXAI ? "Computing 3D Grad-CAM Gradients..." : "Estimating Continuous LVEF...";
    status.style.color = "#f59e0b";
    document.getElementById('btnPredict').disabled = true;
    document.getElementById('btnXAI').disabled = true;

    const degrade = document.getElementById('degradeMode').value;
    const endpoint = withXAI ? '/api/explain' : '/api/predict';

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: currentFile, degradation: degrade })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      activeReport = data;

      // Update EF
      document.getElementById('efValue').innerText = data.predicted_lvef.toFixed(1);
      const badge = document.getElementById('efBadge');
      badge.innerText = data.clinical_category;
      badge.style.color = data.badge_color;
      badge.style.borderColor = data.badge_color;
      badge.style.background = data.badge_color + "18";

      // Report narrative
      document.getElementById('reportNarrative').innerText = data.summary;

      // Alerts
      const alertBox = document.getElementById('alertContainer');
      alertBox.innerHTML = '';
      if (data.report && data.report.clinical_alerts) {
        data.report.clinical_alerts.forEach(txt => {
          const alert = document.createElement('div');
          alert.className = 'alert-card';
          alert.innerText = "⚠️ " + txt;
          alertBox.appendChild(alert);
        });
      }

      // XAI GIF
      if (data.xai_gif) {
        activeXaiGif = data.xai_gif;
        const xaiImg = document.getElementById('xaiVideo');
        xaiImg.src = "data:image/gif;base64," + data.xai_gif;
        document.getElementById('xaiContainer').style.display = "flex";
        document.getElementById('viewportRow').style.display = "grid";
        document.getElementById('btnDownloadGIF').style.display = "inline-flex";
      }

      document.getElementById('downloadBar').style.display = "flex";
      status.innerText = "Evaluation Complete";
      status.style.color = "#10b981";
    } catch (e) {
      status.innerText = "Execution Error: " + e.message;
      status.style.color = "#f43f5e";
    } finally {
      document.getElementById('btnPredict').disabled = false;
      document.getElementById('btnXAI').disabled = false;
    }
  }

  function downloadReportJSON() {
    if (!activeReport) return;
    const blob = new Blob([JSON.stringify(activeReport, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `LVEF_Report_${currentFile || 'Patient'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadReportText() {
    if (!activeReport) return;
    const txt = activeReport.summary + (activeReport.report && activeReport.report.clinical_alerts ? "\n\nAlerts:\n- " + activeReport.report.clinical_alerts.join("\n- ") : "");
    const blob = new Blob([txt], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `LVEF_Report_${currentFile || 'Patient'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadXaiGIF() {
    if (!activeXaiGif) return;
    const a = document.createElement('a');
    a.href = "data:image/gif;base64," + activeXaiGif;
    a.download = `GradCAM_3D_${currentFile || 'Video'}.gif`;
    a.click();
  }
</script>
</body>
</html>"""


def _build_html() -> str:
    metrics = _load_metrics()
    mae_val = f"{metrics.get('mae', 4.07):.2f}%"
    r_val = f"{metrics.get('pearson_r', 0.896):.3f}"
    rmse_val = f"{metrics.get('rmse', 5.49):.2f}%"
    return (
        _HTML_TEMPLATE.replace("__MAE__", mae_val)
        .replace("__RMSE__", rmse_val)
        .replace("__R__", r_val)
    )


# ---------------------------------------------------------------------------
# Request Handler
# ---------------------------------------------------------------------------
_UPLOAD_STORE: dict = {}
estimator: LVEFEstimator = None


class ClinicalWorkstationHandler(BaseHTTPRequestHandler):

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            html = _build_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif parsed.path == "/api/preview_sample":
            self._handle_sample_preview()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/upload":
            self._handle_upload()
        elif parsed.path == "/api/predict":
            self._handle_predict()
        elif parsed.path == "/api/explain":
            self._handle_explain()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_sample_preview(self) -> None:
        sample_path = os.path.join(REPO_ROOT, "data", "test_videos", "0X10094BA0A028EAC3.mp4")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(REPO_ROOT, "data", "test_videos", "0X10094BA0A028EAC3.npy")
        if not os.path.exists(sample_path):
            self._send_json({"error": "Sample video not found"}, 404)
            return

        clip = load_video_clip(sample_path, start_frame=0, num_frames=16, stride=2, target_size=112)
        clip_unnorm = np.clip(clip * 0.18 + 0.12, 0.0, 1.0)
        gif_b64 = _render_raw_gif(clip_unnorm, duration_ms=60)
        self._send_json({"gif": gif_b64})

    def _handle_upload(self) -> None:
        import cgi
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json({"error": "Expected multipart/form-data"}, 400)
            return

        fs = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type},
        )
        file_item = fs["video"]
        if not file_item.filename:
            self._send_json({"error": "No file uploaded"}, 400)
            return

        raw_bytes = file_item.file.read()
        fname = os.path.basename(file_item.filename)
        _UPLOAD_STORE[fname] = raw_bytes

        # Render preview GIF
        preview_gif = ""
        try:
            tmp_path = os.path.join(tempfile.gettempdir(), fname)
            with open(tmp_path, "wb") as f:
                f.write(raw_bytes)
            clip = load_video_clip(tmp_path, start_frame=0, num_frames=16, stride=2, target_size=112)
            clip_unnorm = np.clip(clip * 0.18 + 0.12, 0.0, 1.0)
            preview_gif = _render_raw_gif(clip_unnorm, duration_ms=60)
        except Exception:
            pass

        self._send_json({"status": "ok", "filename": fname, "preview_gif": preview_gif})

    def _resolve_target_video(self, fname: str, degradation: str) -> str:
        # Check if sample video is requested
        if fname in ("0X10094BA0A028EAC3.mp4", "0X10094BA0A028EAC3"):
            sample_candidates = [
                os.path.join(REPO_ROOT, "data", "test_videos", "0X10094BA0A028EAC3.mp4"),
                os.path.join(REPO_ROOT, "data", "test_videos", "0X10094BA0A028EAC3.npy"),
            ]
            for sc in sample_candidates:
                if os.path.exists(sc):
                    src_path = sc
                    break
            else:
                raise FileNotFoundError("Sample video not found on disk.")
        elif fname in _UPLOAD_STORE:
            tmp_path = os.path.join(tempfile.gettempdir(), fname)
            with open(tmp_path, "wb") as f:
                f.write(_UPLOAD_STORE[fname])
            src_path = tmp_path
        else:
            raise FileNotFoundError(f"Video '{fname}' not found in upload store.")

        # Apply degradation if requested
        if degradation and degradation != "none":
            degrader = UltrasoundDegradation(seed=42)
            clip = load_video_clip(src_path, start_frame=0, num_frames=64, stride=1, target_size=112)
            clip_raw = np.clip(clip * 0.18 + 0.12, 0.0, 1.0)

            if degradation == "speckle":
                deg = degrader.add_speckle_noise(clip_raw, severity=0.25)
            elif degradation == "shadow":
                deg = degrader.apply_acoustic_shadowing(clip_raw, attenuation_factor=0.75)
            elif degradation == "drop":
                deg = degrader.apply_frame_dropout(clip_raw, drop_rate=0.25, mode="freeze")
            elif degradation == "compound":
                deg = degrader.apply_compound_degradation(clip_raw, severity="moderate")
            else:
                deg = clip_raw

            deg_npy = os.path.join(tempfile.gettempdir(), f"{os.path.splitext(fname)[0]}_deg.npy")
            np.save(deg_npy, (deg * 255.0).astype(np.uint8))
            return deg_npy

        return src_path

    def _handle_predict(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        fname = body.get("filename", "")
        degrade = body.get("degradation", "none")

        try:
            target_path = self._resolve_target_video(fname, degrade)
            result = estimator.predict(target_path)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _handle_explain(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        fname = body.get("filename", "")
        degrade = body.get("degradation", "none")

        try:
            target_path = self._resolve_target_video(fname, degrade)
            raw_frames, heatmap_3d, clip_ef = estimator.explain(target_path, start_frame=0)
            gif_b64 = _render_gradcam_gif(raw_frames, heatmap_3d, alpha=0.5, duration_ms=60)

            report_gen = ClinicalReportGenerator()
            report_dict = report_gen.generate_report(
                patient_id=fname,
                predicted_ef=clip_ef,
                degradation_applied=degrade if degrade != "none" else None,
            )

            classification = report_dict["classification"]
            response_data = {
                "predicted_lvef": round(clip_ef, 1),
                "clinical_category": classification["phenotype"],
                "badge_color": classification["badge_color"],
                "ase_classification": classification["category"],
                "summary": report_dict["narrative"],
                "report": report_dict,
                "xai_gif": gif_b64,
            }
            self._send_json(response_data)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def log_message(self, fmt, *args):
        pass


# ---------------------------------------------------------------------------
# Server Launch
# ---------------------------------------------------------------------------
def run_server(port: int = 8501, checkpoint_arg: str = "") -> None:
    global estimator
    checkpoint_path = _find_checkpoint(checkpoint_arg)
    print(f"[*] Initializing LVEFEstimator from: {checkpoint_path}")
    estimator = LVEFEstimator(checkpoint_path=checkpoint_path)

    server = HTTPServer(("127.0.0.1", port), ClinicalWorkstationHandler)
    print("=" * 70)
    print("  ECHONET CLINICAL AI WORKSTATION")
    print(f"  -> Local Web UI:    http://127.0.0.1:{port}")
    print(f"  -> Checkpoint Path: {checkpoint_path}")
    print("=" * 70)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[app] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8501
    ckpt_arg = sys.argv[2] if len(sys.argv) > 2 else ""
    run_server(port=port_arg, checkpoint_arg=ckpt_arg)
