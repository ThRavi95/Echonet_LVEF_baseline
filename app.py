"""
EchoNet-Dynamic Clinical AI Workstation
========================================
Minimalist, Modern Diagnostic Interface for:
  • Automated Continuous LVEF Estimation (R(2+1)D-18)
  • Heart Failure Risk Stratification & Clinical Alerts
  • Dual Input Modes: Upload Cineloop & Select Random Sample
  • On-Demand 3D Grad-CAM Explainability Visualization
  • One-Click Diagnostic Report Export (JSON / TXT / GIF)

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
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.clinical.report import ClinicalReportGenerator
from src.data.io import get_random_sample_video, list_test_videos, load_video_clip
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


# ---------------------------------------------------------------------------
# GIF Rendering Helpers
# ---------------------------------------------------------------------------
def _render_raw_gif(raw_frames: np.ndarray, duration_ms: int = 55) -> str:
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


def _render_gradcam_gif(
    raw_frames: np.ndarray,
    heatmap_3d: np.ndarray,
    alpha: float = 0.5,
    duration_ms: int = 55,
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


# ---------------------------------------------------------------------------
# Minimalist HTML/CSS/JS Interface
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CardioAI · Left Ventricular Systolic Assessment</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-base: #07090e;
    --bg-card: #0e1219;
    --bg-elevated: #141a24;
    --border: rgba(255, 255, 255, 0.08);
    --border-strong: rgba(255, 255, 255, 0.16);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.25);
    --emerald: #10b981;
    --emerald-bg: rgba(16, 185, 129, 0.12);
    --amber: #f59e0b;
    --amber-bg: rgba(245, 158, 11, 0.12);
    --orange: #f97316;
    --orange-bg: rgba(249, 115, 22, 0.12);
    --crimson: #ef4444;
    --crimson-bg: rgba(239, 68, 68, 0.12);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-base);
    color: var(--text-primary);
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding-bottom: 40px;
  }

  /* Minimal Header */
  .header {
    width: 100%;
    max-width: 900px;
    padding: 24px 20px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-icon {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    border-radius: 8px;
    display: grid;
    place-items: center;
    font-size: 15px;
    font-weight: 800;
    color: #fff;
    box-shadow: 0 0 16px var(--accent-glow);
  }
  .brand-title { font-size: 16px; font-weight: 700; letter-spacing: -0.3px; }
  .brand-tag { font-size: 11px; color: var(--text-muted); font-weight: 500; }
  .status-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 14px;
    border-radius: 9999px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    font-size: 11px;
    color: var(--text-secondary);
    font-family: 'JetBrains Mono', monospace;
  }
  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
  }

  /* Main Container */
  .container {
    width: 100%;
    max-width: 900px;
    padding: 24px 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  /* Segmented Mode Control */
  .segmented-switch {
    display: flex;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 4px;
    gap: 4px;
  }
  .segment-btn {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-secondary);
    padding: 10px 16px;
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    transition: all 0.2s ease;
  }
  .segment-btn.active {
    background: var(--bg-elevated);
    color: var(--text-primary);
    border: 1px solid var(--border);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  }
  .segment-btn:hover:not(.active) { color: var(--text-primary); }

  /* Card Surfaces */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
  }

  /* Mode 1: Upload Dropzone */
  .dropzone {
    border: 1.5px dashed var(--border);
    border-radius: var(--radius-md);
    padding: 44px 20px;
    text-align: center;
    background: rgba(20, 26, 36, 0.3);
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .dropzone:hover {
    border-color: var(--accent);
    background: rgba(59, 130, 246, 0.04);
  }
  .dropzone input { display: none; }
  .dropzone-icon { font-size: 32px; margin-bottom: 10px; }
  .dropzone-title { font-size: 14px; font-weight: 600; color: var(--text-primary); }
  .dropzone-sub { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
  .file-ready-bar {
    display: none;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    margin-top: 14px;
    font-size: 13px;
  }
  .file-ready-name { color: var(--text-primary); font-family: 'JetBrains Mono', monospace; font-weight: 500; }

  /* Mode 2: Random Video Stage */
  .random-stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
  }
  .random-meta-bar {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px 16px;
    font-size: 12px;
  }
  .btn-rerandom {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--border);
    color: var(--text-primary);
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn-rerandom:hover { background: rgba(255, 255, 255, 0.12); }

  .cineloop-viewport {
    width: 240px;
    height: 240px;
    background: #000;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
    position: relative;
    display: grid;
    place-items: center;
  }
  .cineloop-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }
  .cineloop-tag {
    position: absolute;
    bottom: 8px;
    left: 8px;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: var(--text-secondary);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* Primary Action Button */
  .btn-analyze {
    width: 100%;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: var(--radius-sm);
    padding: 14px 20px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: -0.2px;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 4px 16px var(--accent-glow);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 16px;
  }
  .btn-analyze:hover:not(:disabled) { background: #2563eb; transform: translateY(-1px); }
  .btn-analyze:disabled {
    background: #1e293b;
    color: #475569;
    cursor: not-allowed;
    box-shadow: none;
    transform: none;
  }

  /* Output Results Card */
  .results-card {
    display: none;
    flex-direction: column;
    gap: 20px;
    animation: fadeIn 0.3s ease;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .results-hero {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
  }
  .ef-display { display: flex; align-items: baseline; gap: 8px; }
  .ef-number {
    font-size: 64px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -2px;
    color: #fff;
  }
  .ef-unit { font-size: 26px; font-weight: 600; color: var(--text-muted); }
  .ef-sublabel { font-size: 11px; text-transform: uppercase; color: var(--text-muted); font-weight: 600; letter-spacing: 0.8px; margin-top: 4px; }

  .risk-badge {
    padding: 8px 18px;
    border-radius: 9999px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.3px;
    border: 1px solid;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* Risk Spectrum Bar */
  .risk-meter {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .meter-bar {
    width: 100%;
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #ef4444 0%, #ef4444 30%, #f97316 30%, #f97316 40%, #f59e0b 40%, #f59e0b 50%, #10b981 50%, #10b981 100%);
    position: relative;
  }
  .meter-pin {
    position: absolute;
    top: -4px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid #000;
    box-shadow: 0 0 8px rgba(255, 255, 255, 0.8);
    transform: translateX(-50%);
    transition: left 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .meter-labels {
    display: flex;
    justify-content: space-between;
    font-size: 10px;
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
  }

  /* Diagnostic Summary Box */
  .diag-box {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 18px;
    font-size: 13.5px;
    line-height: 1.6;
    color: #cbd5e1;
  }
  .diag-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--accent);
    margin-bottom: 6px;
  }

  .alert-banner {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-left: 4px solid var(--crimson);
    border-radius: var(--radius-sm);
    padding: 12px 16px;
    font-size: 12.5px;
    color: #fecaca;
    line-height: 1.5;
  }

  /* Expandable Explainability */
  .xai-section {
    border-top: 1px solid var(--border);
    padding-top: 16px;
  }
  .btn-toggle-xai {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    padding: 8px 14px;
    border-radius: var(--radius-sm);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.15s ease;
  }
  .btn-toggle-xai:hover { border-color: var(--border-strong); color: var(--text-primary); }

  .xai-viewer {
    display: none;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 14px;
  }
  @media (max-width: 600px) { .xai-viewer { grid-template-columns: 1fr; } }

  .xai-pane {
    background: #000;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
    text-align: center;
  }
  .xai-pane img { width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }
  .xai-pane-lbl {
    padding: 8px 12px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--text-secondary);
    background: rgba(14, 18, 25, 0.8);
    border-top: 1px solid var(--border);
  }

  /* Actions Toolbar */
  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
  }
  .tool-group { display: flex; gap: 8px; }
  .btn-tool {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    padding: 8px 14px;
    border-radius: var(--radius-sm);
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn-tool:hover { background: rgba(255, 255, 255, 0.04); color: var(--text-primary); border-color: var(--border-strong); }
  .btn-reset {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 8px 14px;
    border-radius: var(--radius-sm);
    font-size: 11.5px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn-reset:hover { color: var(--crimson); border-color: var(--crimson); }
</style>
</head>
<body>

<!-- Header -->
<header class="header">
  <div class="brand">
    <div class="brand-icon">🫀</div>
    <div>
      <div class="brand-title">CardioAI</div>
      <div class="brand-tag">Left Ventricular Systolic Assessment</div>
    </div>
  </div>
  <div class="status-badge">
    <span class="status-dot"></span>
    <span>R(2+1)D-18 Baseline · Active</span>
  </div>
</header>

<!-- Main Container -->
<main class="container">

  <!-- Segmented Mode Control -->
  <div class="segmented-switch">
    <button class="segment-btn active" id="tabUpload" onclick="switchMode('upload')">
      📤 Upload Cineloop
    </button>
    <button class="segment-btn" id="tabRandom" onclick="switchMode('random')">
      🎲 Select Random Sample
    </button>
  </div>

  <!-- Input Card -->
  <section class="card" id="inputCard">

    <!-- Mode 1: Upload View -->
    <div id="uploadView">
      <div class="dropzone" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" accept=".mp4,.avi,.npy" onchange="handleFile(event)">
        <div class="dropzone-icon">📁</div>
        <div class="dropzone-title">Drop Echocardiography Cineloop Here</div>
        <div class="dropzone-sub">Supports MP4, AVI, or NPY format</div>
      </div>
      <div class="file-ready-bar" id="fileReadyBar">
        <div>
          <span style="color:var(--text-muted); font-size:11px;">READY:</span>
          <span class="file-ready-name" id="fileReadyName">--</span>
        </div>
        <span style="color:var(--emerald); font-size:12px; font-weight:600;">✓ Buffered</span>
      </div>
    </div>

    <!-- Mode 2: Select Random View -->
    <div id="randomView" style="display:none;" class="random-stage">
      <div class="random-meta-bar">
        <span>Selected Sample: <code id="randomSampleName" style="color:var(--accent);">--</code></span>
        <button class="btn-sample" onclick="fetchRandomSample()">Pick Another</button>
      </div>
      <!-- Video sample IS shown in Select Random mode -->
      <div class="cineloop-viewport">
        <img id="randomCineloopImg" class="cineloop-img" alt="Test Cineloop">
        <div class="cineloop-tag">Raw A4C Cineloop</div>
      </div>
    </div>

    <!-- Action Button -->
    <button class="btn-analyze" id="btnAnalyze" onclick="executeAnalysis()" disabled>
      Analyze Ventricular Function
    </button>
  </section>

  <!-- Results Output Card -->
  <section class="card results-card" id="resultsCard">
    <div class="results-hero">
      <div>
        <div class="ef-sublabel">Left Ventricular Ejection Fraction</div>
        <div class="ef-display">
          <span class="ef-number" id="outEF">--</span>
          <span class="ef-unit">%</span>
        </div>
      </div>
      <div class="risk-badge" id="outRiskBadge">
        --
      </div>
    </div>

    <!-- Visual Risk Gauge -->
    <div class="risk-meter">
      <div class="meter-bar">
        <div class="meter-pin" id="meterPin" style="left:50%;"></div>
      </div>
      <div class="meter-labels">
        <span>Severe &lt;30%</span>
        <span>Moderate 30-40%</span>
        <span>Mild 41-49%</span>
        <span>Normal &ge;50%</span>
      </div>
    </div>

    <!-- Action Alert (if any) -->
    <div id="alertWrapper"></div>

    <!-- Diagnostic Narrative -->
    <div class="diag-box">
      <div class="diag-title">ASE / EACVI Diagnostic Classification</div>
      <p id="outNarrative">--</p>
    </div>

    <!-- Expandable 3D Grad-CAM Explainability -->
    <div class="xai-section">
      <button class="btn-toggle-xai" id="btnToggleXai" onclick="toggleXAI()">
        <span>👁️ View 3D Spatiotemporal Attention (Grad-CAM)</span>
      </button>

      <div class="xai-viewer" id="xaiViewer">
        <div class="xai-pane">
          <img id="xaiRawImg" alt="Raw Video">
          <div class="xai-pane-lbl">Echocardiogram Cineloop</div>
        </div>
        <div class="xai-pane">
          <img id="xaiGradCamImg" alt="3D Attention">
          <div class="xai-pane-lbl" style="color:var(--emerald);">3D Grad-CAM Saliency</div>
        </div>
      </div>
    </div>

    <!-- Action Toolbar -->
    <div class="toolbar">
      <div class="tool-group">
        <button class="btn-tool" onclick="copyReport()">📋 Copy Report</button>
        <button class="btn-tool" onclick="downloadJSON()">📥 Download JSON</button>
        <button class="btn-tool" onclick="downloadTXT()">📄 Download TXT</button>
      </div>
      <button class="btn-reset" onclick="resetUI()">🔄 Analyze Another Video</button>
    </div>
  </section>

</main>

<script>
  let activeMode = 'upload';
  let selectedFile = null;
  let activePayload = null;
  let isXaiOpen = false;

  function switchMode(mode) {
    activeMode = mode;
    document.getElementById('tabUpload').classList.toggle('active', mode === 'upload');
    document.getElementById('tabRandom').classList.toggle('active', mode === 'random');
    document.getElementById('uploadView').style.display = mode === 'upload' ? 'block' : 'none';
    document.getElementById('randomView').style.display = mode === 'random' ? 'flex' : 'none';

    if (mode === 'random') {
      fetchRandomSample();
    } else {
      document.getElementById('btnAnalyze').disabled = !selectedFile;
    }
  }

  function fetchRandomSample() {
    const statusBtn = document.getElementById('btnAnalyze');
    statusBtn.disabled = true;
    document.getElementById('randomSampleName').innerText = "Loading sample...";

    fetch('/api/random_sample')
      .then(r => r.json())
      .then(d => {
        if (d.error) throw new Error(d.error);
        selectedFile = d.filename;
        document.getElementById('randomSampleName').innerText = d.filename;
        if (d.gif) {
          document.getElementById('randomCineloopImg').src = "data:image/gif;base64," + d.gif;
        }
        statusBtn.disabled = false;
      })
      .catch(e => {
        document.getElementById('randomSampleName').innerText = "Error loading sample";
      });
  }

  function handleFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    selectedFile = file.name;
    document.getElementById('fileReadyName').innerText = file.name;
    document.getElementById('fileReadyBar').style.display = 'flex';
    document.getElementById('btnAnalyze').disabled = true;

    const fd = new FormData();
    fd.append('video', file);

    fetch('/api/upload', { method: 'POST', body: fd })
      .then(r => r.json())
      .then(d => {
        if (d.error) throw new Error(d.error);
        document.getElementById('btnAnalyze').disabled = false;
      })
      .catch(e => {
        alert("Upload error: " + e.message);
      });
  }

  async function executeAnalysis() {
    const btn = document.getElementById('btnAnalyze');
    btn.disabled = true;
    btn.innerText = "Processing 32 spatiotemporal frames...";

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: selectedFile })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      activePayload = data;
      renderResults(data);
    } catch (e) {
      alert("Analysis error: " + e.message);
    } finally {
      btn.disabled = false;
      btn.innerText = "Analyze Ventricular Function";
    }
  }

  function renderResults(data) {
    const card = document.getElementById('resultsCard');
    card.style.display = 'flex';

    // 1. LVEF Number
    document.getElementById('outEF').innerText = data.predicted_lvef.toFixed(1);

    // 2. Heart Risk Badge & Colors
    const riskBadge = document.getElementById('outRiskBadge');
    const riskLevel = data.heart_risk_level || "Undetermined";
    riskBadge.innerText = riskLevel;
    riskBadge.style.color = data.badge_color;
    riskBadge.style.borderColor = data.badge_color;
    riskBadge.style.background = data.badge_color + "18";

    // 3. Risk Meter Pin Position: map LVEF [10, 75] to [0%, 100%]
    const clampedForMeter = Math.max(10, Math.min(75, data.predicted_lvef));
    const meterPct = ((clampedForMeter - 10) / (75 - 10)) * 100;
    document.getElementById('meterPin').style.left = meterPct.toFixed(1) + "%";

    // 4. Clinical Narrative
    document.getElementById('outNarrative').innerText = data.summary;

    // 5. Action Alert
    const alertWrapper = document.getElementById('alertWrapper');
    alertWrapper.innerHTML = '';
    if (data.report && data.report.clinical_alerts) {
      data.report.clinical_alerts.forEach(txt => {
        const div = document.createElement('div');
        div.className = 'alert-banner';
        div.innerText = "⚠️ " + txt;
        alertWrapper.appendChild(div);
      });
    }

    // Scroll to results smoothly
    card.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  async function toggleXAI() {
    const viewer = document.getElementById('xaiViewer');
    const btn = document.getElementById('btnToggleXai');

    if (isXaiOpen) {
      viewer.style.display = 'none';
      btn.innerText = "👁️ View 3D Spatiotemporal Attention (Grad-CAM)";
      isXaiOpen = false;
      return;
    }

    btn.innerText = "Computing 3D Saliency Heatmap...";
    btn.disabled = true;

    try {
      const res = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: selectedFile })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      if (data.xai_gif) {
        document.getElementById('xaiGradCamImg').src = "data:image/gif;base64," + data.xai_gif;
      }
      if (data.raw_gif) {
        document.getElementById('xaiRawImg').src = "data:image/gif;base64," + data.raw_gif;
      }

      viewer.style.display = 'grid';
      btn.innerText = "Hide 3D Attention Heatmap";
      isXaiOpen = true;
    } catch(e) {
      alert("Explainability error: " + e.message);
      btn.innerText = "👁️ View 3D Spatiotemporal Attention (Grad-CAM)";
    } finally {
      btn.disabled = false;
    }
  }

  function resetUI() {
    document.getElementById('resultsCard').style.display = 'none';
    document.getElementById('xaiViewer').style.display = 'none';
    isXaiOpen = false;
    document.getElementById('btnToggleXai').innerText = "👁️ View 3D Spatiotemporal Attention (Grad-CAM)";
    selectedFile = null;
    activePayload = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileReadyBar').style.display = 'none';
    document.getElementById('btnAnalyze').disabled = true;

    if (activeMode === 'random') {
      fetchRandomSample();
    }
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function copyReport() {
    if (!activePayload) return;
    const txt = `CardioAI Diagnostic Report\nLVEF: ${activePayload.predicted_lvef.toFixed(1)}%\nHeart Risk: ${activePayload.heart_risk_level}\nPhenotype: ${activePayload.clinical_category}\n\n${activePayload.summary}`;
    navigator.clipboard.writeText(txt).then(() => {
      alert("Report copied to clipboard.");
    });
  }

  function downloadJSON() {
    if (!activePayload) return;
    const blob = new Blob([JSON.stringify(activePayload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `LVEF_Report_${selectedFile || 'Sample'}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadTXT() {
    if (!activePayload) return;
    const txt = `CardioAI Diagnostic Report\nPatient / File: ${selectedFile}\nEstimated LVEF: ${activePayload.predicted_lvef.toFixed(1)}%\nHeart Risk: ${activePayload.heart_risk_level}\nClassification: ${activePayload.ase_classification} (${activePayload.clinical_category})\n\nClinical Summary:\n${activePayload.summary}\n`;
    const blob = new Blob([txt], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `LVEF_Report_${selectedFile || 'Sample'}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Server Handler
# ---------------------------------------------------------------------------
_UPLOAD_STORE: dict = {}
estimator: LVEFEstimator = None


class MinimalClinicalHandler(BaseHTTPRequestHandler):

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
            html = _HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif parsed.path == "/api/random_sample":
            self._handle_random_sample()
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

    def _handle_random_sample(self) -> None:
        sample_path = get_random_sample_video()
        if not sample_path or not os.path.exists(sample_path):
            self._send_json({"error": "No test videos found in data/test_videos/"}, 404)
            return

        fname = os.path.basename(sample_path)
        try:
            clip = load_video_clip(sample_path, start_frame=0, num_frames=16, stride=2, target_size=112)
            clip_unnorm = np.clip(clip * 0.18 + 0.12, 0.0, 1.0)
            gif_b64 = _render_raw_gif(clip_unnorm, duration_ms=55)
            self._send_json({"filename": fname, "gif": gif_b64})
        except Exception as exc:
            self._send_json({"filename": fname, "gif": "", "warning": str(exc)})

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

        # Only return status - no upfront video sample shown for upload mode
        self._send_json({"status": "ok", "filename": fname})

    def _resolve_video_path(self, fname: str) -> str:
        # Check in test_videos directory
        test_dir = os.path.join(REPO_ROOT, "data", "test_videos")
        candidate = os.path.join(test_dir, fname)
        if os.path.exists(candidate):
            return candidate

        stem = os.path.splitext(fname)[0]
        candidate_npy = os.path.join(test_dir, f"{stem}.npy")
        if os.path.exists(candidate_npy):
            return candidate_npy

        # Check in upload store
        if fname in _UPLOAD_STORE:
            tmp_path = os.path.join(tempfile.gettempdir(), fname)
            with open(tmp_path, "wb") as f:
                f.write(_UPLOAD_STORE[fname])
            return tmp_path

        raise FileNotFoundError(f"Video file '{fname}' not found.")

    def _handle_predict(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        fname = body.get("filename", "")

        try:
            target_path = self._resolve_video_path(fname)
            result = estimator.predict(target_path)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _handle_explain(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        fname = body.get("filename", "")

        try:
            target_path = self._resolve_video_path(fname)
            raw_frames, heatmap_3d, clip_ef = estimator.explain(target_path, start_frame=0)
            raw_gif = _render_raw_gif(raw_frames, duration_ms=55)
            xai_gif = _render_gradcam_gif(raw_frames, heatmap_3d, alpha=0.5, duration_ms=55)

            report_gen = ClinicalReportGenerator()
            report_dict = report_gen.generate_report(
                patient_id=fname,
                predicted_ef=clip_ef,
            )

            classification = report_dict["classification"]
            response_data = {
                "predicted_lvef": round(clip_ef, 1),
                "heart_risk_level": classification.get("heart_risk_level", "Undetermined"),
                "clinical_category": classification["phenotype"],
                "badge_color": classification["badge_color"],
                "ase_classification": classification["category"],
                "summary": report_dict["narrative"],
                "report": report_dict,
                "raw_gif": raw_gif,
                "xai_gif": xai_gif,
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

    server = HTTPServer(("127.0.0.1", port), MinimalClinicalHandler)
    print("=" * 65)
    print("  CARDIOAI · MINIMALIST CLINICAL LVEF WORKSTATION")
    print(f"  -> Local Web UI:    http://127.0.0.1:{port}")
    print(f"  -> Checkpoint Path: {checkpoint_path}")
    print("=" * 65)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[app] Server shutdown requested.")
        server.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8501
    ckpt_arg = sys.argv[2] if len(sys.argv) > 2 else ""
    run_server(port=port_arg, checkpoint_arg=ckpt_arg)
