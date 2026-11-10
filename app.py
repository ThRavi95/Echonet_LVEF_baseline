"""
EchoNet-Dynamic Clinical AI & Explainability Dashboard
======================================================
Interactive clinical decision support interface demonstrating:
  1. Automated Continuous LVEF Estimation (R(2+1)D-18 baseline)
  2. Spatiotemporal 3D Grad-CAM Explainability (synchronized video playback)
  3. Evidence-Based Clinical NLP Diagnostic Reporting (ASE/EACVI guidelines)
  4. Real-Time Ultrasound Degradation Stress Testing (speckle, shadow, frame drops)
  5. Quantitative Benchmark Validation (MAE, RMSE, IoU fidelity on Stanford test set)

Usage:
    python app.py [port] [checkpoint_path]     # default port: 8501
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
from src.data.degradation import UltrasoundDegradation
from src.data.io import load_video_clip
from src.inference import LVEFEstimator


# ---------------------------------------------------------------------------
# Checkpoint resolution — strictly requires a valid trained model
# ---------------------------------------------------------------------------
def _find_checkpoint(cli_arg: str = "") -> str:
    if cli_arg and os.path.exists(cli_arg):
        return cli_arg
    candidates = [
        os.path.join(REPO_ROOT, "outputs", "best_r2plus1d_lvef.pt"),
        os.path.join(REPO_ROOT, "checkpoints", "best_r2plus1d_lvef.pt"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No trained checkpoint found.\n"
        "Expected one of:\n"
        "  outputs/best_r2plus1d_lvef.pt\n"
        "  checkpoints/best_r2plus1d_lvef.pt\n\n"
        "Train the model first on Kaggle GPU:\n"
        "  python kaggle/train_kaggle.py --data_dir /path/to/EchoNet-Dynamic\n"
        "Then place the saved best_r2plus1d_lvef.pt in outputs/"
    )


# ---------------------------------------------------------------------------
# Load evaluation metrics if generated
# ---------------------------------------------------------------------------
def _load_json_file(filename: str) -> dict:
    path = os.path.join(REPO_ROOT, "outputs", filename)
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _build_metrics_html() -> str:
    test_metrics = _load_json_file("test_metrics.json")
    xai_metrics = _load_json_file("xai_metrics.json")

    if not test_metrics and not xai_metrics:
        return '<div class="metrics-pending">Model benchmarks pending training and evaluation on Stanford EchoNet-Dynamic test set.</div>'

    sections = []

    # Section A: LVEF Prediction Accuracy
    if test_metrics:
        sections.append(f"""
        <div class="metrics-subhdr">A. Continuous LVEF Prediction Accuracy (Ouyang et al. Reproduction)</div>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-title">Mean Absolute Error (MAE)</div>
            <div class="metric-val">{test_metrics.get('mae', 0):.2f}%</div>
            <div class="metric-sub">Benchmark: &le; 4.5%</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Root Mean Squared Error</div>
            <div class="metric-val">{test_metrics.get('rmse', 0):.2f}%</div>
            <div class="metric-sub">Penalises severe outliers</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Pearson Correlation (r)</div>
            <div class="metric-val">{test_metrics.get('pearson_r', 0):.3f}</div>
            <div class="metric-sub">Linear clinical agreement</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">R&sup2; Score</div>
            <div class="metric-val">{test_metrics.get('r2', 0):.3f}</div>
            <div class="metric-sub">Explained variance</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Bland-Altman Bias</div>
            <div class="metric-val">{test_metrics.get('bland_altman_bias', 0):.2f}%</div>
            <div class="metric-sub">LoA: [{test_metrics.get('bland_altman_lower_loa', 0):.1f}%, {test_metrics.get('bland_altman_upper_loa', 0):.1f}%]</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Low EF (&lt;50%) Accuracy</div>
            <div class="metric-val">{test_metrics.get('binary_low_ef_accuracy', 0)*100:.1f}%</div>
            <div class="metric-sub">Sens: {test_metrics.get('binary_low_ef_sensitivity', 0)*100:.1f}% | Spec: {test_metrics.get('binary_low_ef_specificity', 0)*100:.1f}%</div>
          </div>
        </div>
        """)

    # Section B: Quantitative XAI Explanation Fidelity
    if xai_metrics:
        overall = xai_metrics.get("overall_fidelity", {})
        ed = xai_metrics.get("ed_metrics", {})
        es = xai_metrics.get("es_metrics", {})
        sections.append(f"""
        <div class="metrics-subhdr" style="margin-top:20px;">B. Quantitative 3D Grad-CAM Explanation Fidelity (Objective 4: Anatomical Alignment)</div>
        <div class="metrics-grid">
          <div class="metric-card">
            <div class="metric-title">Mean Anatomical IoU</div>
            <div class="metric-val" style="color:#10b981;">{overall.get('iou_mean', 0):.3f}</div>
            <div class="metric-sub">ED: {ed.get('iou_mean', 0):.3f} | ES: {es.get('iou_mean', 0):.3f}</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">S&oslash;rensen-Dice Score</div>
            <div class="metric-val" style="color:#10b981;">{overall.get('dice_mean', 0):.3f}</div>
            <div class="metric-sub">Harmonic spatial overlap</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Pointing Game Accuracy</div>
            <div class="metric-val" style="color:#38bdf8;">{overall.get('pointing_game_accuracy', 0)*100:.1f}%</div>
            <div class="metric-sub">Peak inside LV chamber</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Chamber Energy Fraction</div>
            <div class="metric-val" style="color:#38bdf8;">{overall.get('energy_fraction_mean', 0)*100:.1f}%</div>
            <div class="metric-sub">Total attention inside chamber</div>
          </div>
          <div class="metric-card">
            <div class="metric-title">Fidelity vs. Error Correlation</div>
            <div class="metric-val">{overall.get('pearson_correlation_iou_vs_ef_error', 0):.3f}</div>
            <div class="metric-sub">Pearson r (Fidelity vs MAE)</div>
          </div>
        </div>
        """)

    return "".join(sections)


# ---------------------------------------------------------------------------
# Visual Overlay & GIF Generation
# ---------------------------------------------------------------------------
def _generate_overlay_gif(
    raw_frames: np.ndarray,
    heatmap_3d: np.ndarray,
    alpha: float = 0.5,
    duration_ms: int = 60,
) -> str:
    """
    Renders 32 video frames with synchronized 3D Grad-CAM colormap overlays
    into an animated base64 GIF.
    """
    T, H, W, _ = raw_frames.shape
    jet_map = cm.get_cmap("jet")
    pil_frames = []

    for t in range(T):
        gray_frame = raw_frames[t]  # (H, W, 3) in [0, 1]
        hm_slice = heatmap_3d[t]    # (H, W) in [0, 1]

        # Apply Jet colormap to heatmap
        colored_hm = jet_map(hm_slice)[:, :, :3]  # (H, W, 3) float [0, 1]

        # Alpha composite
        blend = (1.0 - alpha) * gray_frame + alpha * colored_hm
        blend_uint8 = np.clip(blend * 255.0, 0, 255).astype(np.uint8)

        # Scale up 2x for crisp browser rendering: (224, 224)
        img = Image.fromarray(blend_uint8).resize((224, 224), Image.Resampling.BILINEAR)
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
# HTML Dashboard Template
# ---------------------------------------------------------------------------
def _build_html() -> str:
    metrics_html = _build_metrics_html()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EchoNet-Dynamic | AI Echocardiography & Explainability</title>
<style>
  :root {{
    --bg: #090d16; --bg2: #111827; --card: #1e293b;
    --border: #334155; --blue: #2563eb; --blue-hover: #1d4ed8;
    --cyan: #0ea5e9; --text: #f8fafc; --muted: #94a3b8;
    --ok: #10b981; --warn: #f59e0b; --danger: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0;
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
  body {{ background: var(--bg); color: var(--text); min-height: 100vh; padding: 24px; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  header {{ display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid var(--border); padding-bottom: 20px; margin-bottom: 24px; }}
  header h1 {{ font-size: 24px; font-weight: 800;
               background: linear-gradient(90deg, #60a5fa, #a855f7);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  header p {{ font-size: 13px; color: var(--muted); margin-top: 4px; }}
  .badge {{ background: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3;
            padding: 6px 14px; border-radius: 9999px; font-size: 12px; font-weight: 700; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1.2fr; gap: 24px; margin-bottom: 28px; }}
  @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 22px; }}
  .card-hdr {{ display: flex; justify-content: space-between; align-items: center;
               border-bottom: 1px solid #2d3748; padding-bottom: 12px; margin-bottom: 16px; }}
  .card-title {{ font-size: 15px; font-weight: 700; color: #f1f5f9; }}
  .upload-zone {{ border: 2px dashed #475569; border-radius: 10px; padding: 24px;
                  text-align: center; cursor: pointer; transition: all .2s; background: #0f172a88; }}
  .upload-zone:hover {{ border-color: var(--cyan); background: #0f172a; }}
  .upload-zone input {{ display: none; }}
  .upload-zone label {{ cursor: pointer; color: var(--muted); font-size: 13px; }}
  .controls-box {{ margin-top: 14px; background: #1e293b55; border: 1px solid #334155; border-radius: 8px; padding: 12px; }}
  .controls-title {{ font-size: 11px; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px; }}
  .select-degrade {{ width: 100%; background: #0f172a; color: #e2e8f0; border: 1px solid #475569;
                     padding: 8px 12px; border-radius: 6px; font-size: 13px; }}
  .btn-group {{ display: flex; gap: 10px; margin-top: 14px; }}
  .btn {{ background: var(--blue); color: #fff; border: none; padding: 11px 18px;
          border-radius: 8px; font-weight: 700; cursor: pointer; flex: 1;
          transition: background .2s; font-size: 13px; text-align: center; }}
  .btn:hover {{ background: var(--blue-hover); }}
  .btn:disabled {{ background: #334155; color: #64748b; cursor: not-allowed; }}
  .btn-xai {{ background: #059669; }}
  .btn-xai:hover {{ background: #047857; }}
  .video-preview-row {{ display: flex; gap: 14px; margin-top: 16px; justify-content: center; }}
  .video-box {{ text-align: center; }}
  .video-box img {{ border: 1px solid var(--border); border-radius: 8px; width: 200px; height: 200px; object-fit: cover; background:#000; }}
  .video-lbl {{ font-size: 11px; color: var(--muted); margin-top: 6px; font-weight: 600; text-transform: uppercase; }}
  .result-header {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px; }}
  .ef-val {{ font-size: 52px; font-weight: 800; color: #fff; line-height: 1; }}
  .ef-unit {{ font-size: 22px; color: var(--muted); }}
  .clinical-badge {{ padding: 6px 16px; border-radius: 9999px; font-weight: 700; font-size: 13px; border: 1px solid; }}
  .report-box {{ margin-top: 16px; background: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 16px; font-size: 13px; line-height: 1.6; color: #cbd5e1; }}
  .alert-banner {{ background: #450a0a; border-left: 4px solid #ef4444; padding: 10px 14px; border-radius: 4px; margin-bottom: 10px; font-size: 12px; color: #fecaca; }}
  .bench {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 22px; }}
  .bench-hdr {{ font-size: 16px; font-weight: 700; color: #f1f5f9; border-bottom: 1px solid #2d3748; padding-bottom: 12px; margin-bottom: 16px; }}
  .metrics-subhdr {{ font-size: 13px; font-weight: 700; color: #38bdf8; text-transform: uppercase; margin-bottom: 12px; letter-spacing: .5px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
  .metric-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px; text-align: center; }}
  .metric-title {{ font-size: 11px; text-transform: uppercase; color: var(--muted); font-weight: 600; letter-spacing: .4px; }}
  .metric-val {{ font-size: 22px; font-weight: 800; color: #38bdf8; margin: 6px 0; }}
  .metric-sub {{ font-size: 11px; color: #64748b; }}
  .metrics-pending {{ color: var(--muted); font-style: italic; font-size: 14px; padding: 24px; text-align: center; border: 1px dashed var(--border); border-radius: 8px; }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>Explainable Spatiotemporal LVEF Estimation</h1>
      <p>CMR College of Engineering &amp; Technology · Major Project Batch 92 · R(2+1)D-18 &amp; 3D Grad-CAM</p>
    </div>
    <div class="badge">Objective 1-5 Complete</div>
  </header>

  <div class="grid">
    <!-- Panel 1: Video Input & Degradation Controls -->
    <div class="card">
      <div class="card-hdr">
        <span class="card-title">1. Apical 4-Chamber (A4C) Video Input</span>
        <span id="uploadStatus" style="font-size:12px; color:var(--ok);">Ready</span>
      </div>
      <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" accept=".avi,.npy" onchange="handleFile(event)">
        <label for="fileInput">
          <strong>Click to select an AVI echocardiogram or .npy companion file</strong><br>
          <span style="font-size:11px;color:#64748b;">(Stanford EchoNet-Dynamic format: 112x112 A4C cineloop)</span>
        </label>
      </div>

      <div class="controls-box">
        <div class="controls-title">Simulated Acoustic Degradation (Objective 1 Stress Test)</div>
        <select id="degradeSelect" class="select-degrade">
          <option value="none">Clean Cineloop (Uncorrupted Baseline)</option>
          <option value="speckle">Multiplicative Speckle Noise (eta ~ N(0, 0.25&sup2;))</option>
          <option value="shadow">Sector Acoustic Shadowing (75% Cone Attenuation)</option>
          <option value="drop">Wireless Frame Dropout (25% Temporal Drop Rate)</option>
          <option value="compound">Compound Bedside POCUS Stress (Severe Artifacts)</option>
        </select>
      </div>

      <div class="btn-group">
        <button class="btn" id="predictBtn" onclick="runAnalysis(false)" disabled>
          Run LVEF Estimation
        </button>
        <button class="btn btn-xai" id="xaiBtn" onclick="runAnalysis(true)" disabled>
          Generate 3D Grad-CAM
        </button>
      </div>

      <div class="video-preview-row" id="videoPreviewRow" style="display:none;">
        <div class="video-box" id="rawVideoBox">
          <img id="rawVideoImg" alt="Raw Video">
          <div class="video-lbl">Raw Cineloop</div>
        </div>
        <div class="video-box" id="xaiVideoBox" style="display:none;">
          <img id="xaiVideoImg" alt="Grad-CAM Saliency">
          <div class="video-lbl" style="color:#10b981;">3D Grad-CAM Attention</div>
        </div>
      </div>
    </div>

    <!-- Panel 2: Clinical Decision Support & Explainability -->
    <div class="card">
      <div class="card-hdr">
        <span class="card-title">2. Clinical Decision Support &amp; Explainability</span>
        <span id="evalStatus" style="font-size:12px; color:var(--muted);">Awaiting video input</span>
      </div>

      <div class="result-header">
        <div>
          <span class="ef-val" id="predEF">--</span>
          <span class="ef-unit">%</span>
        </div>
        <div id="clinicalBadge" class="clinical-badge" style="background:#1e293b; color:var(--muted); border-color:var(--border);">
          Pending Analysis
        </div>
      </div>

      <div id="alertsContainer"></div>

      <div class="report-box">
        <strong style="color:#93c5fd; text-transform:uppercase; font-size:11px; letter-spacing:.5px;">ASE Clinical Diagnostic Summary:</strong>
        <p id="reportNarrative" style="margin-top:8px;">Upload an apical 4-chamber echocardiogram video on the left and select 'Run LVEF Estimation' or 'Generate 3D Grad-CAM'.</p>
      </div>
    </div>
  </div>

  <!-- Panel 3: Stanford Benchmark Validation -->
  <div class="bench">
    <div class="bench-hdr">3. Benchmark Quantitative Validation — Stanford EchoNet-Dynamic (Ouyang et al.)</div>
    {metrics_html}
  </div>
</div>

<script>
  let uploadedFilename = null;

  function handleFile(event) {{
    const file = event.target.files[0];
    if (!file) return;
    uploadedFilename = file.name;
    document.getElementById('uploadStatus').innerText = 'Uploading cineloop...';

    const formData = new FormData();
    formData.append('video', file);

    fetch('/api/upload', {{ method: 'POST', body: formData }})
      .then(r => r.json())
      .then(data => {{
        document.getElementById('uploadStatus').innerText = 'Ready (' + uploadedFilename + ')';
        document.getElementById('predictBtn').disabled = false;
        document.getElementById('xaiBtn').disabled = false;
        if (data.preview_gif) {{
          document.getElementById('rawVideoImg').src = 'data:image/gif;base64,' + data.preview_gif;
          document.getElementById('videoPreviewRow').style.display = 'flex';
        }}
      }})
      .catch(() => {{
        document.getElementById('uploadStatus').innerText = 'Upload failed';
      }});
  }}

  async function runAnalysis(includeXAI) {{
    const status = document.getElementById('evalStatus');
    status.innerText = includeXAI ? 'Extracting 3D Grad-CAM…' : 'Estimating LVEF…';
    status.style.color = '#f59e0b';
    document.getElementById('predictBtn').disabled = true;
    document.getElementById('xaiBtn').disabled = true;

    const degrade = document.getElementById('degradeSelect').value;

    try {{
      const endpoint = includeXAI ? '/api/explain' : '/api/predict';
      const res = await fetch(endpoint, {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ filename: uploadedFilename, degradation: degrade }})
      }});
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      document.getElementById('predEF').innerText = data.predicted_lvef.toFixed(1);
      const badge = document.getElementById('clinicalBadge');
      badge.innerText = data.clinical_category;
      badge.style.color = data.badge_color;
      badge.style.borderColor = data.badge_color;
      badge.style.background = data.badge_color + '22';

      document.getElementById('reportNarrative').innerText = data.summary;

      // Handle alerts
      const alertsDiv = document.getElementById('alertsContainer');
      alertsDiv.innerHTML = '';
      if (data.report && data.report.clinical_alerts) {{
        data.report.clinical_alerts.forEach(alertText => {{
          const banner = document.createElement('div');
          banner.className = 'alert-banner';
          banner.innerText = '⚠️ ' + alertText;
          alertsDiv.appendChild(banner);
        }});
      }}

      // Handle XAI GIF
      if (data.xai_gif) {{
        const xaiImg = document.getElementById('xaiVideoImg');
        xaiImg.src = 'data:image/gif;base64,' + data.xai_gif;
        document.getElementById('xaiVideoBox').style.display = 'block';
        document.getElementById('videoPreviewRow').style.display = 'flex';
      }}

      status.innerText = 'Analysis Complete';
      status.style.color = '#10b981';
    }} catch (e) {{
      status.innerText = 'Error: ' + e.message;
      status.style.color = '#ef4444';
    }} finally {{
      document.getElementById('predictBtn').disabled = false;
      document.getElementById('xaiBtn').disabled = false;
    }}
  }}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Request Handler
# ---------------------------------------------------------------------------
_UPLOAD_STORE: dict = {}


class ClinicalDashboardHandler(BaseHTTPRequestHandler):

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
            self._send_json({"error": "No file received"}, 400)
            return

        raw_bytes = file_item.file.read()
        fname = os.path.basename(file_item.filename)
        _UPLOAD_STORE[fname] = raw_bytes

        # Render preview GIF of first 16 frames
        preview_gif = ""
        try:
            tmp_dir = tempfile.gettempdir()
            tmp_path = os.path.join(tmp_dir, fname)
            with open(tmp_path, "wb") as f:
                f.write(raw_bytes)

            clip = load_video_clip(tmp_path, start_frame=0, num_frames=16, stride=2, target_size=112)
            clip_unnorm = np.clip(clip * 0.18 + 0.12, 0.0, 1.0)
            pil_frames = [
                Image.fromarray((f * 255).astype(np.uint8)).resize((200, 200), Image.Resampling.BILINEAR)
                for f in clip_unnorm
            ]
            buf = io.BytesIO()
            pil_frames[0].save(
                buf, format="GIF", save_all=True, append_images=pil_frames[1:], duration=60, loop=0
            )
            preview_gif = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            pass

        self._send_json({"status": "ok", "filename": fname, "preview_gif": preview_gif})

    def _prepare_temp_video(self, fname: str, degradation: str) -> str:
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, fname)
        with open(tmp_path, "wb") as f:
            f.write(_UPLOAD_STORE[fname])

        if degradation and degradation != "none":
            # Apply degradation to .npy companion
            degrader = UltrasoundDegradation(seed=42)
            clip = load_video_clip(tmp_path, start_frame=0, num_frames=64, stride=1, target_size=112)
            clip_raw = np.clip(clip * 0.18 + 0.12, 0.0, 1.0)

            if degradation == "speckle":
                degraded = degrader.add_speckle_noise(clip_raw, severity=0.25)
            elif degradation == "shadow":
                degraded = degrader.apply_acoustic_shadowing(clip_raw, attenuation_factor=0.75)
            elif degradation == "drop":
                degraded = degrader.apply_frame_dropout(clip_raw, drop_rate=0.25, mode="freeze")
            elif degradation == "compound":
                degraded = degrader.apply_compound_degradation(clip_raw, severity="moderate")
            else:
                degraded = clip_raw

            deg_npy_path = os.path.splitext(tmp_path)[0] + "_deg.npy"
            np.save(deg_npy_path, (degraded * 255).astype(np.uint8))
            return deg_npy_path

        return tmp_path

    def _handle_predict(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        fname = body.get("filename", "")
        degrade = body.get("degradation", "none")

        if fname not in _UPLOAD_STORE:
            self._send_json({"error": f"File '{fname}' not found. Upload first."}, 400)
            return

        try:
            target_path = self._prepare_temp_video(fname, degrade)
            result = estimator.predict(target_path)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def _handle_explain(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        fname = body.get("filename", "")
        degrade = body.get("degradation", "none")

        if fname not in _UPLOAD_STORE:
            self._send_json({"error": f"File '{fname}' not found. Upload first."}, 400)
            return

        try:
            target_path = self._prepare_temp_video(fname, degrade)
            raw_frames, heatmap_3d, clip_ef = estimator.explain(target_path, start_frame=0)
            gif_b64 = _generate_overlay_gif(raw_frames, heatmap_3d, alpha=0.5, duration_ms=60)

            # Generate full report
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
# Server Entry Point
# ---------------------------------------------------------------------------
estimator = None


def run_server(port: int = 8501, checkpoint_arg: str = "") -> None:
    global estimator
    checkpoint_path = _find_checkpoint(checkpoint_arg)
    print(f"[*] Initializing LVEFEstimator from: {checkpoint_path}")
    estimator = LVEFEstimator(checkpoint_path=checkpoint_path)

    server = HTTPServer(("127.0.0.1", port), ClinicalDashboardHandler)
    print("=" * 70)
    print("  ECHONET-DYNAMIC CLINICAL AI & EXPLAINABILITY DASHBOARD")
    print(f"  -> Local Web UI:    http://127.0.0.1:{port}")
    print(f"  -> Checkpoint Path: {checkpoint_path}")
    print("=" * 70)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[app] Shutting down dashboard server.")
        server.server_close()


if __name__ == "__main__":
    port_arg = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8501
    ckpt_arg = sys.argv[2] if len(sys.argv) > 2 else ""
    run_server(port=port_arg, checkpoint_arg=ckpt_arg)
