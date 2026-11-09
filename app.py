"""
EchoNet-Dynamic LVEF Estimation — Minimal Inference Dashboard
=============================================================
A lightweight Python-stdlib HTTP server for demonstrating the trained model.

Requirements:
  • A valid trained checkpoint at outputs/best_r2plus1d_lvef.pt  (or pass
    the path as the first CLI argument).
  • A video file (AVI) or companion .npy to run inference on.

Metrics displayed in the UI are loaded exclusively from
outputs/test_metrics.json — only populated after genuine training &
evaluation on the Stanford EchoNet-Dynamic test set.  No hardcoded numbers.

Usage:
    python app.py [port]            # default port 8501
"""
import base64
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from src.inference import LVEFEstimator

# ---------------------------------------------------------------------------
# Checkpoint resolution — raises FileNotFoundError if no trained model exists
# ---------------------------------------------------------------------------
def _find_checkpoint() -> str:
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
        "Train the model first:\n"
        "  python kaggle/train_kaggle.py --data_dir <path/to/EchoNet-Dynamic>"
    )


CHECKPOINT_PATH = _find_checkpoint()
estimator = LVEFEstimator(checkpoint_path=CHECKPOINT_PATH)
print(f"[app] Loaded checkpoint: {CHECKPOINT_PATH}")

# ---------------------------------------------------------------------------
# Load real test metrics if available
# ---------------------------------------------------------------------------
def _load_test_metrics() -> dict:
    path = os.path.join(REPO_ROOT, "outputs", "test_metrics.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
_METRICS_PENDING_MSG = "Model pending training on Stanford EchoNet-Dynamic"


def _build_metrics_html(metrics: dict) -> str:
    if not metrics:
        return (
            f'<div class="metrics-pending">{_METRICS_PENDING_MSG}</div>'
        )
    return f"""
<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-title">Mean Absolute Error</div>
    <div class="metric-val">{metrics.get('mae', 0):.2f}%</div>
    <div class="metric-sub">Target: &le; 4.5%</div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Root Mean Squared Error</div>
    <div class="metric-val">{metrics.get('rmse', 0):.2f}%</div>
    <div class="metric-sub">Penalises outliers</div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Pearson Correlation (r)</div>
    <div class="metric-val">{metrics.get('pearson_r', 0):.3f}</div>
    <div class="metric-sub">p = {metrics.get('pearson_p', 1):.2e}</div>
  </div>
  <div class="metric-card">
    <div class="metric-title">R&sup2; Score</div>
    <div class="metric-val">{metrics.get('r2', 0):.3f}</div>
    <div class="metric-sub">Explained variance</div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Bland-Altman Bias</div>
    <div class="metric-val">{metrics.get('bland_altman_bias', 0):.2f}%</div>
    <div class="metric-sub">
      LoA: [{metrics.get('bland_altman_lower_loa', 0):.2f}%,
            {metrics.get('bland_altman_upper_loa', 0):.2f}%]
    </div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Low EF (&lt;50%) Accuracy</div>
    <div class="metric-val">{metrics.get('binary_low_ef_accuracy', 0)*100:.1f}%</div>
    <div class="metric-sub">
      Sens: {metrics.get('binary_low_ef_sensitivity', 0)*100:.1f}% &nbsp;|&nbsp;
      Spec: {metrics.get('binary_low_ef_specificity', 0)*100:.1f}%
    </div>
  </div>
</div>
<p class="metrics-note">N = {metrics.get('sample_count', 0):,} blind test videos
(Stanford EchoNet-Dynamic official test split)</p>
"""


def _build_html() -> str:
    metrics = _load_test_metrics()
    metrics_html = _build_metrics_html(metrics)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EchoNet-Dynamic | LVEF Estimation Baseline</title>
<style>
  :root {{
    --bg: #0a0e17; --bg2: #111827; --card: #1f2937;
    --border: #374151; --blue: #3b82f6; --cyan: #06b6d4;
    --text: #f9fafb; --muted: #9ca3af;
    --ok: #10b981; --warn: #f59e0b; --danger: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0;
       font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; }}
  body {{ background: var(--bg); color: var(--text); min-height: 100vh; padding: 24px; }}
  .container {{ max-width: 1100px; margin: 0 auto; }}
  header {{ display: flex; justify-content: space-between; align-items: center;
            border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 26px; }}
  header h1 {{ font-size: 22px; font-weight: 700;
               background: linear-gradient(90deg,#60a5fa,#a78bfa);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  header p {{ font-size: 12px; color: var(--muted); margin-top: 3px; }}
  .badge {{ background: #1e1b4b; color: #a5b4fc; border: 1px solid #3730a3;
            padding: 5px 12px; border-radius: 9999px; font-size: 12px; font-weight: 600; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  @media (max-width: 800px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }}
  .card-hdr {{ display: flex; justify-content: space-between; align-items: center;
               border-bottom: 1px solid #2d3748; padding-bottom: 10px; margin-bottom: 14px; }}
  .card-title {{ font-size: 14px; font-weight: 600; color: #e2e8f0; }}
  .upload-zone {{ border: 2px dashed var(--border); border-radius: 8px; padding: 24px;
                  text-align: center; cursor: pointer; transition: border-color .2s; }}
  .upload-zone:hover {{ border-color: var(--blue); }}
  .upload-zone input {{ display: none; }}
  .upload-zone label {{ cursor: pointer; color: var(--muted); font-size: 14px; }}
  .btn {{ background: var(--blue); color: #fff; border: none; padding: 10px 22px;
          border-radius: 6px; font-weight: 600; cursor: pointer;
          width: 100%; margin-top: 14px; transition: background .2s; font-size: 14px; }}
  .btn:hover {{ background: #1d4ed8; }}
  .btn:disabled {{ background: #374151; cursor: not-allowed; }}
  .result-box {{ display: flex; flex-direction: column; align-items: center;
                 justify-content: center; min-height: 220px; padding: 16px; }}
  .ef-value {{ font-size: 56px; font-weight: 800; color: #fff; line-height: 1; }}
  .ef-unit {{ font-size: 20px; color: var(--muted); }}
  .ef-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
               color: var(--muted); margin-top: 4px; }}
  .clinical-badge {{ margin-top: 14px; padding: 5px 16px; border-radius: 9999px;
                     font-weight: 700; font-size: 13px; border: 1px solid; }}
  .ase-box {{ margin-top: 16px; background: #1a2234; border: 1px solid #2e3d5b;
              border-radius: 8px; padding: 12px; width: 100%; }}
  .ase-title {{ font-size: 11px; font-weight: 700; color: #93c5fd;
                text-transform: uppercase; margin-bottom: 5px; letter-spacing: .5px; }}
  .ase-text {{ font-size: 13px; line-height: 1.5; color: #cbd5e1; }}
  .bench {{ background: var(--bg2); border: 1px solid var(--border);
            border-radius: 10px; padding: 18px; }}
  .bench-hdr {{ font-size: 14px; font-weight: 600; color: #e2e8f0;
                border-bottom: 1px solid #2d3748; padding-bottom: 10px; margin-bottom: 14px; }}
  .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr)); gap: 12px; }}
  .metric-card {{ background: #1e293b; border: 1px solid #334155;
                  border-radius: 8px; padding: 12px; text-align: center; }}
  .metric-title {{ font-size: 11px; text-transform: uppercase; color: var(--muted);
                   font-weight: 600; letter-spacing: .4px; }}
  .metric-val {{ font-size: 20px; font-weight: 700; color: #38bdf8; margin: 4px 0; }}
  .metric-sub {{ font-size: 11px; color: #64748b; }}
  .metrics-pending {{ color: var(--muted); font-style: italic; font-size: 14px;
                      padding: 20px; text-align: center; border: 1px dashed var(--border);
                      border-radius: 8px; }}
  .metrics-note {{ font-size: 11px; color: #64748b; margin-top: 10px; text-align: right; }}
  .status {{ font-size: 12px; color: var(--ok); }}
  #preview {{ max-width: 100%; border-radius: 6px; margin-top: 10px;
              display: none; border: 1px solid var(--border); }}
</style>
</head>
<body>
<div class="container">
  <header>
    <div>
      <h1>Spatiotemporal Baseline: Automated LVEF Estimation</h1>
      <p>CMR College of Engineering &amp; Technology · Batch 92 · R(2+1)D-18 on Stanford EchoNet-Dynamic</p>
    </div>
    <div class="badge">Major Project Baseline</div>
  </header>

  <div class="grid">
    <!-- Input panel -->
    <div class="card">
      <div class="card-hdr">
        <span class="card-title">1. Upload A4C Echocardiography Video</span>
        <span class="status" id="uploadStatus">Ready</span>
      </div>
      <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" accept=".avi,.npy" onchange="handleFile(event)">
        <label for="fileInput">
          Click to select an AVI or .npy echocardiogram file<br>
          <span style="font-size:11px;color:#4b5563">(EchoNet-Dynamic compatible format)</span>
        </label>
      </div>
      <img id="preview" alt="First frame preview">
      <button class="btn" id="predictBtn" onclick="runPrediction()" disabled>
        Run LVEF Deep Learning Estimation
      </button>
    </div>

    <!-- Result panel -->
    <div class="card">
      <div class="card-hdr">
        <span class="card-title">2. Clinical Prediction &amp; Decision Support</span>
        <span class="status" id="evalStatus">Awaiting upload</span>
      </div>
      <div class="result-box">
        <div>
          <span class="ef-value" id="predictedEF">--</span>
          <span class="ef-unit">%</span>
        </div>
        <span class="ef-label">Estimated LVEF</span>
        <div id="clinicalBadge" class="clinical-badge"
             style="background: #1f293700; color: var(--muted); border-color: var(--border);">
          Run inference to see classification
        </div>
        <div class="ase-box">
          <div class="ase-title">ASE Clinical Summary</div>
          <div id="aseSummaryText" class="ase-text">
            Upload an echocardiogram and click the button above.
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Benchmark metrics — loaded from outputs/test_metrics.json -->
  <div class="bench">
    <div class="bench-hdr">3. Baseline Accuracy — Stanford EchoNet-Dynamic Test Set</div>
    {metrics_html}
  </div>
</div>

<script>
  let uploadedFilename = null;

  function handleFile(event) {{
    const file = event.target.files[0];
    if (!file) return;
    uploadedFilename = file.name;
    document.getElementById('uploadStatus').innerText = 'Uploading…';

    const formData = new FormData();
    formData.append('video', file);

    fetch('/api/upload', {{ method: 'POST', body: formData }})
      .then(r => r.json())
      .then(data => {{
        document.getElementById('uploadStatus').innerText = 'Ready';
        document.getElementById('predictBtn').disabled = false;
        if (data.preview_b64) {{
          const img = document.getElementById('preview');
          img.src = 'data:image/jpeg;base64,' + data.preview_b64;
          img.style.display = 'block';
        }}
      }})
      .catch(() => {{
        document.getElementById('uploadStatus').innerText = 'Upload error';
      }});
  }}

  async function runPrediction() {{
    document.getElementById('evalStatus').innerText = 'Estimating…';
    document.getElementById('evalStatus').style.color = '#f59e0b';
    document.getElementById('predictBtn').disabled = true;
    try {{
      const res = await fetch('/api/predict', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{filename: uploadedFilename}})
      }});
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      document.getElementById('predictedEF').innerText = data.predicted_lvef.toFixed(1);
      const badge = document.getElementById('clinicalBadge');
      badge.innerText = data.clinical_category;
      badge.style.color = data.badge_color;
      badge.style.borderColor = data.badge_color;
      badge.style.background = data.badge_color + '22';
      document.getElementById('aseSummaryText').innerText = data.summary;
      document.getElementById('evalStatus').innerText = 'Complete';
      document.getElementById('evalStatus').style.color = '#10b981';
    }} catch(e) {{
      document.getElementById('evalStatus').innerText = 'Error: ' + e.message;
      document.getElementById('evalStatus').style.color = '#ef4444';
    }} finally {{
      document.getElementById('predictBtn').disabled = false;
    }}
  }}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------
# Temporary upload store (single-session, in-memory)
_UPLOAD_STORE: dict = {}


class DashboardHandler(BaseHTTPRequestHandler):

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- GET ----------------------------------------------------------------
    def do_GET(self):
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

    # -- POST ---------------------------------------------------------------
    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/upload":
            self._handle_upload()
        elif parsed.path == "/api/predict":
            self._handle_predict()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_upload(self):
        """Accept a multipart/form-data upload, store bytes in memory."""
        import cgi
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self._send_json({"error": "Expected multipart/form-data"}, 400)
            return

        fs = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )
        file_item = fs["video"]
        if not file_item.filename:
            self._send_json({"error": "No file received"}, 400)
            return

        raw_bytes = file_item.file.read()
        fname = os.path.basename(file_item.filename)
        _UPLOAD_STORE[fname] = raw_bytes

        # Generate a small preview (first frame)
        preview_b64 = ""
        try:
            tmp_path = os.path.join("/tmp", fname)
            with open(tmp_path, "wb") as f:
                f.write(raw_bytes)
            # Try loading first frame for preview
            from src.data.io import load_video_clip
            clip = load_video_clip(tmp_path, start_frame=0, num_frames=1, stride=1,
                                   target_size=112, mean=0.0, std=1.0)
            # clip is normalised; undo partial normalisation to get [0,1], then *255
            frame = np.clip(clip[0], 0.0, 1.0)
            img = Image.fromarray((frame * 255).astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            preview_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            pass  # preview is optional; inference still works without it

        self._send_json({"status": "ok", "filename": fname, "preview_b64": preview_b64})

    def _handle_predict(self):
        """Run inference on a previously uploaded file."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        fname = body.get("filename", "")

        if fname not in _UPLOAD_STORE:
            self._send_json({"error": f"File '{fname}' not found. Upload first."}, 400)
            return

        # Write to a temp file for inference
        tmp_path = os.path.join("/tmp", fname)
        with open(tmp_path, "wb") as f:
            f.write(_UPLOAD_STORE[fname])

        try:
            result = estimator.predict(tmp_path)
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def log_message(self, fmt, *args):
        pass  # Suppress verbose request log spam


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_server(port: int = 8501) -> None:
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    print("=" * 60)
    print("ECHONET-DYNAMIC LVEF ESTIMATION BASELINE — DASHBOARD")
    print(f"  → Local UI:  http://127.0.0.1:{port}")
    print(f"  → Checkpoint: {CHECKPOINT_PATH}")
    print("=" * 60)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[app] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    run_server(port=port)
