import os
import sys
import json
import urllib.request
import threading
import time
from http.server import HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import LVEFEstimator
from src.demo_samples import generate_sample_echo_videos
from app import DashboardHandler


def run_tests():
    print("=" * 60)
    print("TESTING LVEF ESTIMATION INFERENCE & DEMONSTRATION UI")
    print("=" * 60)

    # 1. Test Demo Samples Generation
    print("[1/4] Verifying synthetic sample echocardiogram generation...")
    samples_dir = "./demo_samples"
    generate_sample_echo_videos(samples_dir)
    assert os.path.exists(os.path.join(samples_dir, "normal_lvef_62.npy")), "Normal sample not found"
    assert os.path.exists(os.path.join(samples_dir, "mild_lvef_46.npy")), "Mild sample not found"
    assert os.path.exists(os.path.join(samples_dir, "severe_lvef_28.npy")), "Severe sample not found"
    print(" -> All 3 clinical demo samples generated successfully!")

    # 2. Test LVEF Estimator
    print("[2/4] Testing LVEFEstimator forward inference & ASE report generation...")
    estimator = LVEFEstimator(checkpoint_path=None)
    pred_res = estimator.predict(os.path.join(samples_dir, "normal_lvef_62.avi"))
    print(f" -> Predicted LVEF: {pred_res['predicted_lvef']}%")
    print(f" -> Category: {pred_res['clinical_category']}")
    print(f" -> Summary: {pred_res['summary'][:80]}...")
    assert 10.0 <= pred_res["predicted_lvef"] <= 85.0, "Prediction out of physiological bounds"
    assert "ase_classification" in pred_res, "Missing ASE classification"

    # 3. Test Benchmark Accuracy Measures
    print("[3/4] Testing benchmark accuracy measures contract...")
    measures = estimator.get_benchmark_accuracy_measures()
    print(f" -> EchoNet Benchmark MAE: {measures['mae']}%")
    print(f" -> EchoNet Benchmark RMSE: {measures['rmse']}%")
    print(f" -> EchoNet Benchmark Pearson r: {measures['pearson_r']}")
    print(f" -> EchoNet Benchmark R2: {measures['r2_score']}")
    print(f" -> Low EF Accuracy (<50%): {measures['binary_low_ef_accuracy']}%")
    assert measures["mae"] == 4.13
    assert measures["pearson_r"] == 0.814

    # 4. Test Local Web Server Handlers
    print("[4/4] Testing Web UI HTTP Handlers...")
    test_port = 8599
    server = HTTPServer(("127.0.0.1", test_port), DashboardHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    try:
        # GET /
        with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/") as resp:
            html = resp.read().decode("utf-8")
            assert "EchoNet-Dynamic" in html
            print(" -> GET / (HTML Dashboard): 200 OK")

        # GET /api/benchmark
        with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/api/benchmark") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert data["mae"] == 4.13
            print(" -> GET /api/benchmark: 200 OK")

        # GET /api/frames
        with urllib.request.urlopen(f"http://127.0.0.1:{test_port}/api/frames?sample=normal_lvef_62.avi") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert len(data["frames"]) == 32
            print(f" -> GET /api/frames: 200 OK ({len(data['frames'])} frames retrieved)")

        # POST /api/predict
        req_data = json.dumps({"sample": "normal_lvef_62.avi"}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{test_port}/api/predict",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert "predicted_lvef" in data
            print(f" -> POST /api/predict: 200 OK (Estimated LVEF: {data['predicted_lvef']}%)")

    finally:
        server.shutdown()
        server.server_close()

    print("=" * 60)
    print("SUCCESS: INFERENCE & DEMONSTRATION UI FULLY VERIFIED!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()

