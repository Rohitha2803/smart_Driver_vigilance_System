"""
Download required model files for the Driver Safety Monitoring System.
Run this script after cloning the repository:
    python backend/setup_models.py
"""

import os
import sys
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "models")

MODELS = [
    {
        "name": "YOLOv3-Tiny Weights",
        "filename": "yolov3-tiny.weights",
        "url": "https://pjreddie.com/media/files/yolov3-tiny.weights",
        "size_mb": 34,
    },
    {
        "name": "EfficientDet-Lite0 (Phone Detection)",
        "filename": "efficientdet_lite0.tflite",
        "url": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite",
        "size_mb": 4.4,
    },
]


def download_file(url, dest_path, label):
    """Download a file with progress indicator."""
    print(f"  Downloading {label}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536
            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = downloaded / total * 100
                        print(f"\r  [{pct:5.1f}%] {downloaded // 1024} KB", end="", flush=True)
            print()
        size = os.path.getsize(dest_path)
        print(f"  ✓ Saved: {dest_path} ({size / 1024 / 1024:.1f} MB)")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    print("=" * 55)
    print("  Driver Safety Monitor — Model Setup")
    print("=" * 55)
    print()

    os.makedirs(MODELS_DIR, exist_ok=True)

    success = 0
    for model in MODELS:
        dest = os.path.join(MODELS_DIR, model["filename"])
        if os.path.exists(dest):
            print(f"  ✓ {model['name']} already exists, skipping.")
            success += 1
            continue
        print(f"  [{model['size_mb']} MB] {model['name']}")
        if download_file(model["url"], dest, model["filename"]):
            success += 1

    print()
    if success == len(MODELS):
        print("All models downloaded successfully! ✓")
        print("You can now start the server with:")
        print("  cd backend")
        print("  python -m uvicorn main:app --host 0.0.0.0 --port 8000")
    else:
        print(f"WARNING: {len(MODELS) - success} model(s) failed to download.")
        print("Please check your internet connection and try again.")

    return 0 if success == len(MODELS) else 1


if __name__ == "__main__":
    sys.exit(main())
