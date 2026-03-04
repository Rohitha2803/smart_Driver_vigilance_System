import urllib.request
import os
import sys

MODEL_DIR = os.path.join(os.path.dirname(__file__), "backend", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

URLS = {
    "yolov3-tiny.weights": [
        "https://pjreddie.com/media/files/yolov3-tiny.weights",
        "https://github.com/patrick013/Object-Detection-using-YOLO/raw/master/yolov3-tiny.weights",
        "https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/raw/master/yolov3-tiny.weights"
    ],
    "yolov3-tiny.cfg": [
        "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
        "https://github.com/pjreddie/darknet/raw/master/cfg/yolov3-tiny.cfg"
    ],
    "coco.names": [
        "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
        "https://github.com/pjreddie/darknet/raw/master/data/coco.names"
    ]
}

def download_file(url_list, filename):
    filepath = os.path.join(MODEL_DIR, filename)
    
    # Check if exists and has size > 0
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        print(f"[SKIP] {filename} already exists ({os.path.getsize(filepath)} bytes).")
        return

    print(f"Downloading {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    for url in url_list:
        try:
            print(f"  Trying {url}...")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                with open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
            
            size = os.path.getsize(filepath)
            print(f"  [SUCCESS] {filename} downloaded ({size} bytes).")
            return
        except Exception as e:
            print(f"  [FAIL] Error downloading from {url}: {e}")
    
    print(f"[ERROR] Failed to download {filename} from all mirrors.")

if __name__ == "__main__":
    for filename, urls in URLS.items():
        if isinstance(urls, str):
            urls = [urls]
        download_file(urls, filename)
