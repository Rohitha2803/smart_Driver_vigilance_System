# Smart Driver Vigilance System

A real-time driver safety monitoring system that detects **drowsiness** (eye closure, blink rate, yawning), **mobile phone usage**, and **distracted head turns**, with a premium web-based dashboard.

## Architecture

```
Backend (Python)                    Frontend (Web)
┌──────────────────┐               ┌──────────────────┐
│ FastAPI Server   │──WebSocket──▶│ HTML/CSS/JS UI   │
│ OpenCV Capture   │  (frames +   │ Canvas Rendering │
│ MediaPipe Mesh   │   JSON data) │ Audio Alerts     │
│ EfficientDet     │               │ Status Dashboard │
└──────────────────┘               └──────────────────┘
```

## Features

- **Drowsiness Detection**: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), blink rate
- **Phone Detection**: EfficientDet-Lite0 object detection (MediaPipe Tasks API)
- **Head Turn Detection**: Sideways head movement alarm using face landmark geometry
- **Real-time Streaming**: WebSocket-based annotated video frames at ~15 FPS
- **Premium UI**: Dark-mode glassmorphism dashboard with animated status indicators
- **Audio Alerts**: Web Audio API alarm tones on detection events
- **Event Logging**: Timestamped log of all detection events

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Download AI Models (Required!)

```bash
python setup_models.py
```

This downloads:
- **YOLOv3-Tiny** (~34 MB) — general object detection
- **EfficientDet-Lite0** (~4.4 MB) — phone/remote detection

> ⚠️ **This step is required!** The models are too large for GitHub and must be downloaded separately.

### 4. Run the Server

```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 5. Open the Dashboard

Navigate to **http://localhost:8000** in your browser and click **Start Monitoring**.

## Detection Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| EAR | < 0.22 | Eyes closed |
| MAR | > 0.65 | Yawning |
| Consecutive closed frames | ≥ 20 | Drowsy alert |
| Phone confidence | ≥ 0.35 | Phone detected |
| Head turn ratio | > 0.34 | Looking sideways |
| Head turn frames | ≥ 15 | Distracted alert |

## Tech Stack

- **Backend**: Python, FastAPI, OpenCV, MediaPipe, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **AI Models**: EfficientDet-Lite0 (phone), MediaPipe Face Mesh (drowsiness/head pose)
- **Communication**: WebSocket (real-time bidirectional)
