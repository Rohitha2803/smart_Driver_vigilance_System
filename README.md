# Driver Safety Monitoring System

A real-time driver safety monitoring system that detects **drowsiness** (eye closure, blink rate, yawning) and **mobile phone usage**, with a premium web-based dashboard.

## Architecture

```
Backend (Python)                    Frontend (Web)
┌──────────────────┐               ┌──────────────────┐
│ FastAPI Server   │──WebSocket──▶│ HTML/CSS/JS UI   │
│ OpenCV Capture   │  (frames +   │ Canvas Rendering │
│ MediaPipe Mesh   │   JSON data) │ Audio Alerts     │
│ Phone Detection  │               │ Status Dashboard │
└──────────────────┘               └──────────────────┘
```

## Features

- **Drowsiness Detection**: Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), blink rate
- **Phone Detection**: Hand-near-ear proximity via MediaPipe Hands + Face Detection
- **Real-time Streaming**: WebSocket-based annotated video frames at ~20 FPS
- **Premium UI**: Dark-mode glassmorphism dashboard with animated status indicators
- **Audio Alerts**: Web Audio API alarm tones on detection events
- **Event Logging**: Timestamped log of all detection events

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Run the Server

```bash
cd backend
python main.py
```

Or with uvicorn directly:

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open the Dashboard

Navigate to **http://localhost:8000** in your browser.

### 4. Start Monitoring

Click the **Start Monitoring** button to begin real-time detection.

## Detection Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| EAR    | < 0.22    | Eyes closed |
| MAR    | > 0.65    | Yawning |
| Consecutive closed frames | ≥ 20 | Drowsy alert |
| Phone confidence | ≥ 0.50 | Phone detected |

## Tech Stack

- **Backend**: Python, FastAPI, OpenCV, MediaPipe, NumPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Communication**: WebSocket (real-time bidirectional)
