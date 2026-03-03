"""
Driver Safety Monitoring System — Backend Server
FastAPI application with WebSocket streaming for real-time detection.
"""

import asyncio
import base64
import json
import time
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from detection.drowsiness import DrowsinessDetector
from detection.phone import PhoneDetector
from utils.alarm import AlarmManager

# ─── App Setup ────────────────────────────────────────────────
app = FastAPI(title="Driver Safety Monitoring System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/css", StaticFiles(directory=os.path.join(FRONTEND_DIR, "css")), name="css")
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")
app.mount(
    "/assets",
    StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")),
    name="assets",
)


@app.get("/")
async def serve_frontend():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Driver Safety Monitoring System is running"}


# ─── WebSocket Video Streaming ────────────────────────────────
class CameraManager:
    """Manages camera capture and detection processing."""

    def __init__(self):
        self.cap = None
        self.drowsiness_detector = None
        self.phone_detector = None
        self.alarm_manager = AlarmManager()
        self.is_running = False

    def start(self, camera_index: int = 0):
        """Start camera capture and initialize detectors."""
        if self.is_running:
            return True

        print(f"[CameraManager] Attempting to open camera index {camera_index}...")
        # Add cv2.CAP_DSHOW on Windows for faster initialization
        if os.name == 'nt':
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(camera_index)
            
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_index}")
            return False

        # Set camera properties for performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.drowsiness_detector = DrowsinessDetector()
        self.phone_detector = PhoneDetector()
        self.is_running = True
        print("[CameraManager] Camera detection started successfully")
        return True

    def stop(self):
        """Stop camera and release resources."""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.drowsiness_detector:
            self.drowsiness_detector.release()
            self.drowsiness_detector = None
        if self.phone_detector:
            self.phone_detector.release()
            self.phone_detector = None
        print("[CameraManager] Camera stopped")

    def get_frame(self):
        """Capture a frame and run detection."""
        if not self.is_running or not self.cap:
            return None, None

        ret, frame = self.cap.read()
        if not ret:
            print("[CameraManager] Failed to read frame")
            return None, None

        # Flip horizontally for mirror effect
        try:
            frame = cv2.flip(frame, 1)
        except Exception as e:
            print(f"[CameraManager] Error flipping frame: {e}")
            return None, None

        # Run detections
        drowsy_status = self.drowsiness_detector.process_frame(frame)
        phone_status = self.phone_detector.process_frame(frame)

        # Annotate frame
        annotated = self._annotate_frame(frame, drowsy_status, phone_status)

        # Combine statuses
        status = {
            "timestamp": time.time(),
            "drowsiness": drowsy_status,
            "phone": phone_status,
        }

        # Trigger server-side alarm if needed
        if drowsy_status["drowsy"] or phone_status.get("alert", False):
            self.alarm_manager.play_alarm()

        return annotated, status

    def _annotate_frame(self, frame, drowsy_status, phone_status):
        """Draw detection overlays on the frame."""
        h, w, _ = frame.shape

        # Draw eye landmarks
        if drowsy_status["face_detected"]:
            # Left eye
            for pt in drowsy_status["left_eye_points"]:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)
            # Right eye
            for pt in drowsy_status["right_eye_points"]:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

            # EAR and MAR text
            ear_color = (0, 0, 255) if drowsy_status["drowsy"] else (0, 255, 0)
            cv2.putText(
                frame,
                f"EAR: {drowsy_status['ear_avg']:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                ear_color,
                2,
            )

            mar_color = (0, 0, 255) if drowsy_status["yawning"] else (0, 255, 0)
            cv2.putText(
                frame,
                f"MAR: {drowsy_status['mar']:.3f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                mar_color,
                2,
            )

            # Drowsy alert overlay
            if drowsy_status["drowsy"]:
                cv2.putText(
                    frame,
                    "!! DROWSY !!",
                    (w // 2 - 100, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                )
                # Red border
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

            # Yawning indicator
            if drowsy_status["yawning"]:
                cv2.putText(
                    frame,
                    "YAWNING",
                    (w // 2 - 70, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 165, 255),
                    2,
                )

        # Phone detection overlay
        if phone_status["phone_detected"]:
            bbox = phone_status.get("bbox")
            if bbox:
                x, y, bw, bh = bbox
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"PHONE {phone_status['confidence']:.0%}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )

        if phone_status.get("alert", False):
            cv2.putText(
                frame,
                "!! PHONE DETECTED !!",
                (w // 2 - 140, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 255),
                3,
            )
            cv2.rectangle(frame, (2, 2), (w - 3, h - 3), (255, 0, 255), 4)

        # Blink count
        cv2.putText(
            frame,
            f"Blinks: {drowsy_status['blink_count']}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return frame


# Global camera manager
camera = CameraManager()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.
    Uses concurrent tasks for receiving commands and sending frames.
    """
    await ws.accept()
    print("[WebSocket] Client connected")

    # Event loop for thread pool executor
    loop = asyncio.get_event_loop()

    async def receive_commands():
        """Task to receive commands from client."""
        try:
            while True:
                data = await ws.receive_text()
                command = json.loads(data)
                print(f"[WebSocket] Received command: {command}")
                
                if command.get("action") == "start":
                    camera_index = command.get("camera_index", 0)
                    # Run potentially blocking camera start in a separate thread
                    success = await loop.run_in_executor(None, camera.start, camera_index)
                    
                    response = {
                        "type": "control",
                        "status": "started" if success else "error",
                        "message": "Camera started" if success else f"Failed to open camera {camera_index}"
                    }
                    await ws.send_text(json.dumps(response))
                    
                elif command.get("action") == "stop":
                    camera.stop()
                    await ws.send_text(json.dumps({
                        "type": "control",
                        "status": "stopped",
                        "message": "Camera stopped"
                    }))
        except WebSocketDisconnect:
            print("[WebSocket] Client disconnected (receive task)")
        except Exception as e:
            print(f"[WebSocket] Error in receive task: {e}")

    async def send_frames():
        """Task to send frames to client."""
        try:
            while True:
                if camera.is_running:
                    start_time = time.time()
                    
                    # Run blocking get_frame in thread pool
                    frame, status = await loop.run_in_executor(None, camera.get_frame)
                    
                    if frame is not None:
                        # Encode frame (lower quality = faster transfer)
                        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
                        frame_b64 = base64.b64encode(buffer).decode("utf-8")
                        
                        # Clean status for JSON serialization
                        clean_status = {
                            "timestamp": status["timestamp"],
                            "drowsiness": {
                                k: v for k, v in status["drowsiness"].items() 
                                if k not in ("left_eye_points", "right_eye_points")
                            },
                            "phone": status["phone"]
                        }
                        
                        message = {
                            "type": "frame",
                            "frame": frame_b64,
                            "status": clean_status
                        }
                        
                        await ws.send_text(json.dumps(message))
                    
                    # Target ~15 FPS (67ms per frame) for smoother experience
                    processing_time = time.time() - start_time
                    delay = max(0.001, 0.067 - processing_time)
                    await asyncio.sleep(delay)
                else:
                    # Idle wait if camera not running
                    await asyncio.sleep(0.1)
                    
        except WebSocketDisconnect:
            print("[WebSocket] Client disconnected (send task)")
        except Exception as e:
            print(f"[WebSocket] Error in send task: {e}")
            # Try to report error to client if possible
            try:
                await ws.send_text(json.dumps({
                    "type": "control", 
                    "status": "error", 
                    "message": f"Server error: {str(e)}"
                }))
            except:
                pass

    # Run tasks concurrently
    try:
        # Create tasks
        receive_task = asyncio.create_task(receive_commands())
        send_task = asyncio.create_task(send_frames())
        
        # Wait for either to finish (likely receive_task detects disconnect first)
        done, pending = await asyncio.wait(
            [receive_task, send_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks (stop sending frames if client disconnected)
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
    except Exception as e:
        print(f"[WebSocket] Connection terminated: {e}")
    finally:
        camera.stop()
        print("[WebSocket] Cleaned up resources")


@app.on_event("shutdown")
async def shutdown():
    """Clean up on server shutdown."""
    camera.stop()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
