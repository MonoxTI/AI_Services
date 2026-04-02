"""
camera_client.py — Raspberry Pi camera process.

Responsibilities:
  - Capture frames from camera
  - Apply low-light enhancement if needed
  - Stream MJPEG via Flask (dashboard and server consume this)
  - NO face recognition — that runs on the server

Usage:
    python -m pi.core.camera_client
    python -m pi.core.camera_client --dry-run   # test without camera
"""

import argparse
import logging
import signal
import sys
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response

from pi.ipc.config import (
    FRAME_WIDTH, FRAME_HEIGHT, JPEG_QUALITY,
    STREAM_PORT, CAMERA_INDEX,
)

log = logging.getLogger("camera_client")
app = Flask(__name__)

# ── Shared frame state ────────────────────────────────────────────────────────

_frame: np.ndarray | None = None
_frame_lock = threading.Lock()
_running = False


def _set_frame(frame: np.ndarray):
    global _frame
    with _frame_lock:
        _frame = frame.copy()


def _get_frame() -> np.ndarray | None:
    with _frame_lock:
        return _frame.copy() if _frame is not None else None


# ── Low-light enhancement ─────────────────────────────────────────────────────

def _enhance(frame: np.ndarray, threshold: int = 60) -> np.ndarray:
    if np.mean(frame) < threshold:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eq = cv2.equalizeHist(gray)
        frame = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    return frame


# ── Capture loop (background thread) ─────────────────────────────────────────

def _capture_loop(cap: cv2.VideoCapture):
    global _running
    log.info("Capture loop started")
    while _running:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame read failed, retrying...")
            time.sleep(0.1)
            continue
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame = _enhance(frame)
        _set_frame(frame)
        time.sleep(0.01)  # cap at ~100 fps


def _dummy_capture_loop():
    """Generates a test pattern when --dry-run is set."""
    global _running
    log.info("Dry-run capture loop started")
    t = 0
    while _running:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        frame[:, :, 1] = int(128 + 100 * np.sin(t))  # pulsing green
        cv2.putText(frame, "DRY RUN", (220, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 200, 200), 2)
        _set_frame(frame)
        t += 0.05
        time.sleep(0.03)


# ── MJPEG generator ───────────────────────────────────────────────────────────

def _gen_mjpeg():
    while True:
        frame = _get_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buf.tobytes()
            + b"\r\n"
        )


# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route("/stream")
def stream():
    return Response(_gen_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot")
def snapshot():
    frame = _get_frame()
    if frame is None:
        return "No frame available", 503
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(buf.tobytes(), mimetype="image/jpeg")


@app.route("/health")
def health():
    return {"status": "ok", "has_frame": _frame is not None}


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _running

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--port", type=int, default=STREAM_PORT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-20s %(levelname)-8s  %(message)s",
    )

    _running = True

    if args.dry_run:
        t = threading.Thread(target=_dummy_capture_loop, daemon=True)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        t = threading.Thread(target=_capture_loop, args=(cap,), daemon=True)

    t.start()

    def _shutdown(sig, frame):
        global _running
        log.info("Shutting down camera client")
        _running = False
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("MJPEG stream on http://0.0.0.0:%d/stream", args.port)
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()