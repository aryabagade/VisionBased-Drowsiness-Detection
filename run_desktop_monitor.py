"""
DriveGuard AI - Native Desktop Real-Time Monitor
High-performance live webcam application using OpenCV GUI.
Runs at 30+ FPS on laptop camera with real-time HUD, audio alerts, and event logging.
Press 'q' to quit, 'm' to mute/unmute audio, 's' to save snapshot.
"""

import os
import sys
import time
from datetime import datetime
import cv2

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.detector import DriverMonitoringEngine

SNAPSHOTS_DIR = os.path.join(BASE_DIR, "logs", "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)


def run_live_monitor(camera_index=0, model_path="models/pretrainedModel.pth"):
    print("=" * 70)
    print("  DRIVEGUARD AI - REAL-TIME DRIVER DROWSINESS & MOTION MONITOR  ")
    print("  Department of AI & DS, Vishwakarma Institute of Technology, Pune ")
    print("=" * 70)
    print("[1] Initializing Driver Monitoring AI Engine...")

    engine = DriverMonitoringEngine(
        model_path=model_path,
        ear_threshold=0.22,
        closure_time_threshold=1.5,
        mar_threshold=0.65,
        yawn_time_threshold=1.8,
        yaw_threshold=22.0,
        pitch_threshold=18.0,
        distraction_time_threshold=2.0,
        enable_audio=True,
        log_file=os.path.join(BASE_DIR, "logs", "driver_events.csv")
    )

    print(f"[2] Connecting to Laptop Camera (index={camera_index})...")
    # On Windows, try DirectShow for instant connection without delay
    if os.name == "nt":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(camera_index)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}!")
        print("Please verify:")
        print(" - Camera is plugged in or laptop webcam is not covered.")
        print(" - Windows Camera permissions allow desktop apps to access the camera.")
        print(" - No other application (Zoom, Teams, Chrome) is currently using the camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[3] Live Camera feed initialized successfully!")
    print("=" * 70)
    print("HOTKEYS CONTROLS:")
    print("  [q] : Quit and close monitor")
    print("  [m] : Toggle Audio Alarm Buzzer ON/OFF")
    print("  [s] : Save annotated snapshot to logs/snapshots/")
    print("=" * 70)

    fps_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    window_name = "DriveGuard AI - Real-Time Driver Monitoring"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARNING] Frame dropped or camera disconnected.")
                time.sleep(0.05)
                continue

            # Mirror frame for natural driver view
            frame = cv2.flip(frame, 1)

            # Process frame through full multi-parameter AI pipeline
            annotated_frame, telemetry = engine.process_frame(frame)

            # FPS calculation
            fps_count += 1
            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                current_fps = fps_count / elapsed
                fps_count = 0
                fps_start_time = time.time()

            # Render FPS and Audio status in bottom-left corner
            audio_status = "ON" if engine.enable_audio else "MUTED"
            cv2.putText(annotated_frame, f"FPS: {current_fps:.1f} | Buzzer: {audio_status} [m] | Snap: [s] | Quit: [q]",
                        (15, annotated_frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 'q' or ESC
                print("[DriveGuard AI] Exiting monitoring application...")
                break
            elif key == ord('m'):
                engine.enable_audio = not engine.enable_audio
                print(f"[DriveGuard AI] Audio Buzzer set to: {'ENABLED' if engine.enable_audio else 'MUTED'}")
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{timestamp}_{telemetry['driver_state']}.jpg")
                cv2.imwrite(snap_path, annotated_frame)
                print(f"[DriveGuard AI] Snapshot saved: {snap_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[DriveGuard AI] Session completed. Telemetry logged to: logs/driver_events.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DriveGuard AI - Native Desktop Camera Monitor")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--model", type=str, default="models/pretrainedModel.pth", help="Path to trained .pth model")
    args = parser.parse_args()

    run_live_monitor(camera_index=args.camera, model_path=args.model)
