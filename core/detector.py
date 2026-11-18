"""
DriveGuard AI - Core Real-Time Detection & Multi-Parameter Decision Engine
Combines:
  1. Face & Eye Landmark Detection (MediaPipe Face Mesh)
  2. Deep Learning Eye-State Inference (.pth DrowsyEyeNet / AlexNet)
  3. Eye Aspect Ratio (EAR) & Blink Analysis
  4. Mouth Aspect Ratio (MAR) & Yawn Detection
  5. 3D Head Pose Estimation (Yaw, Pitch, Roll via solvePnP)
  6. Temporal Duration Tracking (Distinguishing blinks from prolonged closure)
  7. Unified Driver State Decision Engine: NORMAL, WARNING, DROWSY, DISTRACTED
  8. Audio Buzzer Alert & Event Logging
"""

import os
import time
import math
import csv
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image

# Import model architecture
from models.eye_classifier import DrowsyEyeNet, get_eye_transforms, load_eye_classifier

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

import mediapipe as mp


class DriverMonitoringEngine:
    """
    Unified multi-parameter Driver Drowsiness and Motion Monitoring System.
    """
    def __init__(
        self,
        model_path=None,
        ear_threshold=0.22,
        closure_time_threshold=1.5,
        mar_threshold=0.65,
        yawn_time_threshold=1.8,
        yaw_threshold=22.0,
        pitch_threshold=18.0,
        distraction_time_threshold=2.0,
        enable_audio=True,
        log_file="logs/driver_events.csv"
    ):
        self.device = torch.device("cpu")
        self.ear_threshold = ear_threshold
        self.closure_time_threshold = closure_time_threshold # Tc (seconds)
        self.mar_threshold = mar_threshold
        self.yawn_time_threshold = yawn_time_threshold
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.distraction_time_threshold = distraction_time_threshold
        self.enable_audio = enable_audio
        self.log_file = log_file

        # Initialize PyTorch Eye Classifier
        self.model, self.preprocess = load_eye_classifier(model_path, device="cpu", image_size=64)
        self.class_names = ["Closed", "Open"]

        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 3D generic facial model points for solvePnP (mm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip (landmark 1)
            (0.0, -330.0, -65.0),        # Chin (landmark 152)
            (-225.0, 170.0, -135.0),     # Left eye outer corner (landmark 263)
            (225.0, 170.0, -135.0),      # Right eye outer corner (landmark 33)
            (-150.0, -150.0, -125.0),    # Left mouth corner (landmark 291)
            (150.0, -150.0, -125.0)      # Right mouth corner (landmark 61)
        ], dtype=np.float64)

        # Landmark Indices in MediaPipe Face Mesh
        self.LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380] # p1, p2, p3, p4, p5, p6
        self.RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
        self.MOUTH_LANDMARKS = [13, 14, 78, 308] # top_lip, bottom_lip, left_corner, right_corner

        # Temporal State Tracking
        self.eye_closed_start_time = None
        self.closure_duration = 0.0
        self.consecutive_closed_frames = 0
        self.total_blinks = 0
        self._was_eye_closed = False

        self.yawn_start_time = None
        self.yawn_duration = 0.0
        self.total_yawns = 0
        self._was_yawning = False

        self.distraction_start_time = None
        self.distraction_duration = 0.0

        # Current Frame State Cache
        self.current_state = "NORMAL" # NORMAL, WARNING, DROWSY, DISTRACTED
        self.left_eye_pred = "Open"
        self.right_eye_pred = "Open"
        self.left_eye_conf = 1.0
        self.right_eye_conf = 1.0
        self.avg_ear = 0.30
        self.current_mar = 0.20
        self.head_yaw = 0.0
        self.head_pitch = 0.0
        self.head_roll = 0.0
        self.head_motion_state = "FORWARD"
        self.alert_active = False

        # Audio thread control
        self._buzzer_active = False
        self._buzzer_lock = threading.Lock()

        # Initialize CSV logging
        self._init_csv_log()

    def _init_csv_log(self):
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else ".", exist_ok=True)
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Driver_State", "Left_Eye_CNN", "Right_Eye_CNN",
                    "Avg_EAR", "Eye_Closure_Sec", "MAR", "Yawn_Sec",
                    "Head_Yaw", "Head_Pitch", "Head_Roll", "Motion_State", "Alert_Triggered"
                ])

    def log_event(self):
        """Logs current telemetry to CSV."""
        try:
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    self.current_state,
                    f"{self.left_eye_pred} ({self.left_eye_conf*100:.1f}%)",
                    f"{self.right_eye_pred} ({self.right_eye_conf*100:.1f}%)",
                    f"{self.avg_ear:.3f}",
                    f"{self.closure_duration:.2f}",
                    f"{self.current_mar:.3f}",
                    f"{self.yawn_duration:.2f}",
                    f"{self.head_yaw:.1f}",
                    f"{self.head_pitch:.1f}",
                    f"{self.head_roll:.1f}",
                    self.head_motion_state,
                    self.alert_active
                ])
        except Exception:
            pass

    def _trigger_audio_alert(self, freq=1800, duration=220):
        if not self.enable_audio or not HAS_WINSOUND:
            return
        with self._buzzer_lock:
            if self._buzzer_active:
                return
            self._buzzer_active = True

        def _sound_thread():
            try:
                winsound.Beep(freq, duration)
            except Exception:
                pass
            finally:
                with self._buzzer_lock:
                    self._buzzer_active = False

        threading.Thread(target=_sound_thread, daemon=True).start()

    @staticmethod
    def _euclidean_dist(pt1, pt2):
        return math.dist(pt1, pt2)

    def compute_ear(self, landmarks, eye_indices, img_w, img_h):
        """
        Calculates Eye Aspect Ratio (EAR) for 6 eye landmarks:
        EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
        """
        pts = [(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in eye_indices]
        p1, p2, p3, p4, p5, p6 = pts
        vertical_1 = self._euclidean_dist(p2, p6)
        vertical_2 = self._euclidean_dist(p3, p5)
        horizontal = self._euclidean_dist(p1, p4)
        if horizontal < 1e-5:
            return 0.30
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def compute_mar(self, landmarks, img_w, img_h):
        """
        Calculates Mouth Aspect Ratio (MAR):
        MAR = |top_lip - bottom_lip| / |left_corner - right_corner|
        """
        top_lip = (int(landmarks[13].x * img_w), int(landmarks[13].y * img_h))
        bottom_lip = (int(landmarks[14].x * img_w), int(landmarks[14].y * img_h))
        left_corner = (int(landmarks[78].x * img_w), int(landmarks[78].y * img_h))
        right_corner = (int(landmarks[308].x * img_w), int(landmarks[308].y * img_h))

        vertical = self._euclidean_dist(top_lip, bottom_lip)
        horizontal = self._euclidean_dist(left_corner, right_corner)
        if horizontal < 1e-5:
            return 0.20
        return vertical / horizontal

    def estimate_head_pose(self, landmarks, img_w, img_h):
        """
        Estimates 3D Head Orientation (Yaw, Pitch, Roll) using solvePnP.
        """
        key_indices = [1, 152, 263, 33, 291, 61]
        image_points = np.array([
            (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
            for idx in key_indices
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0, "FORWARD"

        rmat, _ = cv2.Rodrigues(rvec)
        proj_mat = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)
        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        # Normalize yaw & pitch
        motion_state = "FORWARD"
        if yaw < -self.yaw_threshold:
            motion_state = "LOOKING_LEFT"
        elif yaw > self.yaw_threshold:
            motion_state = "LOOKING_RIGHT"
        elif pitch < -self.pitch_threshold:
            motion_state = "HEAD_DOWN (NODDING)"
        elif pitch > self.pitch_threshold + 5:
            motion_state = "HEAD_UP"

        return yaw, pitch, roll, motion_state

    def classify_eye_crop(self, eye_crop_bgr):
        """Passes eye crop to PyTorch DrowsyEyeNet classifier."""
        if eye_crop_bgr is None or eye_crop_bgr.size == 0 or eye_crop_bgr.shape[0] < 5 or eye_crop_bgr.shape[1] < 5:
            return "Open", 0.50
        try:
            eye_rgb = cv2.cvtColor(eye_crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(eye_rgb)
            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)
            pred_class = self.class_names[pred.item()]
            return pred_class, float(conf.item())
        except Exception:
            return "Open", 0.50

    def extract_eye_crop(self, frame, landmarks, eye_indices, img_w, img_h, pad=8):
        """Extracts square bounding box crop around eye landmarks."""
        pts = np.array([(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in eye_indices])
        x, y, w, h = cv2.boundingRect(pts)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        crop = frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2 - x1, y2 - y1)

    def process_frame(self, frame):
        """
        Executes end-to-end processing pipeline on a single video frame:
        Returns annotated frame and current telemetry metrics dict.
        """
        current_time = time.time()
        img_h, img_w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # Face not detected in frame
            self.current_state = "WARNING (NO FACE)"
            self.alert_active = False
            cv2.putText(frame, "STATUS: DRIVER FACE NOT DETECTED", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2, cv2.LINE_AA)
            return frame, self.get_telemetry()

        landmarks = results.multi_face_landmarks[0].landmark

        # 1. EAR (Eye Aspect Ratio) calculation
        left_ear = self.compute_ear(landmarks, self.LEFT_EYE_LANDMARKS, img_w, img_h)
        right_ear = self.compute_ear(landmarks, self.RIGHT_EYE_LANDMARKS, img_w, img_h)
        self.avg_ear = (left_ear + right_ear) / 2.0

        # 2. Extract crops and run Deep-Learning Model Inference
        left_crop, left_box = self.extract_eye_crop(frame, landmarks, self.LEFT_EYE_LANDMARKS, img_w, img_h)
        right_crop, right_box = self.extract_eye_crop(frame, landmarks, self.RIGHT_EYE_LANDMARKS, img_w, img_h)

        self.left_eye_pred, self.left_eye_conf = self.classify_eye_crop(left_crop)
        self.right_eye_pred, self.right_eye_conf = self.classify_eye_crop(right_crop)

        # 3. Yawn / Mouth Aspect Ratio (MAR)
        self.current_mar = self.compute_mar(landmarks, img_w, img_h)
        is_yawning = self.current_mar > self.mar_threshold
        if is_yawning:
            if self.yawn_start_time is None:
                self.yawn_start_time = current_time
            self.yawn_duration = current_time - self.yawn_start_time
            if self.yawn_duration >= self.yawn_time_threshold and not self._was_yawning:
                self.total_yawns += 1
                self._was_yawning = True
        else:
            self.yawn_start_time = None
            self.yawn_duration = 0.0
            self._was_yawning = False

        # 4. 3D Head Pose Estimation
        self.head_yaw, self.head_pitch, self.head_roll, self.head_motion_state = self.estimate_head_pose(landmarks, img_w, img_h)

        # 5. Dual-Verification of Eye Closure:
        # Considered closed if EAR is below threshold OR DL model predicts Closed with high confidence
        dl_closed = (self.left_eye_pred == "Closed" and self.right_eye_pred == "Closed") or \
                    (self.left_eye_pred == "Closed" and self.left_eye_conf > 0.85) or \
                    (self.right_eye_pred == "Closed" and self.right_eye_conf > 0.85)
        ear_closed = self.avg_ear < self.ear_threshold
        is_eye_closed = ear_closed or dl_closed

        # 6. Temporal Analysis (Distinguishing blinks from prolonged closure)
        if is_eye_closed:
            self.consecutive_closed_frames += 1
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time
            self.closure_duration = current_time - self.eye_closed_start_time
            self._was_eye_closed = True
        else:
            if self._was_eye_closed and self.closure_duration < self.closure_time_threshold:
                # Registered a normal complete blink
                self.total_blinks += 1
            self.eye_closed_start_time = None
            self.closure_duration = 0.0
            self.consecutive_closed_frames = 0
            self._was_eye_closed = False

        # 7. Head Distraction Temporal Tracking
        is_head_distracted = (self.head_motion_state in ["LOOKING_LEFT", "LOOKING_RIGHT", "HEAD_DOWN (NODDING)"])
        if is_head_distracted:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            self.distraction_duration = current_time - self.distraction_start_time
        else:
            self.distraction_start_time = None
            self.distraction_duration = 0.0

        # 8. Unified Decision Engine
        if self.closure_duration >= self.closure_time_threshold:
            self.current_state = "DROWSY"
            self.alert_active = True
            self._trigger_audio_alert(freq=2000, duration=250)
            self.log_event()
        elif is_yawning and self.yawn_duration >= self.yawn_time_threshold:
            self.current_state = "DROWSY (YAWNING)"
            self.alert_active = True
            self._trigger_audio_alert(freq=1700, duration=200)
            self.log_event()
        elif self.distraction_duration >= self.distraction_time_threshold:
            self.current_state = "DISTRACTED"
            self.alert_active = True
            self._trigger_audio_alert(freq=1200, duration=180)
            self.log_event()
        elif self.closure_duration > 0.4 or self.distraction_duration > 0.8:
            self.current_state = "WARNING"
            self.alert_active = False
        else:
            self.current_state = "NORMAL"
            self.alert_active = False

        # 9. Render Professional Visual HUD
        self._render_hud(frame, left_box, right_box)

        return frame, self.get_telemetry()

    def _render_hud(self, frame, left_box, right_box):
        """Draws bounding boxes, telemetry cards, and alert banners onto the frame."""
        h, w, _ = frame.shape

        # Color codes by state
        state_colors = {
            "NORMAL": (0, 200, 0),        # Green
            "WARNING": (0, 215, 255),     # Yellow / Gold
            "DROWSY": (0, 0, 240),        # Red
            "DROWSY (YAWNING)": (0, 0, 240),
            "DISTRACTED": (0, 140, 255)   # Orange
        }
        status_color = state_colors.get(self.current_state, (0, 200, 0))

        # Top Header Bar
        cv2.rectangle(frame, (0, 0), (w, 75), (20, 20, 20), -1)
        cv2.line(frame, (0, 75), (w, 75), status_color, 2)

        # Title & State Badge
        cv2.putText(frame, "DRIVEGUARD AI | DRIVER MONITORING SYSTEM", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        badge_text = f"STATE: {self.current_state}"
        cv2.putText(frame, badge_text, (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2, cv2.LINE_AA)

        # Telemetry metrics row
        metrics_text = f"EAR: {self.avg_ear:.2f} | MAR: {self.current_mar:.2f} | Closure: {self.closure_duration:.1f}s / {self.closure_time_threshold:.1f}s | Yaw: {self.head_yaw:.0f} deg"
        cv2.putText(frame, metrics_text, (w - 580, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 240), 1, cv2.LINE_AA)

        pose_text = f"Motion: {self.head_motion_state} | Blinks: {self.total_blinks} | Yawns: {self.total_yawns}"
        cv2.putText(frame, pose_text, (w - 580, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 240), 1, cv2.LINE_AA)

        # Draw Eye Bounding Boxes & Deep Learning Predictions
        lx, ly, lw, lh = left_box
        rx, ry, rw, rh = right_box
        l_color = (0, 255, 0) if self.left_eye_pred == "Open" else (0, 0, 255)
        r_color = (0, 255, 0) if self.right_eye_pred == "Open" else (0, 0, 255)

        cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), l_color, 2)
        cv2.putText(frame, f"L: {self.left_eye_pred} {self.left_eye_conf*100:.0f}%",
                    (lx, max(20, ly - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, l_color, 1, cv2.LINE_AA)

        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), r_color, 2)
        cv2.putText(frame, f"R: {self.right_eye_pred} {self.right_eye_conf*100:.0f}%",
                    (rx, max(20, ry - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_color, 1, cv2.LINE_AA)

        # Red Flashing Alert Banner when Drowsy or Distracted
        if self.alert_active:
            # Flashing bar at bottom
            flash_on = int(time.time() * 4) % 2 == 0
            if flash_on:
                cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 220), -1)
                alert_msg = f"! CRITICAL ALERT: {self.current_state} DETECTED ! TAKE A BREAK"
                cv2.putText(frame, alert_msg, (w // 2 - 290, h - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    def get_telemetry(self):
        """Returns structured dictionary of real-time monitoring parameters."""
        return {
            "driver_state": self.current_state,
            "left_eye": self.left_eye_pred,
            "left_conf": f"{self.left_eye_conf * 100:.1f}%",
            "right_eye": self.right_eye_pred,
            "right_conf": f"{self.right_eye_conf * 100:.1f}%",
            "avg_ear": round(self.avg_ear, 3),
            "closure_duration_sec": round(self.closure_duration, 2),
            "mar": round(self.current_mar, 3),
            "yawn_duration_sec": round(self.yawn_duration, 2),
            "head_yaw": round(self.head_yaw, 1),
            "head_pitch": round(self.head_pitch, 1),
            "head_roll": round(self.head_roll, 1),
            "head_motion": self.head_motion_state,
            "total_blinks": self.total_blinks,
            "total_yawns": self.total_yawns,
            "alert_active": self.alert_active
        }
