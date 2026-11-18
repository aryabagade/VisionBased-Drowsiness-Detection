"""
DriveGuard AI - Streamlit Web Dashboard
Department of Artificial Intelligence & Data Science
Vishwakarma Institute of Technology, Pune
"""

import os
import sys
import time
import threading
from pathlib import Path
from datetime import datetime

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.detector import DriverMonitoringEngine

# =========================================================
# 1. STREAMLIT CONFIGURATION & CUSTOM CSS
# =========================================================
st.set_page_config(
    page_title="DriveGuard AI - Driver Monitoring",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4A5568;
        margin-bottom: 15px;
    }
    .state-card {
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.3rem;
        text-align: center;
        color: white;
        margin-bottom: 12px;
    }
    .state-normal { background-color: #2F855A; }
    .state-warning { background-color: #D69E2E; }
    .state-drowsy { background-color: #E53E3E; }
    .state-distracted { background-color: #DD6B20; }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 2. SIDEBAR CONFIGURATION
# =========================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/car--v1.png", width=70)
    st.title("DriveGuard AI")
    st.caption("AI-Based Real-Time Driver Drowsiness & Motion Monitoring")
    st.markdown("---")

    st.subheader("⚙ Detection Parameters")
    ear_thresh = st.slider("EAR Threshold (Eye Closure)", 0.15, 0.35, 0.22, 0.01)
    closure_sec = st.slider("Prolonged Closure Time (Tc in sec)", 0.5, 3.0, 1.5, 0.1)
    mar_thresh = st.slider("MAR Threshold (Yawning)", 0.40, 0.90, 0.65, 0.05)
    yaw_thresh = st.slider("Head Turn Yaw Threshold (deg)", 15.0, 45.0, 22.0, 1.0)
    distract_sec = st.slider("Head Distraction Time (sec)", 1.0, 4.0, 2.0, 0.5)
    enable_audio = st.checkbox("Enable Audio Buzzer Alert", value=True)

    st.markdown("---")
    st.subheader("System Specifications")
    st.write("**Model:** `DrowsyEyeNet` (.pth)")
    st.write("**Input Resolution:** 1280 × 720 (30 FPS)")
    st.write("**Landmarks:** MediaPipe 468 3D Points")
    st.write("**Head Pose:** solvePnP Euler (Yaw, Pitch, Roll)")
    st.write("**Affiliation:** VIT Pune (Dept. of AI & DS)")


# =========================================================
# 3. BACKGROUND CAMERA THREAD WORKER
# =========================================================
class StreamlitCameraWorker:
    def __init__(self, ear_th, closure_th, mar_th, yaw_th, distract_th, audio_on):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.running = False
        self.latest_frame = None
        self.latest_telemetry = None
        self.error_msg = None

        model_path = Path(__file__).resolve().parent / "pretrainedModel.pth"
        if not model_path.is_file():
            model_path = BASE_DIR / "models" / "pretrainedModel.pth"

        self.engine = DriverMonitoringEngine(
            model_path=str(model_path) if model_path.is_file() else None,
            ear_threshold=ear_th,
            closure_time_threshold=closure_th,
            mar_threshold=mar_th,
            yawn_time_threshold=1.8,
            yaw_threshold=yaw_th,
            pitch_threshold=18.0,
            distraction_time_threshold=distract_th,
            enable_audio=audio_on,
            log_file=str(BASE_DIR / "logs" / "driver_events.csv")
        )

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.5)
        self.running = False

    def get_snapshot(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            telemetry = dict(self.latest_telemetry) if self.latest_telemetry else {}
            return frame, telemetry, self.running, self.error_msg

    def _capture_loop(self):
        cap = None
        try:
            if os.name == "nt":
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(0)
            else:
                cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                raise RuntimeError("Could not connect to laptop webcam. Ensure camera is plugged in and permissions are enabled.")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.04)
                    continue

                frame = cv2.flip(frame, 1)
                annotated_frame, telemetry = self.engine.process_frame(frame)

                with self.lock:
                    self.latest_frame = annotated_frame
                    self.latest_telemetry = telemetry

        except Exception as e:
            with self.lock:
                self.error_msg = str(e)
        finally:
            if cap:
                cap.release()
            with self.lock:
                self.running = False


# =========================================================
# 4. SESSION STATE INITIALIZATION
# =========================================================
if "camera_worker" not in st.session_state:
    st.session_state.camera_worker = StreamlitCameraWorker(
        ear_thresh, closure_sec, mar_thresh, yaw_thresh, distract_sec, enable_audio
    )

worker = st.session_state.camera_worker

# Update engine parameters dynamically
worker.engine.ear_threshold = ear_thresh
worker.engine.closure_time_threshold = closure_sec
worker.engine.mar_threshold = mar_thresh
worker.engine.yaw_threshold = yaw_thresh
worker.engine.distraction_time_threshold = distract_sec
worker.engine.enable_audio = enable_audio


# =========================================================
# 5. HEADER & CONTROLS
# =========================================================
st.markdown("<div class='main-header'>AI-Based Real-Time Driver Drowsiness & Motion Monitoring System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Inventors: Palomi Gawli, Sankalp Akkar, Suyash Ambi, Arya Bagade | Department of AI & DS, VIT Pune</div>", unsafe_allow_html=True)

btn_c1, btn_c2, btn_c3, btn_c4 = st.columns([1.2, 1.2, 2, 2])
with btn_c1:
    if st.button("▶ Start Camera", type="primary", use_container_width=True):
        worker.start()

with btn_c2:
    if st.button("■ Stop Camera", use_container_width=True):
        worker.stop()

with btn_c3:
    log_path = BASE_DIR / "logs" / "driver_events.csv"
    if log_path.exists():
        with open(log_path, "rb") as f:
            st.download_button(
                label="📥 Download Event Log (CSV)",
                data=f,
                file_name=f"driver_monitoring_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

st.divider()

# =========================================================
# 6. LIVE DASHBOARD DISPLAY FRAGMENT
# =========================================================
@st.fragment(run_every=0.15)
def display_live_dashboard():
    frame, telemetry, running, error = worker.get_snapshot()

    if error:
        st.error(f"Camera Error: {error}")

    driver_state = telemetry.get("driver_state", "STANDBY")

    # State Badge Banner
    if driver_state == "NORMAL":
        st.markdown(f"<div class='state-card state-normal'>DRIVER STATE: {driver_state} (SAFE)</div>", unsafe_allow_html=True)
    elif driver_state == "WARNING" or "WARNING" in driver_state:
        st.markdown(f"<div class='state-card state-warning'>DRIVER STATE: {driver_state}</div>", unsafe_allow_html=True)
    elif "DROWSY" in driver_state:
        st.markdown(f"<div class='state-card state-drowsy'>🚨 ALERT: {driver_state} DETECTED! TAKE ACTION</div>", unsafe_allow_html=True)
    elif driver_state == "DISTRACTED":
        st.markdown(f"<div class='state-card state-distracted'>⚠ ALERT: DRIVER DISTRACTED (HEAD TURNED)</div>", unsafe_allow_html=True)
    else:
        st.info("Click '▶ Start Camera' above to begin real-time driver monitoring.")

    # Telemetry Metric Strip
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Avg EAR", f"{telemetry.get('avg_ear', 0.0):.2f}", delta="Closed" if telemetry.get('avg_ear', 0.0) < ear_thresh else "Open")
    m2.metric("Closure Duration", f"{telemetry.get('closure_duration_sec', 0.0):.1f} s", delta=f"Limit: {closure_sec:.1f}s")
    m3.metric("Mouth MAR (Yawn)", f"{telemetry.get('mar', 0.0):.2f}")
    m4.metric("Head Yaw", f"{telemetry.get('head_yaw', 0.0):.0f}°", delta=telemetry.get('head_motion', 'CENTER'))
    m5.metric("Total Blinks", telemetry.get("total_blinks", 0))
    m6.metric("Total Yawns", telemetry.get("total_yawns", 0))

    # Live Frame Display
    col_video, col_info = st.columns([2.5, 1.2])

    with col_video:
        if frame is not None:
            st.image(frame, channels="BGR", use_container_width=True)
        elif running:
            st.info("Connecting to webcam; please wait...")
        else:
            st.caption("Live video stream is stopped. Press 'Start Camera' to resume.")

    with col_info:
        st.subheader("Deep Learning Inference")
        st.write(f"**Left Eye:** `{telemetry.get('left_eye', '-')}` ({telemetry.get('left_conf', '-')})")
        st.write(f"**Right Eye:** `{telemetry.get('right_eye', '-')}` ({telemetry.get('right_conf', '-')})")
        st.write(f"**Head Pitch:** `{telemetry.get('head_pitch', 0.0):.1f}°`")
        st.write(f"**Head Roll:** `{telemetry.get('head_roll', 0.0):.1f}°`")
        st.write(f"**Audio Buzzer:** `{'ON' if enable_audio else 'MUTED'}`")

        # Live Event Log Preview
        log_path = BASE_DIR / "logs" / "driver_events.csv"
        if log_path.exists():
            try:
                df = pd.read_csv(log_path)
                if not df.empty:
                    st.caption("Recent Logged Incidents:")
                    st.dataframe(df.tail(6)[["Timestamp", "Driver_State", "Avg_EAR", "Eye_Closure_Sec", "Motion_State"]], hide_index=True)
            except Exception:
                pass


display_live_dashboard()

# =========================================================
# 7. DOCUMENTATION TABS & PATENT SPECIFICATIONS
# =========================================================
st.divider()
tab1, tab2, tab3 = st.tabs(["📋 System Architecture", "📐 Mathematical Formulations", "📜 Patent Claims & Inventors"])

with tab1:
    st.markdown("""
    ### System Pipeline Architecture
    1. **Frame Acquisition:** Laptop webcam captures 30 FPS color stream (1280 × 720).
    2. **Face & Eye Landmark Extraction:** MediaPipe Face Mesh identifies 468 spatial coordinates with sub-millimeter precision.
    3. **Deep Learning Inference:** Left and right eye patches are normalized and fed into `DrowsyEyeNet` (.pth) for classification.
    4. **Eye Aspect Ratio (EAR) & Mouth Aspect Ratio (MAR):** Continuous geometric aspect tracking.
    5. **3D Head Pose Analysis (solvePnP):** Estimates driver head rotation (Yaw, Pitch, Roll) relative to the camera focal center.
    6. **Temporal Reasoning:** Differentiates healthy 150–350ms blinks from dangerous prolonged eye closure ($T_e \ge T_c$).
    7. **Unified Decision Engine:** Assigns `NORMAL`, `WARNING`, `DROWSY`, or `DISTRACTED` states.
    8. **Audio/Visual Alert:** Immediate buzzer sound and red visual warning banner to prevent collisions.
    """)

with tab2:
    st.markdown(r"""
    ### Mathematical Modeling
    - **Eye Aspect Ratio (EAR):**
      $$EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \cdot ||p_1 - p_4||}$$
    - **Mouth Aspect Ratio (MAR):**
      $$MAR = \frac{||p_{top} - p_{bottom}||}{||p_{left} - p_{right}||}$$
    - **Prolonged Eye Closure Condition:**
      $$T_e = t_1 - t_0 \ge T_c \quad (T_c \approx 1.5\text{ s})$$
    - **Head Pose Rotation via solvePnP:**
      $$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \cdot [R | T] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}$$
    """)

with tab3:
    st.markdown("""
    **Invention Title:** AI-Based Real-Time Driver Drowsiness and Motion Monitoring System Using Deep Learning  
    **Applicant:** Vishwakarma Institute of Technology, Pune & Vishwakarma University  
    **Inventors:**
    - Palomi Sandeep Gawli (`palomi.gawli@vit.edu`)
    - Sankalp Santosh Akkar (`sankalp.1251010354@vit.edu`)
    - Suyash Sudhir Ambi (`suyash.1251010552@vit.edu`)
    - Arya Manish Bagade (`arya.1251010556@vit.edu`)
    """)
