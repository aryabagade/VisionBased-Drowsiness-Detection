
import os
import threading
import time
from pathlib import Path

import cv2
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import alexnet
from torchvision import transforms


# =========================================================
# 1. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DriveGuard AI",
    page_icon="🚘",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.environ.get(
        "DRIVEGUARD_MODEL_PATH",
        str(BASE_DIR / "pretrainedModel.pth"),
    )
)

DEVICE = torch.device("cpu")
CLASS_NAMES = ["Class 0", "Class 1"]


# =========================================================
# 3. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_detector(model_path):
    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Place pretrainedModel.pth beside app.py."
        )

    # Create the same AlexNet-based architecture.
    model = alexnet(weights=None)

    model.avgpool = nn.AdaptiveAvgPool2d((7, 7))
    model.classifier[1] = nn.Linear(12544, 4096)

    # IMPORTANT: checkpoint has 2 classes, not 1000.
    model.classifier[6] = nn.Linear(4096, 2)

    try:
        checkpoint = torch.load(
            str(model_path),
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(
            str(model_path),
            map_location=DEVICE,
        )

    # Support common checkpoint formats.
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    # Handle models saved with DataParallel.
    if isinstance(checkpoint, dict) and checkpoint:
        if all(k.startswith("module.") for k in checkpoint):
            checkpoint = {
                k.removeprefix("module."): v
                for k, v in checkpoint.items()
            }

    # Strict loading ensures incompatible layers are not ignored.
    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE)
    model.eval()

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_eye_tree_eyeglasses.xml"
    )

    eye_cascade = cv2.CascadeClassifier(cascade_path)

    if eye_cascade.empty():
        raise RuntimeError("Could not load the OpenCV eye cascade.")

    # Must match the preprocessing used to train your model.
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, eye_cascade, preprocess


def classify_eye(eye_bgr, model, preprocess):
    eye_rgb = cv2.cvtColor(eye_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(eye_rgb)
    tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)

    return int(prediction.item()), float(confidence.item())


# =========================================================
# 4. DIRECT OPENCV LIVE CAMERA
# =========================================================

class LiveCameraMonitor:
    def __init__(self, model, eye_cascade, preprocess):
        self.model = model
        self.eye_cascade = eye_cascade
        self.preprocess = preprocess

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

        self.latest_frame = None
        self.error = None
        self.running = False
        self.frames_processed = 0
        self.eye_count = 0
        self.predictions = []

    def start(self):
        if self.thread and self.thread.is_alive():
            return

        self.stop_event.clear()

        with self.lock:
            self.latest_frame = None
            self.error = None
            self.frames_processed = 0
            self.eye_count = 0
            self.predictions = []
            self.running = True

        self.thread = threading.Thread(
            target=self._camera_loop,
            daemon=True,
        )
        self.thread.start()

    def stop(self):
        self.stop_event.set()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

        with self.lock:
            self.running = False

    def snapshot(self):
        with self.lock:
            frame = (
                self.latest_frame.copy()
                if self.latest_frame is not None
                else None
            )

            return {
                "frame": frame,
                "error": self.error,
                "running": self.running,
                "frames": self.frames_processed,
                "eyes": self.eye_count,
                "predictions": list(self.predictions),
            }

    def _camera_loop(self):
        camera = None

        try:
            # Prefer DirectShow on Windows.
            if os.name == "nt":
                camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

                if not camera.isOpened():
                    camera.release()
                    camera = cv2.VideoCapture(0)
            else:
                camera = cv2.VideoCapture(0)

            if not camera.isOpened():
                raise RuntimeError(
                    "Webcam could not be opened. Close other camera "
                    "applications and check Windows camera permissions."
                )

            # Request higher resolution; hardware support varies.
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            count = 0

            while not self.stop_event.is_set():
                ok, frame = camera.read()

                if not ok or frame is None:
                    time.sleep(0.05)
                    continue

                # Mirror the webcam image.
                frame = cv2.flip(frame, 1)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                eyes = self.eye_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 20),
                )

                current_predictions = []

                for x, y, w, h in eyes:
                    crop = frame[y:y + h, x:x + w]

                    if crop.size == 0:
                        continue

                    class_id, score = classify_eye(
                        crop,
                        self.model,
                        self.preprocess,
                    )

                    class_name = (
                        CLASS_NAMES[class_id]
                        if class_id < len(CLASS_NAMES)
                        else f"Class {class_id}"
                    )

                    current_predictions.append({
                        "Eye": len(current_predictions) + 1,
                        "Prediction": class_name,
                        "Confidence": f"{score * 100:.2f}%",
                    })

                    # Yellow boxes avoid assuming either class is
                    # safe or unsafe before label verification.
                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 220, 255),
                        2,
                    )

                    label = f"{class_name} | {score * 100:.1f}%"

                    cv2.putText(
                        frame,
                        label,
                        (x, max(y - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )

                count += 1

                # Live status overlay.
                cv2.rectangle(
                    frame,
                    (0, 0),
                    (frame.shape[1], 65),
                    (25, 25, 25),
                    -1,
                )

                cv2.putText(
                    frame,
                    "DRIVEGUARD AI | LIVE CAMERA",
                    (15, 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    frame,
                    f"Eyes detected: {len(current_predictions)}",
                    (15, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

                with self.lock:
                    self.latest_frame = frame.copy()
                    self.frames_processed = count
                    self.eye_count = len(current_predictions)
                    self.predictions = current_predictions

        except Exception as exc:
            with self.lock:
                self.error = str(exc)

        finally:
            if camera is not None:
                camera.release()

            with self.lock:
                self.running = False


# =========================================================
# 5. DASHBOARD HEADER AND MODEL INITIALIZATION
# =========================================================

st.title("🚘 DriveGuard AI")
st.caption(
    "Real-time driver eye monitoring using computer vision"
)

with st.sidebar:
    st.header("System Information")
    st.write("**Camera:** Direct OpenCV webcam")
    st.write("**Inference:** PyTorch on CPU")
    st.write("**Input size:** 224 × 224")
    st.write("**Architecture:** AlexNet")
    st.write(f"**Model file:** {MODEL_PATH.name}")

try:
    model, eye_cascade, preprocess = load_detector(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the AI model: {exc}")
    st.info(
        "Check the model file and verify that its architecture "
        "and preprocessing match the original training code."
    )
    st.stop()


# =========================================================
# 6. SESSION STATE
# =========================================================

if "monitor" not in st.session_state:
    st.session_state.monitor = LiveCameraMonitor(
        model,
        eye_cascade,
        preprocess,
    )

monitor = st.session_state.monitor


# =========================================================
# 7. CAMERA CONTROLS
# =========================================================

start_col, stop_col, _ = st.columns([1, 1, 3])

with start_col:
    if st.button(
        "▶ Start Live Camera",
        type="primary",
        use_container_width=True,
    ):
        monitor.start()

with stop_col:
    if st.button(
        "■ Stop Camera",
        use_container_width=True,
    ):
        monitor.stop()


# =========================================================
# 8. LIVE DISPLAY
# =========================================================

@st.fragment(run_every=0.2)
def show_live_dashboard():
    data = monitor.snapshot()

    if data["error"]:
        st.error(f"Camera error: {data['error']}")

    st.subheader("Live Camera Feed")

    if data["frame"] is not None:
        st.image(
            data["frame"],
            channels="BGR",
            use_container_width=True,
        )
    elif data["running"]:
        st.info("Opening webcam; waiting for the first frame...")
    else:
        st.info("Click Start Live Camera to begin.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Eyes Detected", data["eyes"])
    col2.metric("Frames Processed", data["frames"])
    col3.metric(
        "Camera Status",
        "LIVE" if data["running"] else "STOPPED",
    )

    st.subheader("Latest Predictions")

    if data["predictions"]:
        st.dataframe(
            data["predictions"],
            use_container_width=True,
            hide_index=True,
        )
    elif data["running"]:
        st.caption(
            "No eyes detected in the latest frame. "
            "Try facing the camera in good lighting."
        )
    else:
        st.caption("Predictions will appear when the camera starts.")


show_live_dashboard()


# =========================================================
# 9. PROJECT DESCRIPTION AND LIMITATIONS
# =========================================================

st.divider()
st.subheader("System Workflow")

st.write("1. OpenCV captures live frames from the laptop webcam.")
st.write("2. Haar Cascade identifies candidate eye regions.")
st.write("3. The trained PyTorch model classifies each eye crop.")
st.write("4. Live predictions and annotated frames update on the dashboard.")

st.warning(
    "Class 0 and Class 1 are generic labels until verified against "
    "the training dataset. Do not interpret them as open/closed eyes "
    "or alert/drowsy status until the class mapping is confirmed. "
    "This is an academic prototype, not a validated safety system."
)

st.caption(
    "This code does not save video recordings to disk. "
    "Camera frames are processed by the local Python application."
)
