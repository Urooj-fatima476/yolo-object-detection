
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="AI Object Detection",
    page_icon="🤖",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "yolov8n.pt"


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


model = load_model()

st.markdown(
    """
    <style>
    .main-title {font-size: 2.5rem; font-weight: 800; margin-bottom: 0.2rem;}
    .subtitle {font-size: 1.05rem; opacity: 0.75; margin-bottom: 1.5rem;}
    .metric-card {padding: 1rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.25);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🤖 AI Object Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">YOLOv8-powered image and video object detection</div>',
    unsafe_allow_html=True,
)

tab_image, tab_video = st.tabs(["🖼️ Image Detection", "🎥 Video Detection"])

CLASS_NAMES = {0: "People", 1: "Bicycle", 2: "Cars", 3: "Motorcycles"}


def get_counts(result):
    counts = {name: 0 for name in CLASS_NAMES.values()}
    if result.boxes is None:
        return counts
    for box in result.boxes:
        cls = int(box.cls[0])
        if cls in CLASS_NAMES:
            counts[CLASS_NAMES[cls]] += 1
    return counts


def show_counts(counts):
    cols = st.columns(4)
    for col, name in zip(cols, CLASS_NAMES.values()):
        col.metric(name, counts[name])


with tab_image:
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key="image_upload",
    )

    if uploaded_image is not None:
        data = np.frombuffer(uploaded_image.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("This image could not be read. Please upload a valid JPG or PNG.")
        else:
            with st.spinner("Running YOLO detection..."):
                result = model(frame, verbose=False)[0]
                annotated = result.plot()
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                counts = get_counts(result)

            st.subheader("Detection Result")
            st.image(annotated_rgb, use_container_width=True)
            show_counts(counts)

            ok, encoded = cv2.imencode(".png", annotated)
            if ok:
                st.download_button(
                    "⬇️ Download Detected Image",
                    data=encoded.tobytes(),
                    file_name="detected_image.png",
                    mime="image/png",
                    use_container_width=True,
                )


with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"],
        key="video_upload",
    )

    if uploaded_video is not None:
        if st.button("▶️ Start Video Detection", type="primary", use_container_width=True):
            input_suffix = Path(uploaded_video.name).suffix or ".mp4"

            with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as input_file:
                input_file.write(uploaded_video.getbuffer())
                input_path = input_file.name

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Could not open this video.")
                os.unlink(input_path)
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                fps = fps if fps and fps > 0 else 30
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Keep the original aspect ratio while limiting very large videos.
                max_width = 1280
                if width > max_width:
                    new_width = max_width
                    new_height = int(height * max_width / width)
                else:
                    new_width, new_height = width, height

                writer = cv2.VideoWriter(
                    output_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (new_width, new_height),
                )

                progress = st.progress(0)
                status = st.empty()
                preview = st.empty()

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
                frame_index = 0
                last_counts = {name: 0 for name in CLASS_NAMES.values()}

                with st.spinner("Processing video with YOLO..."):
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame = cv2.resize(frame, (new_width, new_height))
                        result = model(frame, verbose=False)[0]
                        annotated = result.plot()

                        last_counts = get_counts(result)
                        cv2.putText(
                            annotated,
                            " | ".join(
                                f"{name}: {last_counts[name]}"
                                for name in CLASS_NAMES.values()
                            ),
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                        )

                        writer.write(annotated)

                        # Show occasional preview without storing every frame in RAM.
                        if frame_index % max(int(fps), 1) == 0:
                            preview.image(
                                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                channels="RGB",
                                use_container_width=True,
                            )

                        frame_index += 1
                        if total_frames:
                            progress.progress(min(frame_index / total_frames, 1.0))
                            status.write(f"Processed {frame_index:,} / {total_frames:,} frames")
                        else:
                            status.write(f"Processed {frame_index:,} frames")

                cap.release()
                writer.release()

                st.success("Video detection finished.")
                st.subheader("Last-frame Detection Counts")
                show_counts(last_counts)

                with open(output_path, "rb") as output_file:
                    st.download_button(
                        "⬇️ Download Detected Video",
                        data=output_file.read(),
                        file_name="detected_video.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )

                try:
                    os.unlink(input_path)
                    os.unlink(output_path)
                except OSError:
                    pass

st.divider()
st.caption("YOLOv8 Object Detection • Image & Video • Browser-based interface")
