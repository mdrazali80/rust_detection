import streamlit as st
import cv2
import time
import tempfile
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# === Default Config ===
DEFAULT_MODEL_PATH = "yolo11n.pt"
RTSP_URL = "rtsp://username:password@ip_address:port/endpoint"

# === Streamlit UI Setup ===
st.set_page_config(
    page_title="YOLOv11 Object Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
    )


def list_available_cameras(max_ids=5):
    available = []
    for i in range(max_ids):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(i)
        cap.release()
    return available


st.title(" 🤖 Rust Detection Module ֎🇦🇮")
st.caption("Choose from Image, Video, or RTSP camera stream")

logo = Image.open("logo-uthm-web.png")
# === Sidebar: Model upload ===
with st.sidebar:
    st.sidebar.image(logo, use_container_width=True)
    st.header("⚙️ Settings")
    model_file = st.file_uploader("📦 Upload AI detection model (.pt)", type=["pt"])
    if model_file:
        model_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pt").name
        with open(model_path, "wb") as f:
            f.write(model_file.read())
    else:
        model_path = DEFAULT_MODEL_PATH

    model = YOLO(model_path)

    # Detection filters
    all_classes = model.model.names
    selected_classes = st.multiselect("🎯 Detect only these classes:", list(all_classes.values()), default=list(all_classes.values()))

    export_results = st.checkbox("📝 Export detection results (CSV)", value=False)

# === Layout: Dual display ===
col1, col2 = st.columns(2)
with col1:
    st.subheader("🖼️ Original Source")
    original_display = st.image([], use_container_width=True)
with col2:
    st.subheader("📡 Detected Items")
    processed_display = st.image([], use_container_width=True)

fps_display = st.empty()
object_count_display = st.empty()




# === Detection Filtering Helper ===
def filter_detections(results):
    filtered_boxes = []
    class_map = model.model.names

    for box in results.boxes.data:
        cls_id = int(box[5])
        cls_name = class_map.get(cls_id, "unknown")
        if cls_name in selected_classes:
            filtered_boxes.append(box)

    results.boxes.data = np.array(filtered_boxes) if filtered_boxes else np.empty((0, 6))
    return results

# === Display Handler ===
def display_results(original, annotated, fps=None, class_counts=None):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    original_display.image(original_rgb)
    processed_display.image(annotated_rgb)

    if fps:
        fps_display.markdown(f"🕒 **FPS:** `{fps:.2f}`")
    if class_counts:
        counts_text = ", ".join([f"{k}: {v}" for k, v in class_counts.items()])
        object_count_display.markdown(f"🧮 **Objects Detected:** {counts_text}")

# === Detection & Counting Function ===
def process_frame(frame):
    results = model(frame, verbose=False)[0]
    results = filter_detections(results)

    # Count objects
    class_counts = {}
    for box in results.boxes.data:
        cls_id = int(box[5])
        cls_name = model.model.names[cls_id]
        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return results.plot(), class_counts, results

# === CSV Export Helper ===
def extract_csv(results, frame_id):
    records = []
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls_id = box.tolist()
        cls_name = model.model.names[int(cls_id)]
        records.append({
            "frame": frame_id,
            "class": cls_name,
            "confidence": round(conf, 3),
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)
        })
    return records


#list all available camera
def list_available_cameras(max_ids=5):
    available = []
    for i in range(max_ids):
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            available.append(i)
        cap.release()
    return available

# === Main Modes ===
input_mode = st.radio("Select Input Source:", ["Image", "Video", "Camera"], horizontal=True)

detection_log = []

# === IMAGE MODE ===
if input_mode == "Image":
    uploaded_image = st.file_uploader("📁 Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        # Read and process the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

         # Run YOLO
        annotated, counts, results = process_frame(image)

        # Create two side-by-side image placeholders
        col_original, col_detected = st.columns(2)
        with col_original:
            original_placeholder = st.empty()
            original_placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
        with col_detected:
            detected_placeholder = st.empty()
            detected_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"**Detected Image** | Objects: {counts}", use_container_width=True)

    
    if export_results:
        detection_log += extract_csv(results, frame_id=0)

# === VIDEO MODE ===
elif input_mode == "Video":
    video_file = st.file_uploader("📁 Upload Video", type=["mp4", "avi", "mov", "mkv"])
    if video_file:
        temp_vid = tempfile.NamedTemporaryFile(delete=False)
        temp_vid.write(video_file.read())
        cap = cv2.VideoCapture(temp_vid.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS)    



        prev_time = time.time()

        if "video_frame_idx" not in st.session_state:
            st.session_state.video_frame_idx = 0

        # Setup session state
        if "last_frame" not in st.session_state:
            st.session_state.last_frame = None
            st.session_state.last_annotated = None
            st.session_state.last_counts = ""
            st.session_state.last_fps = 0.0

        # UI controls
        playback_mode = st.radio(
            "🎮 Playback Control",
            ["▶️ Play", "⏸️ Pause", "⏹️ Stop"],
            horizontal=True,
            key="video_playback_control"
        )

        # Empty containers for image display
        col1, col2 = st.columns(2)
        original_placeholder = col1.empty()
        annotated_placeholder = col2.empty()

        while cap.isOpened():
            if playback_mode == "⏹️ Stop":
                st.warning("🛑 Video playback stopped.")
                break

            elif playback_mode == "⏸️ Pause":
                # Draw last stored frame from session state
                if st.session_state.last_frame is not None:
                    original_placeholder.image(
                        cv2.cvtColor(st.session_state.last_frame, cv2.COLOR_BGR2RGB),
                        caption="Original",
                        use_container_width=True
                    )
                    annotated_placeholder.image(
                        cv2.cvtColor(st.session_state.last_annotated, cv2.COLOR_BGR2RGB),
                        caption=f"YOLO Detection | FPS: {st.session_state.last_fps:.2f} | Objects: {st.session_state.last_counts}",
                        use_container_width=True
                    )
                time.sleep(0.2)
                continue

            if playback_mode =="▶️ Play":
                cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.video_frame_idx)
                ret, frame = cap.read()
                if not ret:
                    st.info("✅ End of video.")
                    break

                # Run YOLO
                annotated, counts, results = process_frame(frame)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                         # Update display
                original_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    caption="Original",
                    use_container_width=True
                )
                annotated_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption=f"YOLO Detection | FPS: {fps:.2f} | Objects: {counts}",
                    use_container_width=True
                )

               
                # Store last frame to session state for pause display
                st.session_state.last_frame = frame
                st.session_state.last_annotated = annotated
                st.session_state.last_counts = counts
                st.session_state.last_fps = fps

                if export_results:
                    detection_log += extract_csv(results,st.session_state.video_frame_idx)
                
                st.session_state.video_frame_idx += 1

        cap.release()



# === RTSP MODE ===
elif input_mode == "Camera":
    st.subheader("🎥 Camera Source")
    source_type = st.radio("Choose input source:", ["Webcam", "RTSP URL"], horizontal=True)

    if source_type == "Webcam":
        def list_available_cameras(max_ids=5):
            available = []
            for i in range(max_ids):
                cap = cv2.VideoCapture(i)
                if cap.read()[0]:
                    available.append(i)
                cap.release()
            return available

        available_cams = list_available_cameras()
        if available_cams:
            selected_cam_id = st.selectbox("Select webcam device ID:", available_cams)
        else:
            st.warning("⚠️ No webcam detected.")
            selected_cam_id = None
    else:
        rtsp_url = st.text_input("🔗 Enter RTSP URL", "rtsp://your-ip")


    if st.toggle("▶️ Start Camera Stream"):
        if source_type == "Webcam" and selected_cam_id is not None:
            cap = cv2.VideoCapture(selected_cam_id)
        elif source_type == "RTSP URL" and rtsp_url:
            cap = cv2.VideoCapture(rtsp_url)
        else:
            st.error("❌ Invalid source selected.")
            cap = None

        if cap and cap.isOpened():
            # Placeholders and detection loop same as before...
            col1, col2 = st.columns(2)
            original_placeholder = col1.empty()
            detected_placeholder = col2.empty()
            text_placeholder = st.empty()

            prev_time = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⚠️ Failed to read frame.")
                    break

                #results = model(frame, verbose=False)[0]
                #annotated = results.plot()
                annotated, counts, results = process_frame(frame)

                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time

                original_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
                detected_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"YOLO Detection | FPS: {fps:.2f} | Objects: {counts}", use_container_width=True)
                #text_placeholder.markdown(f"Detected Objects: {len(results.boxes.data)}")

                if export_results:
                    detection_log += extract_csv(results, frame_id)

                time.sleep(0.2)

            cap.release()
        else:
            st.error("❌ Could not open selected video source.")

  
# === Export as CSV ===
if export_results and detection_log:
    df = pd.DataFrame(detection_log)
    st.download_button(
        label="⬇️ Download Detections CSV",
        data=df.to_csv(index=False).encode(),
        file_name="detections.csv",
        mime="text/csv"
    )
