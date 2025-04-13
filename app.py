import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
import tempfile
import zipfile
import glob
from detector import detect_damage, estimate_position_cm, load_model

# === Setup Paths ===
OLD_MODEL_PATH = "models/old_best.pt"
NEW_MODEL_PATH = "models/new_best.pt"

# === Load Models Once ===
old_model = load_model(OLD_MODEL_PATH)
new_model = load_model(NEW_MODEL_PATH)

# === Streamlit UI Config ===
st.set_page_config(page_title="🧠 Fabric Defect Dashboard", layout="wide")
st.title("🧵 Fabric Defect Detection Dashboard")

# === Session State Init ===
if "inputs" not in st.session_state:
    st.session_state.inputs = []
if "results" not in st.session_state:
    st.session_state.results = []

# === Tabs ===
tab1, tab2, tab3, tab4 = st.tabs(["📂 Input", "🧠 Predictions", "📊 Analytics", "📅 Export"])

# ----------------------------------------------------------
# 📂 TAB 1 - INPUT
# ----------------------------------------------------------
with tab1:
    st.header("📂 Choose Input Source")
    input_mode = st.radio("Select Input Type", ["Image", "Video", "Webcam", "Image Folder"])

    if input_mode == "Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.session_state.inputs = [("image", image, uploaded_file.name, old_model)]

    elif input_mode == "Video":
        video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            output_path = os.path.join(tempfile.gettempdir(), "annotated_output.mp4")
            out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))

            all_results = []
            frame_index = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, results = detect_damage(frame, new_model)
                for r in results:
                    r["position_cm"] = estimate_position_cm(r["center"], frame_index=frame_index, fps=fps)
                    r["source"] = f"video_frame_{frame_index}"
                all_results.extend(results)
                out.write(annotated)
                frame_index += 1

            cap.release()
            out.release()

            st.success("Video processed.")
            st.session_state.inputs = [("video", output_path)]
            st.session_state.results = all_results

    elif input_mode == "Webcam":
        st.header("🧠 Real-Time Webcam Feed")
        st.warning("Press the button below to start webcam feed.")

        # Button to trigger webcam feed
        if st.button("Start Webcam"):
            cap = cv2.VideoCapture(0)  # 0 means the default webcam

            # Check if the webcam is opened
            if not cap.isOpened():
                st.error("Could not open webcam.")
            else:
                stframe = st.empty()  # Create an empty container for real-time video
                all_results = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to grab frame.")
                        break

                    # Apply defect detection to the current frame
                    annotated_frame, results = detect_damage(frame, new_model)

                    # Convert the frame (BGR) to RGB for Streamlit display
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                    # Show the frame in the Streamlit app
                    stframe.image(annotated_frame_rgb, channels="RGB", use_column_width=True)

                    # Optionally, process additional info or store results
                    for result in results:
                        result["position_cm"] = estimate_position_cm(result["center"], frame_index=0, fps=30)
                        result["source"] = "webcam"
                    all_results.extend(results)

                st.session_state.inputs = [("webcam", all_results)]
                st.session_state.results = all_results
                cap.release()

    elif input_mode == "Image Folder":
        zip_file = st.file_uploader("Upload a ZIP folder of images", type=["zip"])
        if zip_file:
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(tmpdir)
                paths = glob.glob(os.path.join(tmpdir, "**", "*.*"), recursive=True)
                images = []
                for p in paths:
                    if p.lower().endswith((".jpg", ".jpeg", ".png")):
                        img = cv2.imread(p)
                        if img is not None:
                            images.append(("image", img, os.path.basename(p), old_model))
                st.success(f"Found {len(images)} valid images.")
                st.session_state.inputs = images

# ----------------------------------------------------------
# 🧠 TAB 2 - PREDICTIONS
# ----------------------------------------------------------
with tab2:
    st.header("🧠 Detection Results")

    if not st.session_state.inputs:
        st.warning("No input found. Upload from the Input tab.")
    else:
        input_type = st.session_state.inputs[0][0]

        if input_type == "video":
            st.subheader("Annotated Video Output")
            st.video(st.session_state.inputs[0][1])

        elif input_type == "image":
            all_results = []
            for _, image, name, model in st.session_state.inputs:
                st.subheader(f"{name}")
                annotated, results = detect_damage(image, model)
                for r in results:
                    r["position_cm"] = estimate_position_cm(r["center"])
                    r["source"] = name
                st.image(annotated[:, :, ::-1], caption="Detected Image", use_column_width=True)
                if results:
                    st.dataframe(pd.DataFrame(results))
                    all_results.extend(results)
                else:
                    st.info("No defects detected.")
            st.session_state.results = all_results

# ----------------------------------------------------------
# 📊 TAB 3 - ANALYTICS
# ----------------------------------------------------------
with tab3:
    st.header("📊 Defect Summary")
    if not st.session_state.results:
        st.info("Run predictions first to see analytics.")
    else:
        df = pd.DataFrame(st.session_state.results)
        st.bar_chart(df["label"].value_counts())
        st.subheader("Defect Counts per Source")
        st.dataframe(df.groupby("source")["label"].value_counts().unstack(fill_value=0))

# ----------------------------------------------------------
# 📅 TAB 4 - EXPORT
# ----------------------------------------------------------
with tab4:
    st.header("📅 Export Results")

    if not st.session_state.results:
        st.warning("No results to export.")
    else:
        df = pd.DataFrame(st.session_state.results)
        st.download_button("Download CSV", df.to_csv(index=False), "defect_results.csv", "text/csv")
        if st.session_state.inputs[0][0] == "video":
            with open(st.session_state.inputs[0][1], "rb") as f:
                st.download_button("Download Annotated Video", f.read(), "annotated_video.mp4", "video/mp4")
        st.success("Results ready to export.")
