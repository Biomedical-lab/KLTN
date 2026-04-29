"""
🖥️ Ứng dụng chính — Hệ thống nhận diện & đếm phương tiện giao thông
KLTN — Lớp CDLT422CNTT — GVHD: Cô Nguyễn Thị Lượt

Chạy: streamlit run app.py
"""
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time
from datetime import datetime

from config import (
    VEHICLE_CLASSES, MODEL_PATH, CONFIDENCE_THRESHOLD,
    COUNT_LINE_Y_RATIO, COUNT_LINE_COLOR, COUNT_LINE_THICKNESS,
    APP_TITLE, APP_ICON, PAGE_LAYOUT
)
from line_counter import LineCounter
from export_utils import save_count_to_csv, export_to_excel, get_statistics
from evaluate import FPSCounter

# ====== PAGE CONFIG ======
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout=PAGE_LAYOUT)


@st.cache_resource
def load_model(model_path=MODEL_PATH):
    """Tải model YOLOv8 (cache để không tải lại)."""
    return YOLO(model_path)


def detect_vehicles(model, frame, conf=CONFIDENCE_THRESHOLD):
    """Nhận diện phương tiện trong frame."""
    results = model(frame, conf=conf, verbose=False)[0]
    detections = []
    frame_count = {}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            info = VEHICLE_CLASSES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_val = float(box.conf[0])
            name = info["name"]
            frame_count[name] = frame_count.get(name, 0) + 1
            detections.append({
                "bbox": (x1, y1, x2, y2),
                "class_name": name,
                "class_id": cls_id,
                "confidence": conf_val,
                "color": info["color"],
            })
    return detections, frame_count


def draw_detections(frame, detections):
    """Vẽ bounding box và nhãn lên frame."""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = det["color"]
        label = f'{det["class_name"]} {det["confidence"]:.0%}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def draw_count_line(frame, line_y):
    """Vẽ vạch đếm lên frame."""
    h, w = frame.shape[:2]
    cv2.line(frame, (0, line_y), (w, line_y), COUNT_LINE_COLOR, COUNT_LINE_THICKNESS)
    cv2.putText(frame, "VACH DEM", (10, line_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COUNT_LINE_COLOR, 2)
    return frame


# ====== MAIN APP ======
st.title(APP_TITLE)
st.caption("**Khóa luận tốt nghiệp** — Lớp CDLT422CNTT — GVHD: Cô Nguyễn Thị Lượt")
st.divider()

# ====== SIDEBAR ======
with st.sidebar:
    st.header("⚙️ Cài đặt")
    source = st.radio("📹 Nguồn video:", [
        "📱 Camera (DroidCam)",
        "🎥 Upload Video",
        "🖼️ Upload Ảnh"
    ], index=2)

    conf_threshold = st.slider("🎯 Ngưỡng tin cậy", 0.1, 0.9, CONFIDENCE_THRESHOLD, 0.05)
    enable_line_count = st.checkbox("📍 Bật đếm qua vạch", value=True)

    if enable_line_count:
        line_ratio = st.slider("📏 Vị trí vạch đếm (%)", 20, 80, int(COUNT_LINE_Y_RATIO * 100))
    else:
        line_ratio = 60

    st.divider()
    st.header("📊 Loại xe nhận diện")
    for cls_id, info in VEHICLE_CLASSES.items():
        st.markdown(f"- {info['name']} ({info['name_en']})")

# Load model
model = load_model()

# ====== UPLOAD ẢNH ======
if source == "🖼️ Upload Ảnh":
    uploaded = st.file_uploader("Upload ảnh giao thông", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded)
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detections, frame_count = detect_vehicles(model, frame, conf_threshold)
        frame = draw_detections(frame, detections)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(frame_rgb, caption="Kết quả nhận diện", use_container_width=True)
        with col2:
            total = sum(frame_count.values())
            st.metric("🚗 Tổng phương tiện", total)
            st.divider()
            for name, num in frame_count.items():
                if num > 0:
                    st.metric(name, num)

# ====== UPLOAD VIDEO ======
elif source == "🎥 Upload Video":
    uploaded = st.file_uploader("Upload video giao thông", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = int(h * line_ratio / 100)
        counter = LineCounter(line_y)
        fps_counter = FPSCounter()

        stframe = st.empty()
        col1, col2, col3 = st.columns(3)
        total_ph = col1.empty()
        class_ph = col2.empty()
        fps_ph = col3.empty()
        stop = st.button("⏹️ Dừng")

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            detections, frame_count = detect_vehicles(model, frame, conf_threshold)
            frame = draw_detections(frame, detections)

            if enable_line_count:
                result = counter.update(detections)
                frame = draw_count_line(frame, line_y)
                total_ph.metric("🔢 Xe qua vạch", result["total"])
                class_info = " | ".join(f"{k}: {v}" for k, v in result["by_class"].items())
                class_ph.markdown(f"**{class_info}**" if class_info else "—")
            else:
                total = sum(frame_count.values())
                total_ph.metric("🚗 Xe trong frame", total)

            fps = fps_counter.update()
            fps_ph.metric("⚡ FPS", f"{fps:.1f}")
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        cap.release()

        # Lưu kết quả
        if enable_line_count:
            count_data = {"total": counter.total_count, "by_class": counter.count_by_class}
            csv_path = save_count_to_csv(count_data)
            st.success(f"✅ Đã lưu kết quả: {csv_path}")
            if st.button("📊 Xuất Excel"):
                excel_path = export_to_excel()
                st.success(f"✅ Đã xuất: {excel_path}")

# ====== CAMERA ======
elif source == "📱 Camera (DroidCam)":
    st.info("📱 Kết nối điện thoại qua DroidCam để nhận diện real-time.")
    cam_method = st.radio("Kết nối:", ["DroidCam Client", "IP Camera"])

    if cam_method == "IP Camera":
        ip_url = st.text_input("URL:", "http://192.168.1.x:4747/video")

    col1, col2 = st.columns(2)
    start = col1.button("▶️ Bắt đầu", type="primary")
    stop = col2.button("⏹️ Dừng")

    if start:
        cap = cv2.VideoCapture(ip_url if cam_method == "IP Camera" else 0)
        if not cap.isOpened():
            st.error("❌ Không kết nối được camera!")
        else:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            line_y = int(h * line_ratio / 100)
            counter = LineCounter(line_y)
            fps_counter = FPSCounter()

            stframe = st.empty()
            col1, col2, col3 = st.columns(3)
            total_ph = col1.empty()
            class_ph = col2.empty()
            fps_ph = col3.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections, frame_count = detect_vehicles(model, frame, conf_threshold)
                frame = draw_detections(frame, detections)

                if enable_line_count:
                    result = counter.update(detections)
                    frame = draw_count_line(frame, line_y)
                    total_ph.metric("🔢 Xe qua vạch", result["total"])
                    info = " | ".join(f"{k}: {v}" for k, v in result["by_class"].items())
                    class_ph.markdown(f"**{info}**" if info else "—")
                else:
                    total_ph.metric("🚗 Xe trong frame", sum(frame_count.values()))

                fps = fps_counter.update()
                fps_ph.metric("⚡ FPS", f"{fps:.1f}")
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                time.sleep(0.03)

            cap.release()

# ====== FOOTER ======
st.divider()
stats = get_statistics()
if stats:
    st.subheader("📊 Lịch sử thống kê")
    st.dataframe(stats["dataframe"], use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Tổng bản ghi", stats["total_records"])
    col2.metric("Tổng xe đã đếm", stats["total_vehicles"])
    col3.metric("TB xe/lần", stats["avg_per_record"])

st.divider()
st.caption("📚 **KLTN:** Phát triển hệ thống nhận diện giao thông — GVHD: Cô Nguyễn Thị Lượt")
