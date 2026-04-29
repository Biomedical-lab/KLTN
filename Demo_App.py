"""
🚦 Demo App — Hệ thống nhận diện & đếm phương tiện giao thông
Cuộc thi: "Phát triển hệ thống nhận diện giao thông" — BETU 2026

Chạy: streamlit run Demo_App.py
"""
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import time

# ====== CONFIG ======
VEHICLE_CLASSES = {
    2: ("Ô tô", "#2ecc71"),
    3: ("Xe máy", "#e74c3c"),
    5: ("Xe buýt", "#3498db"),
    7: ("Xe tải", "#f39c12"),
    1: ("Xe đạp", "#9b59b6"),
}

COLORS_BGR = {
    2: (46, 204, 113),
    3: (231, 76, 60),
    5: (52, 152, 219),
    7: (243, 156, 18),
    1: (155, 89, 182),
}


@st.cache_resource
def load_model():
    """Tải model YOLOv8 (cache để không tải lại)"""
    return YOLO("yolov8n.pt")


def detect_vehicles(model, frame, conf_threshold=0.3):
    """Nhận diện phương tiện trong frame"""
    results = model(frame, conf=conf_threshold, verbose=False)[0]

    detections = []
    count = {info[0]: 0 for info in VEHICLE_CLASSES.values()}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            name = VEHICLE_CLASSES[cls_id][0]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            count[name] += 1
            detections.append({
                "name": name,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "cls_id": cls_id,
            })

    return detections, count


def draw_detections(frame, detections):
    """Vẽ bounding box và nhãn lên frame"""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS_BGR.get(det["cls_id"], (0, 255, 0))
        label = f'{det["name"]} {det["conf"]:.0%}'

        # Vẽ box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Vẽ nhãn với nền
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


# ====== STREAMLIT APP ======
st.set_page_config(
    page_title="🚦 Nhận diện giao thông — BETU 2026",
    page_icon="🚗",
    layout="wide"
)

st.title("🚦 Hệ thống nhận diện & đếm phương tiện giao thông")
st.caption("Cuộc thi: **Phát triển hệ thống nhận diện giao thông** — Ngày hội CNTT BETU 2026")

st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    source = st.radio(
        "📹 Nguồn video:",
        ["📱 Camera Điện thoại (DroidCam)", "🎥 Upload Video", "🖼️ Upload Ảnh"],
        index=2
    )
    conf_threshold = st.slider("🎯 Ngưỡng tin cậy", 0.1, 0.9, 0.3, 0.05)

    st.divider()
    st.header("📱 Kết nối DroidCam")
    st.markdown("""
    1. Cài **DroidCam** trên điện thoại
    2. Cài **DroidCam Client** trên laptop
    3. Kết nối **cùng WiFi**
    4. Chọn nguồn camera ở trên
    """)

    st.divider()
    st.header("📊 Loại xe nhận diện")
    st.markdown("""
    - 🟢 Ô tô (car)
    - 🔴 Xe máy (motorcycle)
    - 🔵 Xe buýt (bus)
    - 🟡 Xe tải (truck)
    - 🟣 Xe đạp (bicycle)
    """)

# Load model
model = load_model()

# ====== XỬ LÝ THEO NGUỒN ======

if source == "🖼️ Upload Ảnh":
    uploaded = st.file_uploader("Upload ảnh giao thông", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect
        detections, count = detect_vehicles(model, frame, conf_threshold)
        frame = draw_detections(frame, detections)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.image(frame_rgb, caption="Kết quả nhận diện", use_container_width=True)
        with col2:
            total = sum(count.values())
            st.metric("🚗 Tổng phương tiện", total)
            st.divider()
            for name, num in count.items():
                if num > 0:
                    st.metric(name, num)

elif source == "🎥 Upload Video":
    uploaded = st.file_uploader("Upload video giao thông", type=["mp4", "avi", "mov"])

    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        stats_placeholder = st.empty()

        stop_btn = st.button("⏹️ Dừng")

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            detections, count = detect_vehicles(model, frame, conf_threshold)
            frame = draw_detections(frame, detections)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            stframe.image(frame_rgb, use_container_width=True)

            total = sum(count.values())
            info = f"**Tổng: {total}** | " + " | ".join(
                f"{k}: {v}" for k, v in count.items() if v > 0
            )
            stats_placeholder.markdown(info)

        cap.release()

elif source == "📱 Camera Điện thoại (DroidCam)":
    st.info("📱 Kết nối điện thoại qua DroidCam để nhận diện phương tiện real-time.")

    cam_method = st.radio("Cách kết nối:", ["DroidCam Client (camera điện thoại)", "IP Camera (nhập URL)"])

    if cam_method == "IP Camera (nhập URL)":
        ip_url = st.text_input("Nhập URL camera:", "http://192.168.1.x:4747/video")
    
    col1, col2 = st.columns(2)
    start = col1.button("▶️ Bắt đầu", type="primary")
    stop = col2.button("⏹️ Dừng")

    if start:
        if cam_method == "IP Camera (nhập URL)":
            cap = cv2.VideoCapture(ip_url)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Không thể kết nối camera. Kiểm tra DroidCam và thử lại.")
        else:
            stframe = st.empty()
            stats_placeholder = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detections, count = detect_vehicles(model, frame, conf_threshold)
                frame = draw_detections(frame, detections)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                stframe.image(frame_rgb, use_container_width=True)

                total = sum(count.values())
                info = f"**Tổng: {total}** | " + " | ".join(
                    f"{k}: {v}" for k, v in count.items() if v > 0
                )
                stats_placeholder.markdown(info)

                time.sleep(0.03)  # ~30 FPS

            cap.release()

# ====== FOOTER ======
st.divider()
st.caption("💡 **Gợi ý cải tiến:** Thêm tracking (DeepSORT), phân làn, heatmap, dashboard thống kê để được điểm cao!")
st.caption("📚 **Tham khảo:** [YOLOv8 Docs](https://docs.ultralytics.com) | [OpenCV](https://opencv.org) | [Streamlit](https://docs.streamlit.io)")
