# 🚦 Phát triển hệ thống nhận diện giao thông

### Khóa luận tốt nghiệp — Lớp CDLT422CNTT — Trường Cao đẳng Công nghệ cao Đồng An

| Thông tin | Chi tiết |
|-----------|----------|
| 📝 Đề tài | Phát triển hệ thống nhận diện giao thông |
| 👩‍🏫 GVHD | Cô Nguyễn Thị Lượt |
| 👥 Nhóm | Đào Hoàng Hải, Lê Đặng Đức Trung, Phạm Đăng Khoa, Phạm Đức Hào |

---

## 📌 Chủ Đề

> **"Phát triển hệ thống nhận diện giao thông"**
>
> Sản phẩm **bắt buộc phải kết nối camera** (dùng **điện thoại cá nhân** qua app DroidCam), có thuật toán để **đếm số lượng xe** và **nhận diện hình ảnh các loại xe**.

---

## 📱 CÁCH HOẠT ĐỘNG

> Điện thoại = **camera quay phim** (DroidCam) → Laptop = **bộ não AI** (YOLOv8) → Kết quả hiển thị trên màn hình

```
📱 Điện thoại                💻 Laptop
┌──────────────────┐  WiFi  ┌───────────────────────────┐
│  App DroidCam     │──────►│ Phần mềm SV viết          │
│  (chỉ quay video) │       │ → Nhận hình từ điện thoại  │
│                   │       │ → YOLOv8 detect xe         │
│  Chĩa camera     │       │ → Đếm xe, hiển thị kết quả │
│  ra ngoài đường   │       │ → Lưu thống kê CSV/Excel   │
└──────────────────┘        └───────────────────────────┘
```

**Điện thoại nào cũng làm được** — chỉ cần có camera + WiFi. Không cần mua thêm thiết bị!

---

## 📁 Cấu Trúc Dự Án

```
KLTN-Traffic-Detection/
├── count_vehicles.py       ← ⭐ Code mẫu đếm xe qua điện thoại (~65 dòng)
├── Demo_App.py             ← Ứng dụng Streamlit đầy đủ (DroidCam + video + ảnh)
├── Train_YOLOv8_Bien_Bao.ipynb  ← Notebook train YOLOv8 trên Google Colab
├── requirements.txt        ← Thư viện cần cài
└── README.md               ← File hướng dẫn này
```

---

## 🚀 Bắt Đầu Nhanh

### 1. Cài thư viện
```bash
pip install -r requirements.txt
```

### 2. Cài DroidCam
- 📱 Điện thoại: Tải **DroidCam** từ Google Play / App Store
- 💻 Laptop: Tải **DroidCam Client** từ [dev47apps.com](https://dev47apps.com)
- Kết nối cùng WiFi → mở DroidCam → laptop nhận camera

### 3. Chạy code mẫu
```bash
python count_vehicles.py
```
→ Cầm điện thoại chĩa ra đường → laptop hiện bounding box + đếm xe. Nhấn `Q` để thoát.

### 4. Chạy ứng dụng Streamlit
```bash
streamlit run Demo_App.py
```
→ Mở `http://localhost:8501` → Chọn **Camera Điện thoại** / Video / Ảnh → Xem kết quả.

---

## 🛠️ HƯỚNG DẪN KỸ THUẬT

### A. Cài đặt DroidCam (biến điện thoại thành camera)

| Bước | Điện thoại 📱 | Laptop 💻 |
|------|-------------|-----------|
| 1 | Tải **DroidCam** từ Store | Tải **DroidCam Client** từ [dev47apps.com](https://dev47apps.com) |
| 2 | Mở app → cấp quyền Camera | Cài đặt → mở app |
| 3 | Ghi nhớ **WiFi IP** hiển thị | Nhập IP → nhấn **Start** |
| 4 | ✅ Điện thoại = camera cho laptop | ✅ Laptop nhận hình từ điện thoại |

> 💡 Cũng có thể kết nối qua **cáp USB** để hình ảnh mượt hơn WiFi.

### B. YOLOv8 — Nhận diện phương tiện

YOLOv8 pretrained trên COCO **đã nhận diện sẵn** xe, không cần train:

| Class ID | Tên | Ghi chú |
|----------|-----|---------|
| 1 | bicycle | Xe đạp |
| 2 | car | Ô tô |
| 3 | motorcycle | Xe máy |
| 5 | bus | Xe buýt |
| 7 | truck | Xe tải |

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # Tự động tải lần đầu (~6MB)
results = model("traffic.jpg")
results[0].show()
```

### C. Kết hợp Điện thoại + YOLOv8 = Đếm xe

```python
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}

# Cách 1: DroidCam Client (điện thoại kết nối laptop)
cap = cv2.VideoCapture(0)

# Cách 2: IP Camera
# cap = cv2.VideoCapture("http://192.168.1.x:4747/video")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)[0]
    count = {}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLES:
            name = VEHICLES[cls_id]
            count[name] = count.get(name, 0) + 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {float(box.conf[0]):.0%}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame, f"Total: {sum(count.values())}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Vehicle Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
```

### D. Streamlit — Tạo giao diện web

```python
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("yolov8n.pt")
st.title("🚦 Nhận diện giao thông")
uploaded = st.file_uploader("Upload ảnh", type=["jpg", "png"])

if uploaded:
    image = Image.open(uploaded)
    results = model(np.array(image))[0]
    st.image(results.plot(), caption="Kết quả", use_container_width=True)
```

### E. So sánh Model YOLOv8

| Model | Kích thước | Tốc độ | Chính xác | Gợi ý |
|-------|-----------|--------|-----------|-------|
| `yolov8n.pt` | 6 MB | ~45 FPS | ⭐⭐⭐ | Laptop yếu, phát triển |
| `yolov8s.pt` | 22 MB | ~35 FPS | ⭐⭐⭐⭐ | Bản nộp |
| `yolov8m.pt` | 50 MB | ~20 FPS | ⭐⭐⭐⭐⭐ | Laptop mạnh |

---

## 🎯 YÊU CẦU KLTN (theo đề tài Cô Nguyễn Thị Lượt)

### Yêu cầu bắt buộc
- 📱 **Kết nối camera** điện thoại qua DroidCam
- 🔢 **Đếm số lượng** xe theo từng loại (tối thiểu 3 loại)
- 📍 **Đếm xe qua vạch** (line counting) — đếm không trùng
- 🖥️ **Giao diện** hiển thị bounding box, số đếm, loại xe
- 💾 **Lưu trữ & thống kê** — xuất file CSV / Excel
- 📈 **Đánh giá hệ thống** — độ chính xác, FPS

### Gợi ý phát triển thêm
| Tính năng | Mô tả |
|-----------|-------|
| 📊 Biểu đồ real-time | Chart số xe theo thời gian |
| 🔔 Cảnh báo | Thông báo khi mật độ giao thông cao |
| 📹 Ghi video | Lưu lại video đã detect |
| 🆔 Tracking xe | Gán ID từng xe (SORT/DeepSORT) |
| 🗺️ Heatmap | Bản đồ nhiệt mật độ giao thông |

---

## 👥 Phân Công Nhóm (gợi ý)

| Thành viên | Nhiệm vụ chính |
|-----------|----------------|
| Đào Hoàng Hải | Kết nối camera + nhận diện phương tiện |
| Lê Đặng Đức Trung | Giao diện Streamlit + hiển thị kết quả |
| Phạm Đăng Khoa | Đếm xe qua vạch + lưu trữ CSV/Excel |
| Phạm Đức Hào | Đánh giá hệ thống + viết báo cáo KLTN |

---

## 📎 Tài Nguyên

| Tài nguyên | Link |
|-----------|------|
| YOLOv8 Docs | [docs.ultralytics.com](https://docs.ultralytics.com) |
| DroidCam | [dev47apps.com](https://dev47apps.com) |
| OpenCV | [opencv.org](https://opencv.org) |
| Streamlit | [docs.streamlit.io](https://docs.streamlit.io) |
| SORT Tracking | [github.com/abewley/sort](https://github.com/abewley/sort) |
| VisDrone Dataset | [github.com/VisDrone](https://github.com/VisDrone/VisDrone-Dataset) |

---

## 📄 Thông Tin KLTN

- **Đề tài:** Phát triển hệ thống nhận diện giao thông
- **GVHD:** Cô Nguyễn Thị Lượt
- **Lớp:** CDLT422CNTT
- **Trường:** Cao đẳng Công nghệ cao Đồng An
- **Năm:** 2026
