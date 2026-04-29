# 🚦 Hệ Thống Nhận Diện & Đếm Phương Tiện Giao Thông

### Khóa luận tốt nghiệp — Lớp CDLT422CNTT — Trường CĐ Đông An

> **Đề tài:** Phát triển hệ thống nhận diện giao thông  
> **GVHD:** Cô Nguyễn Thị Lượt  
> **Nhóm sinh viên:** Đào Hoàng Hải, Lê Đặng Đức Trung, Phạm Đăng Khoa, Phạm Đức Hào

---

## 📌 Giới Thiệu

Hệ thống sử dụng **trí tuệ nhân tạo (AI)** và **thị giác máy tính (Computer Vision)** để:

- 📱 **Kết nối camera** điện thoại qua DroidCam (real-time)
- 🚗 **Nhận diện phương tiện** giao thông (xe máy, ô tô, xe tải, xe buýt, xe đạp)
- 🔢 **Đếm xe qua vạch** (line counting) — đếm chính xác, không trùng lặp
- 📊 **Thống kê & xuất báo cáo** CSV/Excel
- 🖥️ **Giao diện web** trực quan (Streamlit)

---

## 🏗️ Kiến Trúc Hệ Thống

```
📱 Điện thoại (DroidCam)          💻 Laptop (Hệ thống AI)
┌─────────────────────┐   WiFi   ┌────────────────────────────────┐
│  Camera quay video   │────────►│  1. Nhận video (OpenCV)         │
│  (DroidCam App)      │         │  2. Nhận diện xe (YOLOv8)       │
│                      │         │  3. Đếm xe qua vạch (Tracker)   │
│                      │         │  4. Hiển thị kết quả (Streamlit) │
│                      │         │  5. Lưu thống kê (CSV/Excel)    │
└─────────────────────┘          └────────────────────────────────┘
```

---

## 📁 Cấu Trúc Dự Án

```
KLTN-Traffic-Detection/
├── app.py                  ← 🖥️ Ứng dụng chính (Streamlit)
├── count_vehicles.py       ← ⭐ Code mẫu đếm xe cơ bản
├── line_counter.py         ← 📍 Module đếm xe qua vạch
├── export_utils.py         ← 📊 Module xuất báo cáo CSV/Excel
├── evaluate.py             ← 📈 Module đánh giá hệ thống
├── config.py               ← ⚙️ Cấu hình hệ thống
├── requirements.txt        ← 📦 Thư viện cần cài
├── .gitignore              ← Git ignore
├── assets/                 ← Thư mục chứa ảnh/video mẫu
│   └── sample_traffic.jpg
├── results/                ← Kết quả thống kê (tự tạo khi chạy)
│   └── (CSV files)
└── README.md               ← File hướng dẫn này
```

---

## 🚀 Hướng Dẫn Cài Đặt

### Bước 1: Clone dự án
```bash
git clone https://github.com/<username>/KLTN-Traffic-Detection.git
cd KLTN-Traffic-Detection
```

### Bước 2: Cài thư viện
```bash
pip install -r requirements.txt
```

### Bước 3: Cài DroidCam
- 📱 **Điện thoại:** Tải app **DroidCam** từ Google Play / App Store
- 💻 **Laptop:** Tải **DroidCam Client** từ [dev47apps.com](https://dev47apps.com)
- Kết nối cùng WiFi → Mở DroidCam → Ghi nhớ IP

### Bước 4: Chạy ứng dụng

**Cách 1: Chạy ứng dụng Streamlit (đầy đủ)**
```bash
streamlit run app.py
```
→ Mở trình duyệt: `http://localhost:8501`

**Cách 2: Chạy code mẫu đơn giản**
```bash
python count_vehicles.py
```
→ Nhấn `Q` để thoát

---

## 📱 Kết Nối Camera

| Cách kết nối | Lệnh |
|-------------|-------|
| DroidCam Client (USB/WiFi) | `cv2.VideoCapture(0)` |
| DroidCam IP (WiFi) | `cv2.VideoCapture("http://192.168.1.x:4747/video")` |
| Webcam laptop | `cv2.VideoCapture(0)` |
| Video file | `cv2.VideoCapture("video.mp4")` |

---

## 📊 Các Chức Năng Chính

### 1. Nhận diện phương tiện
- Sử dụng model **YOLOv8** pretrained trên COCO dataset
- Nhận diện 5+ loại xe: ô tô, xe máy, xe tải, xe buýt, xe đạp

### 2. Đếm xe qua vạch (Line Counting)
- Đặt vạch đếm trên khung hình
- Khi xe đi qua vạch → tăng bộ đếm
- Đếm theo tổng và theo từng loại xe
- Tránh đếm trùng bằng centroid tracking

### 3. Thống kê & Xuất báo cáo
- Lưu dữ liệu đếm theo thời gian thực
- Xuất file **CSV** và **Excel** (.xlsx)
- Biểu đồ thống kê trên giao diện

### 4. Đánh giá hệ thống
- Đo **FPS** (Frames Per Second)
- Đánh giá **độ chính xác** nhận diện
- So sánh hiệu năng giữa các model YOLOv8

---

## 🔧 Công Nghệ Sử Dụng

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| Python | 3.10+ | Ngôn ngữ lập trình |
| YOLOv8 (Ultralytics) | ≥ 8.0 | Model nhận diện đối tượng |
| OpenCV | ≥ 4.8 | Xử lý ảnh/video, kết nối camera |
| Streamlit | ≥ 1.30 | Giao diện web app |
| NumPy | ≥ 1.24 | Xử lý mảng, tính toán |
| Pandas | ≥ 2.0 | Xử lý dữ liệu, xuất Excel |
| Pillow | ≥ 10.0 | Xử lý ảnh |

---

## 👥 Phân Công Nhóm

| Thành viên | Nhiệm vụ chính |
|-----------|----------------|
| Đào Hoàng Hải | Module nhận diện + đếm xe |
| Lê Đặng Đức Trung | Module kết nối camera + giao diện |
| Phạm Đăng Khoa | Module thống kê + xuất báo cáo |
| Phạm Đức Hào | Đánh giá hệ thống + viết KLTN |

---

## 📎 Tài Nguyên Tham Khảo

| Tài nguyên | Link |
|-----------|------|
| YOLOv8 Docs | [docs.ultralytics.com](https://docs.ultralytics.com) |
| DroidCam | [dev47apps.com](https://dev47apps.com) |
| OpenCV | [opencv.org](https://opencv.org) |
| Streamlit | [docs.streamlit.io](https://docs.streamlit.io) |
| COCO Dataset | [cocodataset.org](https://cocodataset.org) |

---

## 📄 Giấy Phép

Dự án này được phát triển cho mục đích học thuật — Khóa luận tốt nghiệp lớp CDLT422CNTT.

**GVHD:** Cô Nguyễn Thị Lượt  
**Trường:** Cao đẳng Đông An  
**Năm:** 2026
