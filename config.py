"""
⚙️ Cấu hình hệ thống nhận diện giao thông
KLTN — Lớp CDLT422CNTT — GVHD: Cô Nguyễn Thị Lượt
"""

# ===== CẤU HÌNH MODEL =====
MODEL_PATH = "yolov8n.pt"       # Model YOLOv8 (nano: nhanh, nhẹ)
CONFIDENCE_THRESHOLD = 0.3      # Ngưỡng tin cậy (0.0 - 1.0)
IOU_THRESHOLD = 0.5             # Ngưỡng IoU cho NMS

# ===== PHƯƠNG TIỆN GIAO THÔNG (COCO Dataset) =====
VEHICLE_CLASSES = {
    1: {"name": "Xe đạp",   "name_en": "bicycle",    "color": (155, 89, 182)},
    2: {"name": "Ô tô",     "name_en": "car",        "color": (46, 204, 113)},
    3: {"name": "Xe máy",   "name_en": "motorcycle",  "color": (231, 76, 60)},
    5: {"name": "Xe buýt",  "name_en": "bus",         "color": (52, 152, 219)},
    7: {"name": "Xe tải",   "name_en": "truck",       "color": (243, 156, 18)},
}

# ===== CẤU HÌNH CAMERA =====
CAMERA_SOURCE = 0               # 0 = webcam/DroidCam Client
# CAMERA_SOURCE = "http://192.168.1.x:4747/video"   # DroidCam IP
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ===== CẤU HÌNH LINE COUNTING =====
# Vạch đếm mặc định (tỷ lệ % so với chiều rộng/cao khung hình)
COUNT_LINE_Y_RATIO = 0.6        # Vạch ngang ở 60% chiều cao
COUNT_LINE_COLOR = (0, 0, 255)  # Màu đỏ (BGR)
COUNT_LINE_THICKNESS = 2
CROSSING_THRESHOLD = 15         # Khoảng cách pixel để xác nhận xe đã qua vạch

# ===== CẤU HÌNH TRACKING =====
MAX_DISAPPEARED = 30            # Số frame tối đa trước khi xóa tracker
MAX_DISTANCE = 80               # Khoảng cách tối đa giữa 2 centroid (pixel)

# ===== CẤU HÌNH XUẤT DỮ LIỆU =====
RESULTS_DIR = "results"         # Thư mục lưu kết quả
CSV_FILENAME = "traffic_count.csv"
EXCEL_FILENAME = "traffic_report.xlsx"
SAVE_INTERVAL = 5               # Lưu kết quả mỗi 5 giây

# ===== CẤU HÌNH GIAO DIỆN =====
APP_TITLE = "🚦 Hệ thống nhận diện & đếm phương tiện giao thông"
APP_ICON = "🚗"
PAGE_LAYOUT = "wide"
