"""
🚗 Code mẫu đếm xe cơ bản — Phiên bản đơn giản
KLTN — Lớp CDLT422CNTT — GVHD: Cô Nguyễn Thị Lượt

Cách dùng:
  1. Cài DroidCam trên điện thoại + laptop
  2. Kết nối cùng WiFi
  3. Chạy: python count_vehicles.py
  4. Nhấn Q để thoát
"""
from ultralytics import YOLO
import cv2
from config import VEHICLE_CLASSES, MODEL_PATH, CONFIDENCE_THRESHOLD, CAMERA_SOURCE

# ===== LOAD MODEL =====
model = YOLO(MODEL_PATH)  # Tự động tải lần đầu
print(f"✅ Đã tải model: {MODEL_PATH}")

# ===== KẾT NỐI CAMERA =====
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print("❌ Không thể kết nối camera!")
    print("   → Kiểm tra DroidCam hoặc đổi CAMERA_SOURCE trong config.py")
    exit()
print(f"✅ Đã kết nối camera: {CAMERA_SOURCE}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện phương tiện
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

    # Đếm và vẽ bounding box
    count = {info["name"]: 0 for info in VEHICLE_CLASSES.values()}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            info = VEHICLE_CLASSES[cls_id]
            name = info["name"]
            color = info["color"]
            count[name] += 1

            # Vẽ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Vẽ nhãn
            label = f"{name} {conf:.0%}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Hiển thị tổng đếm
    total = sum(count.values())
    info_text = f"Tong: {total}"
    for name, num in count.items():
        if num > 0:
            info_text += f" | {name}: {num}"
    cv2.putText(frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Hiển thị
    cv2.imshow("KLTN - Dem xe (Q de thoat)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Đã thoát chương trình.")
