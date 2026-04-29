"""
🚗 Đếm xe real-time qua điện thoại — Code mẫu
Cuộc thi: "Phát triển hệ thống nhận diện giao thông" — BETU 2026

Cách dùng:
  1. Cài app DroidCam trên điện thoại (Android/iOS)
  2. Cài DroidCam Client trên laptop (dev47apps.com)
  3. Kết nối cùng WiFi → mở DroidCam → ghi nhớ IP
  4. Chạy: python count_vehicles.py
  5. Nhấn Q để thoát
"""
from ultralytics import YOLO
import cv2

# ===== CẤU HÌNH CAMERA =====
# Cách 1: DroidCam (cài DroidCam Client → điện thoại kết nối laptop)
cap = cv2.VideoCapture(0)

# Cách 2: IP Camera (dùng app DroidCam trên điện thoại)
# cap = cv2.VideoCapture("http://192.168.1.x:8080/video")

# Cách 3: DroidCam qua IP trực tiếp
# cap = cv2.VideoCapture("http://192.168.1.x:4747/video")

# ===== LOAD MODEL =====
model = YOLO("yolov8n.pt")  # Tự động tải lần đầu (~6MB)

# Các class xe trong COCO dataset
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 1: "bicycle"}
COLORS = {2: (46,204,113), 3: (231,76,60), 5: (52,152,219), 7: (243,156,18), 1: (155,89,182)}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Nhận diện phương tiện
    results = model(frame, verbose=False)[0]

    # Đếm và vẽ bounding box
    count = {name: 0 for name in VEHICLES.values()}
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLES:
            name = VEHICLES[cls_id]
            count[name] += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            color = COLORS.get(cls_id, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf:.0%}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Hiển thị số đếm
    total = sum(count.values())
    info = f"Total: {total} | " + " | ".join(f"{k}: {v}" for k, v in count.items() if v > 0)
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Vehicle Counter - BETU 2026 (Q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
