"""
📈 Module đánh giá hệ thống
KLTN — Lớp CDLT422CNTT — GVHD: Cô Nguyễn Thị Lượt

Đánh giá:
  - FPS (Frames Per Second)
  - Độ chính xác nhận diện
  - So sánh hiệu năng các model YOLOv8
"""
import time
import cv2
from ultralytics import YOLO
from config import VEHICLE_CLASSES, CONFIDENCE_THRESHOLD


class FPSCounter:
    """Đo FPS real-time."""
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def update(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()
        return self.fps

    def get_fps(self):
        return round(self.fps, 1)


def benchmark_model(model_path, video_source, num_frames=100):
    """
    Đo hiệu năng model trên video.

    Args:
        model_path: đường dẫn model (vd: "yolov8n.pt")
        video_source: nguồn video (file path hoặc camera index)
        num_frames: số frame để đo

    Returns:
        dict: kết quả benchmark
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        return {"error": f"Không mở được video: {video_source}"}

    total_time = 0
    total_detections = 0
    frame_count = 0

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
        elapsed = time.time() - start

        total_time += elapsed
        vehicle_count = sum(1 for b in results.boxes if int(b.cls[0]) in VEHICLE_CLASSES)
        total_detections += vehicle_count
        frame_count += 1

    cap.release()

    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_inference = (total_time / frame_count * 1000) if frame_count > 0 else 0

    return {
        "model": model_path,
        "frames_tested": frame_count,
        "avg_fps": round(avg_fps, 1),
        "avg_inference_ms": round(avg_inference, 1),
        "total_detections": total_detections,
        "avg_detections_per_frame": round(total_detections / max(frame_count, 1), 1),
    }


def compare_models(video_source, num_frames=100):
    """
    So sánh hiệu năng 3 model YOLOv8 (nano, small, medium).

    Returns:
        list[dict]: kết quả benchmark của từng model
    """
    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    results = []

    for model_path in models:
        print(f"🔄 Đang benchmark: {model_path}...")
        result = benchmark_model(model_path, video_source, num_frames)
        results.append(result)
        print(f"   ✅ FPS: {result.get('avg_fps', 0)} | "
              f"Inference: {result.get('avg_inference_ms', 0)}ms | "
              f"Detections: {result.get('avg_detections_per_frame', 0)}/frame")

    return results


if __name__ == "__main__":
    import sys
    video = sys.argv[1] if len(sys.argv) > 1 else 0
    print("📈 BẮT ĐẦU ĐÁNH GIÁ HỆ THỐNG")
    print("=" * 50)
    results = compare_models(video, num_frames=50)
    print("\n📊 KẾT QUẢ SO SÁNH:")
    print(f"{'Model':<15} {'FPS':<10} {'Inference':<15} {'Detections/frame':<20}")
    print("-" * 60)
    for r in results:
        if "error" not in r:
            print(f"{r['model']:<15} {r['avg_fps']:<10} {r['avg_inference_ms']}ms{'':<8} {r['avg_detections_per_frame']:<20}")
