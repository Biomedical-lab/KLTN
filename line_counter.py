"""
📍 Module đếm xe qua vạch (Line Counting)
KLTN — Lớp CDLT422CNTT — GVHD: Cô Nguyễn Thị Lượt

Module này thực hiện:
  - Tracking đối tượng bằng centroid
  - Đếm xe khi đi qua vạch đếm
  - Tránh đếm trùng lặp
"""
import numpy as np
from collections import OrderedDict
from scipy.spatial import distance as dist


class CentroidTracker:
    """
    Theo dõi đối tượng dựa trên tâm (centroid).
    Gán ID duy nhất cho mỗi xe để tránh đếm trùng.
    """

    def __init__(self, max_disappeared=30, max_distance=80):
        """
        Args:
            max_disappeared: Số frame tối đa trước khi xóa đối tượng
            max_distance: Khoảng cách tối đa (pixel) để match 2 centroid
        """
        self.next_id = 0
        self.objects = OrderedDict()       # {ID: centroid}
        self.disappeared = OrderedDict()   # {ID: số frame đã mất}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        """Đăng ký đối tượng mới"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        """Xóa đối tượng"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, bboxes):
        """
        Cập nhật tracker với danh sách bounding box mới.

        Args:
            bboxes: list of (x1, y1, x2, y2) - bounding boxes

        Returns:
            dict: {object_id: centroid}
        """
        # Nếu không có bounding box
        if len(bboxes) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        # Tính centroid cho mỗi bbox
        input_centroids = np.zeros((len(bboxes), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids[i] = (cx, cy)

        # Nếu chưa có đối tượng nào → đăng ký tất cả
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Match centroid hiện tại với centroid mới
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = dist.cdist(np.array(object_centroids), input_centroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Xử lý đối tượng chưa match
            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects


class LineCounter:
    """
    Đếm xe khi đi qua vạch đếm (counting line).
    """

    def __init__(self, line_y, crossing_threshold=15):
        """
        Args:
            line_y: Tọa độ Y của vạch đếm (pixel)
            crossing_threshold: Khoảng cách (pixel) để xác nhận xe đã qua vạch
        """
        self.line_y = line_y
        self.crossing_threshold = crossing_threshold
        self.tracker = CentroidTracker()
        self.counted_ids = set()           # Các ID đã đếm (tránh trùng)
        self.previous_centroids = {}       # Vị trí trước đó của mỗi ID
        self.total_count = 0               # Tổng số xe đã đếm
        self.count_by_class = {}           # Đếm theo loại xe

    def update(self, detections):
        """
        Cập nhật bộ đếm với danh sách detection mới.

        Args:
            detections: list of dict, mỗi dict gồm:
                - "bbox": (x1, y1, x2, y2)
                - "class_name": tên loại xe
                - "class_id": ID class
                - "confidence": độ tin cậy

        Returns:
            dict: Kết quả đếm
                - "total": tổng xe đã qua vạch
                - "by_class": {tên_xe: số_lượng}
                - "objects": {id: centroid} - đối tượng đang tracking
                - "new_crosses": list of dict - xe vừa qua vạch frame này
        """
        # Lấy bounding boxes
        bboxes = [det["bbox"] for det in detections]
        class_map = {}
        for det in detections:
            cx = int((det["bbox"][0] + det["bbox"][2]) / 2)
            cy = int((det["bbox"][1] + det["bbox"][3]) / 2)
            class_map[(cx, cy)] = det["class_name"]

        # Cập nhật tracker
        objects = self.tracker.update(bboxes)

        new_crosses = []

        # Kiểm tra xe qua vạch
        for obj_id, centroid in objects.items():
            cx, cy = centroid

            if obj_id in self.counted_ids:
                self.previous_centroids[obj_id] = cy
                continue

            if obj_id in self.previous_centroids:
                prev_y = self.previous_centroids[obj_id]

                # Xe đi từ trên xuống qua vạch
                if prev_y < self.line_y and cy >= self.line_y:
                    self._count_vehicle(obj_id, centroid, class_map)
                    new_crosses.append({
                        "id": obj_id,
                        "centroid": centroid,
                        "direction": "down"
                    })

                # Xe đi từ dưới lên qua vạch
                elif prev_y > self.line_y and cy <= self.line_y:
                    self._count_vehicle(obj_id, centroid, class_map)
                    new_crosses.append({
                        "id": obj_id,
                        "centroid": centroid,
                        "direction": "up"
                    })

            self.previous_centroids[obj_id] = cy

        return {
            "total": self.total_count,
            "by_class": dict(self.count_by_class),
            "objects": objects,
            "new_crosses": new_crosses,
        }

    def _count_vehicle(self, obj_id, centroid, class_map):
        """Ghi nhận 1 xe đã qua vạch"""
        self.counted_ids.add(obj_id)
        self.total_count += 1

        # Tìm class gần nhất
        cx, cy = centroid
        best_class = "Khác"
        min_dist = float("inf")
        for (c_cx, c_cy), class_name in class_map.items():
            d = abs(cx - c_cx) + abs(cy - c_cy)
            if d < min_dist:
                min_dist = d
                best_class = class_name

        self.count_by_class[best_class] = self.count_by_class.get(best_class, 0) + 1

    def reset(self):
        """Reset bộ đếm"""
        self.counted_ids.clear()
        self.previous_centroids.clear()
        self.total_count = 0
        self.count_by_class.clear()

    def set_line_y(self, line_y):
        """Đặt lại vị trí vạch đếm"""
        self.line_y = line_y
