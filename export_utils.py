"""
📊 Module xuất báo cáo thống kê (CSV / Excel)
KLTN — Lớp CDLT422CNTT — GVHD: Cô Nguyễn Thị Lượt
"""
import os, csv
from datetime import datetime
import pandas as pd
from config import RESULTS_DIR, CSV_FILENAME, EXCEL_FILENAME


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_count_to_csv(count_data, filename=None):
    """Lưu dữ liệu đếm vào file CSV."""
    ensure_results_dir()
    filepath = os.path.join(RESULTS_DIR, filename or CSV_FILENAME)
    file_exists = os.path.exists(filepath)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            headers = ["Thời gian", "Tổng xe"] + list(count_data.get("by_class", {}).keys())
            writer.writerow(headers)
        row = [timestamp, count_data.get("total", 0)] + list(count_data.get("by_class", {}).values())
        writer.writerow(row)
    return filepath


def export_to_excel(csv_filepath=None, excel_filepath=None):
    """Chuyển CSV sang Excel có format."""
    ensure_results_dir()
    csv_filepath = csv_filepath or os.path.join(RESULTS_DIR, CSV_FILENAME)
    excel_filepath = excel_filepath or os.path.join(RESULTS_DIR, EXCEL_FILENAME)

    if not os.path.exists(csv_filepath):
        return None

    df = pd.read_csv(csv_filepath, encoding="utf-8")
    with pd.ExcelWriter(excel_filepath, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Chi tiết", index=False)
        if len(df) > 0 and "Tổng xe" in df.columns:
            summary = pd.DataFrame({
                "Chỉ số": ["Tổng bản ghi", "Tổng xe", "TB xe/lần", "Cao nhất", "Thấp nhất"],
                "Giá trị": [len(df), df["Tổng xe"].sum(), round(df["Tổng xe"].mean(), 1),
                            df["Tổng xe"].max(), df["Tổng xe"].min()]
            })
            summary.to_excel(writer, sheet_name="Tổng hợp", index=False)
    return excel_filepath


def get_statistics(csv_filepath=None):
    """Đọc và tính thống kê từ CSV."""
    csv_filepath = csv_filepath or os.path.join(RESULTS_DIR, CSV_FILENAME)
    if not os.path.exists(csv_filepath):
        return None
    df = pd.read_csv(csv_filepath, encoding="utf-8")
    if len(df) == 0:
        return None
    return {
        "total_records": len(df),
        "total_vehicles": int(df["Tổng xe"].sum()) if "Tổng xe" in df.columns else 0,
        "avg_per_record": round(df["Tổng xe"].mean(), 1) if "Tổng xe" in df.columns else 0,
        "dataframe": df,
    }


def clear_results():
    """Xóa toàn bộ file kết quả."""
    for fn in [CSV_FILENAME, EXCEL_FILENAME]:
        fp = os.path.join(RESULTS_DIR, fn)
        if os.path.exists(fp):
            os.remove(fp)
