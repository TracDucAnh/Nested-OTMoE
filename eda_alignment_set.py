"""
eda_alignment_set.py

EDA (Exploratory Data Analysis) cơ bản cho 3 tập dữ liệu alignment:
    data/processed_alignment/bible.json
    data/processed_alignment/flores.json
    data/processed_alignment/ntrex.json

Mỗi record trong file json có dạng:
{
    "id": 0,
    "sna_Latn": "...",
    "est_Latn": "...",
    ...
}
Tất cả key trừ "id" là mã ngôn ngữ đã được chuẩn hoá trên cả 3 set.
Script KHÔNG hardcode / regex tên các key ngôn ngữ mà tự động lấy bằng set().

Chạy:
    python eda_alignment_set.py
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed_alignment"

DATASET_FILES = {
    "bible": DATA_DIR / "bible.json",
    "flores": DATA_DIR / "flores.json",
    "ntrex": DATA_DIR / "ntrex.json",
}

ID_KEY = "id"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Đọc 1 file json, trả về list record (list of dict)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # phòng trường hợp file json được bọc thêm 1 lớp {"data": [...]}
    if isinstance(data, dict):
        for key in ("data", "records", "examples"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError(f"Không parse được {path} thành list record.")

    return data


def get_langs_of_record(record: Dict[str, Any]) -> Set[str]:
    """
    Trả về set các ngôn ngữ (key) có trong 1 record, bỏ qua id và
    những giá trị rỗng / không phải string.
    """
    return {
        k for k, v in record.items()
        if k != ID_KEY and isinstance(v, str) and v.strip() != ""
    }


# ---------------------------------------------------------------------------
# Phân tích 1 dataset
# ---------------------------------------------------------------------------
def analyze_dataset(name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """EDA cho 1 dataset, trả về dict thống kê + list raw để gộp toàn cục."""
    n_records = len(records)

    all_langs: Set[str] = set()
    sentence_lengths: List[int] = []   # độ dài (số ký tự) từng câu
    n_langs_per_record: List[int] = []

    for rec in records:
        langs = get_langs_of_record(rec)
        all_langs.update(langs)
        n_langs_per_record.append(len(langs))
        for lang in langs:
            sentence_lengths.append(len(rec[lang]))

    n_sentences = len(sentence_lengths)

    return {
        "name": name,
        "n_records": n_records,
        "n_unique_langs": len(all_langs),
        "langs": all_langs,
        "n_sentences": n_sentences,
        "sentence_length": {
            "mean": (sum(sentence_lengths) / n_sentences) if n_sentences else 0.0,
            "max": max(sentence_lengths) if sentence_lengths else 0,
            "min": min(sentence_lengths) if sentence_lengths else 0,
        },
        "lang_per_record": {
            "mean": (sum(n_langs_per_record) / n_records) if n_records else 0.0,
            "max": max(n_langs_per_record) if n_langs_per_record else 0,
            "min": min(n_langs_per_record) if n_langs_per_record else 0,
        },
        # giữ lại raw list để tính thống kê gộp toàn bộ 3 set
        "_sentence_lengths": sentence_lengths,
        "_n_langs_per_record": n_langs_per_record,
    }


# ---------------------------------------------------------------------------
# In báo cáo
# ---------------------------------------------------------------------------
def print_report(per_dataset_stats: Dict[str, Dict[str, Any]]) -> None:
    sep = "=" * 72
    print(sep)
    print("EDA - ALIGNMENT DATASETS (bible / flores / ntrex)")
    print(sep)

    # ---- 1. Tổng ngôn ngữ độc nhất toàn bộ 3 set ----
    union_langs: Set[str] = set()
    for s in per_dataset_stats.values():
        union_langs |= s["langs"]

    print("\n[1] SỐ NGÔN NGỮ ĐỘC NHẤT")
    for name, s in per_dataset_stats.items():
        print(f"  - {name:8s}: {s['n_unique_langs']:4d} ngôn ngữ")
    print(f"  => Tổng ngôn ngữ độc nhất (union cả 3 set): {len(union_langs)}")

    # ---- 2. Tổng record ----
    total_records = sum(s["n_records"] for s in per_dataset_stats.values())
    print("\n[2] SỐ RECORD")
    for name, s in per_dataset_stats.items():
        print(f"  - {name:8s}: {s['n_records']:6d} record")
    print(f"  => Tổng record toàn bộ 3 set: {total_records}")

    # ---- 3. Tổng sentence ----
    total_sentences = sum(s["n_sentences"] for s in per_dataset_stats.values())
    print("\n[3] SỐ SENTENCE")
    for name, s in per_dataset_stats.items():
        print(f"  - {name:8s}: {s['n_sentences']:6d} sentence")
    print(f"  => Tổng sentence toàn bộ 3 set: {total_sentences}")

    # ---- 4. Độ dài sentence ----
    print("\n[4] ĐỘ DÀI SENTENCE (số ký tự)")
    all_lengths: List[int] = []
    for name, s in per_dataset_stats.items():
        sl = s["sentence_length"]
        print(f"  - {name:8s}: mean={sl['mean']:.2f}  max={sl['max']:6d}  min={sl['min']:6d}")
        all_lengths.extend(s["_sentence_lengths"])
    if all_lengths:
        print(
            f"  => Toàn bộ 3 set: mean={sum(all_lengths) / len(all_lengths):.2f}  "
            f"max={max(all_lengths)}  min={min(all_lengths)}"
        )

    # ---- 5. Số ngôn ngữ / record ----
    print("\n[5] SỐ NGÔN NGỮ TRÊN 1 RECORD")
    all_lang_counts: List[int] = []
    for name, s in per_dataset_stats.items():
        lr = s["lang_per_record"]
        print(f"  - {name:8s}: mean={lr['mean']:.2f}  max={lr['max']:4d}  min={lr['min']:4d}")
        all_lang_counts.extend(s["_n_langs_per_record"])
    if all_lang_counts:
        print(
            f"  => Toàn bộ 3 set: mean={sum(all_lang_counts) / len(all_lang_counts):.2f}  "
            f"max={max(all_lang_counts)}  min={min(all_lang_counts)}"
        )

    print(sep)


# ---------------------------------------------------------------------------
# Xuất bảng tổng hợp (csv) để dễ xem lại / đưa vào báo cáo khác
# ---------------------------------------------------------------------------
def build_summary_dataframe(per_dataset_stats: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, s in per_dataset_stats.items():
        rows.append({
            "dataset": name,
            "n_records": s["n_records"],
            "n_unique_langs": s["n_unique_langs"],
            "n_sentences": s["n_sentences"],
            "sent_len_mean": round(s["sentence_length"]["mean"], 2),
            "sent_len_max": s["sentence_length"]["max"],
            "sent_len_min": s["sentence_length"]["min"],
            "lang_per_record_mean": round(s["lang_per_record"]["mean"], 2),
            "lang_per_record_max": s["lang_per_record"]["max"],
            "lang_per_record_min": s["lang_per_record"]["min"],
        })

    # dòng tổng hợp toàn bộ 3 set
    all_lengths = [x for s in per_dataset_stats.values() for x in s["_sentence_lengths"]]
    all_lang_counts = [x for s in per_dataset_stats.values() for x in s["_n_langs_per_record"]]
    union_langs: Set[str] = set()
    for s in per_dataset_stats.values():
        union_langs |= s["langs"]

    rows.append({
        "dataset": "ALL (union)",
        "n_records": sum(s["n_records"] for s in per_dataset_stats.values()),
        "n_unique_langs": len(union_langs),
        "n_sentences": sum(s["n_sentences"] for s in per_dataset_stats.values()),
        "sent_len_mean": round(sum(all_lengths) / len(all_lengths), 2) if all_lengths else 0,
        "sent_len_max": max(all_lengths) if all_lengths else 0,
        "sent_len_min": min(all_lengths) if all_lengths else 0,
        "lang_per_record_mean": round(sum(all_lang_counts) / len(all_lang_counts), 2) if all_lang_counts else 0,
        "lang_per_record_max": max(all_lang_counts) if all_lang_counts else 0,
        "lang_per_record_min": min(all_lang_counts) if all_lang_counts else 0,
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    per_dataset_stats: Dict[str, Dict[str, Any]] = {}

    for name, path in DATASET_FILES.items():
        if not path.exists():
            print(f"[CẢNH BÁO] Không tìm thấy file: {path} -> bỏ qua.")
            continue
        records = load_dataset(path)
        per_dataset_stats[name] = analyze_dataset(name, records)

    if not per_dataset_stats:
        print("Không load được dataset nào, kiểm tra lại đường dẫn data/processed_alignment/.")
        return

    print_report(per_dataset_stats)

    df_summary = build_summary_dataframe(per_dataset_stats)
    out_csv = BASE_DIR / "eda_summary.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\nĐã lưu bảng tổng hợp vào: {out_csv}")


if __name__ == "__main__":
    main()