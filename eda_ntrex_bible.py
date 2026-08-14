import json
import numpy as np
import pandas as pd
from pathlib import Path
import tiktoken
from tqdm import tqdm

# ---------------------------------------------------------
# 1. THIẾT LẬP ĐƯỜNG DẪN
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed_alignment"
EDA_DIR = SCRIPT_DIR / "EDA"

# Định nghĩa các bộ dữ liệu cần chạy
DATASETS = [
    {
        "name": "NTREX-128",
        "input_path": PROCESSED_DIR / "ntrex.json",
        "out_dir": EDA_DIR / "ntrex"
    },
    {
        "name": "Bible",
        "input_path": PROCESSED_DIR / "bible.json",
        "out_dir": EDA_DIR / "bible"
    }
]

# ---------------------------------------------------------
# 2. HÀM CHUNG CHO EDA (Tái sử dụng)
# ---------------------------------------------------------
def run_eda(dataset_name: str, input_path: Path, out_dir: Path):
    print(f"\n{'='*50}")
    print(f"=== BẮT ĐẦU EDA CHO {dataset_name.upper()} ===")
    print(f"{'='*50}")
    
    # Đảm bảo thư mục output tồn tại
    out_dir.mkdir(parents=True, exist_ok=True)

    # Đọc dữ liệu
    if not input_path.exists():
        print(f"[!] Không tìm thấy file: {input_path} -> Bỏ qua.")
        return
        
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    total_samples = len(records)
    print(f"-> Đã load {total_samples} bản ghi từ {input_path.name}")

    # Lấy danh sách ngôn ngữ (loại bỏ cột 'id')
    all_keys = set()
    for rec in records:
        all_keys.update(rec.keys())
    languages = sorted([k for k in all_keys if k != "id"])
    total_langs = len(languages)

    # =========================================================
    # TASK 1: TÍNH TOÀN VẸN DỮ LIỆU & SỐ LƯỢNG NGÔN NGỮ (.txt)
    # =========================================================
    missing_info = []
    for i, rec in enumerate(records):
        # Kiểm tra xem ngôn ngữ có bị thiếu hoặc chuỗi rỗng không
        missing_langs = [lang for lang in languages if lang not in rec or not str(rec[lang]).strip()]
        if missing_langs:
            missing_info.append(f"Row ID {rec.get('id', i)} missing: {', '.join(missing_langs)}")

    integrity_path = out_dir / "integrity_report.txt"
    with open(integrity_path, "w", encoding="utf-8") as f:
        f.write(f"=== BÁO CÁO TOÀN VẸN DỮ LIỆU {dataset_name.upper()} ===\n")
        f.write(f"1. Tổng số ngôn ngữ (cột): {total_langs}\n")
        f.write(f"2. Tổng số bản ghi (hàng): {total_samples}\n")
        f.write("-" * 40 + "\n")
        if not missing_info:
            f.write("KẾT LUẬN: Dữ liệu SẠCH 100%. Không có ID nào bị khuyết ngôn ngữ.\n")
        else:
            f.write(f"KẾT LUẬN: Phát hiện {len(missing_info)} dòng bị khuyết dữ liệu:\n")
            for info in missing_info[:100]:  # In tối đa 100 dòng để file không bị quá nặng
                f.write(info + "\n")
            if len(missing_info) > 100:
                f.write(f"... và {len(missing_info) - 100} dòng khác.\n")
    print(f"-> Đã xuất: {integrity_path}")

    # =========================================================
    # PREPARATION CHO TASK 2 & 3: TOKENIZATION
    # =========================================================
    print(f"-> Đang đếm Token cho {dataset_name} (Sử dụng tiktoken cl100k_base)...")
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Dictionary lưu mảng số lượng token của từng ngôn ngữ
    token_counts = {lang: [] for lang in languages}
    
    for rec in tqdm(records, desc=f"Tokenizing {dataset_name}"):
        for lang in languages:
            text = str(rec.get(lang, ""))
            tokens = enc.encode(text)
            token_counts[lang].append(len(tokens))

    df_tokens = pd.DataFrame(token_counts)

    # =========================================================
    # TASK 2: PHÂN PHỐI ĐỘ DÀI TOKEN (.csv)
    # =========================================================
    stats_list = []
    for lang in languages:
        data = df_tokens[lang].values
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        # Outlier bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers_count = np.sum((data < lower_bound) | (data > upper_bound))
        
        stats_list.append({
            "Language": lang,
            "Min": np.min(data),
            "Max": np.max(data),
            "Mean": round(np.mean(data), 2),
            "Median": np.median(data),
            "Q1": q1,
            "Q3": q3,
            "Outliers_Count": outliers_count
        })

    df_stats = pd.DataFrame(stats_list)
    len_dist_path = out_dir / "length_distribution.csv"
    df_stats.to_csv(len_dist_path, index=False)
    print(f"-> Đã xuất: {len_dist_path}")

    # =========================================================
    # TASK 3: TỶ LỆ NÉN (EXPANSION RATE) SO VỚI TIẾNG ANH (.csv)
    # =========================================================
    anchor_lang = "eng_Latn"
    if anchor_lang not in df_tokens.columns:
        print(f"[!] Không tìm thấy '{anchor_lang}'. Bỏ qua tính tỷ lệ nén cho {dataset_name}.")
    else:
        expansion_list = []
        # Tránh lỗi chia cho 0
        eng_tokens = np.where(df_tokens[anchor_lang].values == 0, 1, df_tokens[anchor_lang].values)
        
        for lang in languages:
            ratio_array = df_tokens[lang].values / eng_tokens
            
            mean_ratio = np.mean(ratio_array)
            median_ratio = np.median(ratio_array)
            
            expansion_list.append({
                "Language": lang,
                "Mean_Expansion_Rate": round(mean_ratio, 4),
                "Median_Expansion_Rate": round(median_ratio, 4),
                "Interpretation": f"Dài bằng {round(mean_ratio * 100, 1)}% tiếng Anh"
            })
            
        df_expansion = pd.DataFrame(expansion_list)
        df_expansion = df_expansion.sort_values(by="Mean_Expansion_Rate", ascending=False)
        
        expansion_path = out_dir / "expansion_rate.csv"
        df_expansion.to_csv(expansion_path, index=False, encoding='utf-8-sig')
        print(f"-> Đã xuất: {expansion_path}")

    print(f"=== HOÀN TẤT EDA {dataset_name.upper()} ===")


# ---------------------------------------------------------
# 3. THỰC THI CHÍNH
# ---------------------------------------------------------
if __name__ == "__main__":
    for ds in DATASETS:
        run_eda(
            dataset_name=ds["name"], 
            input_path=ds["input_path"], 
            out_dir=ds["out_dir"]
        )
