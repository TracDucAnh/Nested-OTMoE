import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tiktoken
from tqdm import tqdm

# ---------------------------------------------------------
# 1. THIẾT LẬP ĐƯỜNG DẪN
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
FLORES_INPUT_PATH = SCRIPT_DIR / "data" / "processed_alignment" / "ntrex.json"

# Tạo cấu trúc thư mục EDA
EDA_DIR = SCRIPT_DIR / "EDA"
FLORE_OUT_DIR = EDA_DIR / "flore"
NTREX_OUT_DIR = EDA_DIR / "ntrex"
BIBLE_OUT_DIR = EDA_DIR / "bible"

# Đảm bảo các thư mục đã tồn tại
for d in [FLORE_OUT_DIR, NTREX_OUT_DIR, BIBLE_OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. HÀM CHÍNH CHO EDA FLORES
# ---------------------------------------------------------
def eda_flores():
    print("=== BẮT ĐẦU EDA CHO FLORES-200 ===")
    
    # Đọc dữ liệu
    if not FLORES_INPUT_PATH.exists():
        print(f"[!] Không tìm thấy file: {FLORES_INPUT_PATH}")
        return
        
    with open(FLORES_INPUT_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    total_samples = len(records)
    print(f"-> Đã load {total_samples} bản ghi từ flores.json")

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
        missing_langs = [lang for lang in languages if lang not in rec or not rec[lang].strip()]
        if missing_langs:
            missing_info.append(f"Row ID {rec.get('id', i)} missing: {', '.join(missing_langs)}")

    integrity_path = FLORE_OUT_DIR / "integrity_report.txt"
    with open(integrity_path, "w", encoding="utf-8") as f:
        f.write("=== BÁO CÁO TOÀN VẸN DỮ LIỆU FLORES-200 ===\n")
        f.write(f"1. Tổng số ngôn ngữ (cột): {total_langs}\n")
        f.write(f"2. Tổng số bản ghi (hàng): {total_samples}\n")
        f.write("-" * 40 + "\n")
        if not missing_info:
            f.write("KẾT LUẬN: Dữ liệu SẠCH 100%. Không có ID nào bị khuyết ngôn ngữ.\n")
        else:
            f.write(f"KẾT LUẬN: Phát hiện {len(missing_info)} dòng bị khuyết dữ liệu:\n")
            for info in missing_info[:100]:  # In tối đa 100 dòng để tránh file quá dài
                f.write(info + "\n")
            if len(missing_info) > 100:
                f.write(f"... và {len(missing_info) - 100} dòng khác.\n")
    print(f"-> Đã xuất: {integrity_path}")

    # =========================================================
    # TASK 2: BIỂU ĐỒ TRÒN TRAIN/TEST SPLIT (.jpg)
    # =========================================================
    dev_count = sum(1 for r in records if str(r.get("id", "")).startswith("dev_"))
    devtest_count = sum(1 for r in records if str(r.get("id", "")).startswith("devtest_"))
    other_count = total_samples - dev_count - devtest_count

    labels = ['Dev (Train/Val)', 'DevTest (Test)']
    sizes = [dev_count, devtest_count]
    if other_count > 0:
        labels.append('Other')
        sizes.append(other_count)

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}%\n({int(p * sum(sizes) / 100)} samples)', 
            startangle=140, colors=['#66b3ff', '#99ff99', '#ffcc99'])
    plt.title('FLORES-200: Dev vs DevTest Split')
    
    pie_chart_path = FLORE_OUT_DIR / "split_pie_chart.jpg"
    plt.savefig(pie_chart_path, format='jpg', dpi=300)
    plt.close()
    print(f"-> Đã xuất: {pie_chart_path}")

    # =========================================================
    # PREPARATION CHO TASK 3 & 4: TOKENIZATION
    # =========================================================
    print("-> Đang đếm Token cho từng ngôn ngữ (có thể mất vài phút)...")
    # Sử dụng cl100k_base của tiktoken (Tokenizer tiêu chuẩn của GPT-4 / OpenAI)
    enc = tiktoken.get_encoding("cl100k_base")
    
    # Dictionary lưu mảng số lượng token của từng ngôn ngữ
    token_counts = {lang: [] for lang in languages}
    
    for rec in tqdm(records, desc="Tokenizing"):
        for lang in languages:
            text = rec.get(lang, "")
            # Đếm số token
            tokens = enc.encode(text)
            token_counts[lang].append(len(tokens))

    # Đưa vào DataFrame để tính toán thống kê dễ dàng hơn
    df_tokens = pd.DataFrame(token_counts)

    # =========================================================
    # TASK 3: PHÂN PHỐI ĐỘ DÀI TOKEN (.csv)
    # =========================================================
    stats_list = []
    for lang in languages:
        data = df_tokens[lang].values
        
        # Tính toán Q1, Q3, IQR
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        # Tính số điểm Outlier (Nhỏ hơn Q1 - 1.5*IQR hoặc Lớn hơn Q3 + 1.5*IQR)
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
    len_dist_path = FLORE_OUT_DIR / "length_distribution.csv"
    df_stats.to_csv(len_dist_path, index=False)
    print(f"-> Đã xuất: {len_dist_path}")

    # =========================================================
    # TASK 4: TỶ LỆ NÉN (EXPANSION RATE) SO VỚI TIẾNG ANH (.csv)
    # =========================================================
    anchor_lang = "eng_Latn"
    if anchor_lang not in df_tokens.columns:
        print(f"[!] Không tìm thấy ngôn ngữ mỏ neo '{anchor_lang}'. Bỏ qua tính tỷ lệ nén.")
    else:
        expansion_list = []
        # Tránh chia cho 0 (dù hiếm khi câu tiếng Anh có 0 token)
        eng_tokens = np.where(df_tokens[anchor_lang].values == 0, 1, df_tokens[anchor_lang].values)
        
        for lang in languages:
            # Tỷ lệ của từng câu: Token(Lang X) / Token(Eng)
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
        # Sắp xếp để xem ngôn ngữ nào "phình" to nhất, ngôn ngữ nào "nén" chặt nhất
        df_expansion = df_expansion.sort_values(by="Mean_Expansion_Rate", ascending=False)
        
        expansion_path = FLORE_OUT_DIR / "expansion_rate.csv"
        df_expansion.to_csv(expansion_path, index=False, encoding='utf-8-sig')
        print(f"-> Đã xuất: {expansion_path}")

    print("=== HOÀN TẤT EDA FLORES-200 ===")

if __name__ == "__main__":
    eda_flores()
