"""
process_data.py
================
Chuẩn hoá 3 bộ dữ liệu alignment (flores-200, ntrex-128, ted-2025) về CÙNG 1
định dạng N-way, lưu trữ theo kiểu NTREX (mỗi bản ghi = 1 dict gồm "id" +
các cột "{iso639_3}_{iso15924}": "câu dịch").

Input  (đọc từ):
    data/alignment/flores-200/dev.json
    data/alignment/flores-200/devtest.json
    data/alignment/ntrex-128/test.json
    data/alignment/ted-2025/multi_way.jsonl   (N-way sẵn, mã ngôn ngữ 2 chữ)

Output (ghi ra):
    data/processed_alignment/flores.json   (gộp dev + devtest, bỏ prefix "sentence_")
    data/processed_alignment/ntrex.json
    data/processed_alignment/ted.json      (đổi mã 2 chữ -> lang_script, id = talk_id_timestamp;
                                             CHỈ giữ lại record có số ngôn ngữ > ngưỡng
                                             TED2025_MIN_LANGS_PER_SAMPLE, mặc định = 0 tức
                                             không lọc -- xem cấu hình phía dưới. Bộ lọc này
                                             CHỈ áp dụng cho TED-2025, không áp dụng cho
                                             FLORES-200 / NTREX-128)

Cài thêm (ngoài requirements.txt gốc):
    pip install langcodes language_data unicodedataplus
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

from tqdm import tqdm
import langcodes
import unicodedataplus

# --------------------------------------------------------------------------
# ĐƯỜNG DẪN
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
ALIGNMENT_DIR = SCRIPT_DIR / "data" / "alignment"
OUT_DIR = SCRIPT_DIR / "data" / "processed_alignment"

# --------------------------------------------------------------------------
# CẤU HÌNH LỌC RIÊNG CHO TED-2025 (KHÔNG áp dụng cho FLORES-200 / NTREX-128)
# --------------------------------------------------------------------------
# FLORES-200 và NTREX-128 vốn đã có số ngôn ngữ cố định cho mọi record nên
# không cần/không áp dụng bộ lọc này. TED-2025 thì số ngôn ngữ có mặt (para
# có câu dịch không rỗng) khác nhau tuỳ record, nên có thể muốn chỉ giữ lại
# những record "phủ" nhiều ngôn ngữ.
#
# 1 record TED-2025 chỉ được GIỮ LẠI trong ted.json nếu SỐ NGÔN NGỮ có câu
# dịch không rỗng trong record đó > TED2025_MIN_LANGS_PER_SAMPLE.
# Mặc định = 0 -> không lọc gì thêm (giữ nguyên hành vi cũ: mọi record có
# >= 1 ngôn ngữ đều được giữ).
#
# Ví dụ: chỉ muốn giữ các record TED-2025 phủ nhiều ngôn ngữ (> 12 ngôn ngữ
# / record) thì đặt TED2025_MIN_LANGS_PER_SAMPLE = 12.
TED2025_MIN_LANGS_PER_SAMPLE = 0

# --------------------------------------------------------------------------
# BẢNG ÁNH XẠ TÊN SCRIPT UNICODE -> MÃ ISO 15924 (4 ký tự)
# unicodedataplus.script(ch) trả về tên script dạng dài theo chuẩn Unicode
# (vd: "Latin", "Cyrillic"...) -> cần đổi sang mã 4 ký tự như flores/ntrex
# dùng (Latn, Cyrl...).
# --------------------------------------------------------------------------
SCRIPT_NAME_TO_ISO15924 = {
    "Latin": "Latn", "Cyrillic": "Cyrl", "Greek": "Grek", "Arabic": "Arab",
    "Hebrew": "Hebr", "Han": "Hani", "Hiragana": "Hira", "Katakana": "Kana",
    "Hangul": "Hang", "Devanagari": "Deva", "Bengali": "Beng",
    "Gurmukhi": "Guru", "Gujarati": "Gujr", "Oriya": "Orya", "Tamil": "Taml",
    "Telugu": "Telu", "Kannada": "Knda", "Malayalam": "Mlym",
    "Sinhala": "Sinh", "Thai": "Thai", "Lao": "Laoo", "Tibetan": "Tibt",
    "Myanmar": "Mymr", "Georgian": "Geor", "Armenian": "Armn",
    "Ethiopic": "Ethi", "Cherokee": "Cher", "Canadian_Aboriginal": "Cans",
    "Ogham": "Ogam", "Runic": "Runr", "Khmer": "Khmr", "Mongolian": "Mong",
    "Yi": "Yiii", "Vai": "Vaii", "Bopomofo": "Bopo", "Coptic": "Copt",
    "Glagolitic": "Glag", "Thaana": "Thaa", "Nko": "Nkoo", "Syriac": "Syrc",
    "Osmanya": "Osma", "Tifinagh": "Tfng", "Balinese": "Bali",
    "Batak": "Batk", "Buginese": "Bugi", "Buhid": "Buhd", "Tagalog": "Tglg",
    "Hanunoo": "Hano", "Limbu": "Limb", "Tai_Le": "Tale",
    "New_Tai_Lue": "Talu", "Cham": "Cham", "Javanese": "Java",
    "Sundanese": "Sund", "Tai_Viet": "Tavt", "Lepcha": "Lepc",
    "Ol_Chiki": "Olck", "Meetei_Mayek": "Mtei", "Saurashtra": "Saur",
    "Kayah_Li": "Kali", "Bamum": "Bamu", "Adlam": "Adlm", "Miao": "Plrd",
}


def save_json(records, out_path: Path, desc: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  -> đã lưu {len(records)} bản ghi vào {out_path}")


# --------------------------------------------------------------------------
# 1. FLORES-200: gộp dev + devtest, bỏ prefix "sentence_"
# --------------------------------------------------------------------------
def process_flores():
    src_dir = ALIGNMENT_DIR / "flores-200"
    splits = [("dev", src_dir / "dev.json"), ("devtest", src_dir / "devtest.json")]

    results = []
    for split, path in splits:
        if not path.exists():
            print(f"  [!] không thấy {path}, bỏ qua split '{split}'.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        for rec in tqdm(records, desc=f"  flores/{split}"):
            new_rec = {"id": f"{split}_{rec.get('id')}"}
            for key, value in rec.items():
                if key.startswith("sentence_"):
                    lang_script = key[len("sentence_"):]  # sentence_vie_Latn -> vie_Latn
                    new_rec[lang_script] = value
            results.append(new_rec)

    save_json(results, OUT_DIR / "flores.json", desc="flores.json")


# --------------------------------------------------------------------------
# 2. NTREX-128: đã đúng định dạng N-way rồi, chỉ chuẩn hoá field "id"
# --------------------------------------------------------------------------
def process_ntrex():
    src_path = ALIGNMENT_DIR / "ntrex-128" / "test.json"
    if not src_path.exists():
        print(f"  [!] không thấy {src_path}, bỏ qua ntrex.")
        return
    with open(src_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    results = []
    for idx, rec in enumerate(tqdm(records, desc="  ntrex/test")):
        new_rec = {"id": rec.get("id", idx)}
        for key, value in rec.items():
            if key == "id":
                continue
            new_rec[key] = value
        results.append(new_rec)

    save_json(results, OUT_DIR / "ntrex.json", desc="ntrex.json")


# --------------------------------------------------------------------------
# 3. TED-2025: đã là N-way sẵn (mỗi dòng jsonl = 1 bản ghi, field "para_data"
#    map mã ngôn ngữ 2 chữ -> câu dịch, "lang_list" liệt kê các ngôn ngữ có
#    mặt trong bản ghi đó). Cần: (a) đổi mã 2 chữ -> "{iso639_3}_{iso15924}"
#    để thống nhất với flores/ntrex, (b) tạo "id" mới = "{talk_id}_{timestamp}".
# --------------------------------------------------------------------------
def to_iso639_3(code: str) -> str:
    """Đổi mã ngôn ngữ (2 hoặc 3 ký tự) sang ISO 639-3 bằng thư viện langcodes."""
    code = code.strip().lower()
    try:
        alpha3 = langcodes.Language.get(code).to_alpha3()
        if alpha3:
            return alpha3
    except Exception:
        pass
    # một số mã có thể đã là ISO 639-3 sẵn (3 ký tự) hoặc langcodes không
    # nhận diện được -> giữ nguyên mã gốc.
    return code


def detect_script(samples: list) -> str:
    """Phát hiện script THỰC TẾ từ nội dung câu bằng thư viện unicodedataplus
    (đếm script của từng ký tự chữ cái trong mẫu câu, lấy script chiếm đa số),
    KHÔNG suy ra script từ mã ngôn ngữ.

    LƯU Ý: thuộc tính Script của Unicode KHÔNG phân biệt được các biến thể
    dùng chung 1 script, vd tiếng Hoa giản thể/phồn thể đều ra "Han", tiếng
    Anh Mỹ/Anh-Anh/Ấn Độ đều ra "Latin". Vì vậy detect_script() không thể
    (và không có nhiệm vụ) khôi phục lại sự khác biệt vùng miền đã mất ở
    to_iso639_3() -- việc đó do to_lang_script() xử lý bằng cách giữ lại
    region subtag gốc, xem bên dưới."""
    counts = Counter()
    for text in samples:
        for ch in text:
            if not ch.isalpha():
                continue
            try:
                script_name = unicodedataplus.script(ch)
            except Exception:
                continue
            if script_name in ("Common", "Inherited", "Unknown"):
                continue
            counts[script_name] += 1

    if not counts:
        return "Latn"  # không đủ dữ liệu để phát hiện -> mặc định Latin (phổ biến nhất)

    top_script_name = counts.most_common(1)[0][0]
    return SCRIPT_NAME_TO_ISO15924.get(top_script_name, "Latn")


def to_lang_script(code: str, samples: list) -> str:
    """Đổi 1 mã ngôn ngữ TED-2025 gốc (vd 'eng', 'pt-br', 'zh-cn') sang key
    thống nhất "{iso639_3}_{script}".

    BUG ĐÃ SỬA: trước đây hàm này chỉ trả về f"{to_iso639_3(code)}_{detect_script(...)}",
    bỏ hẳn phần region/variant subtag của code gốc. Vì langcodes.to_alpha3()
    bỏ region khi rút gọn (vd 'pt-br' và 'pt-pt' đều ra "por"), và
    detect_script() cũng không phân biệt được (xem docstring ở trên), nên 2
    mã TED-2025 KHÁC NHAU (vd 'pt' và 'pt-br', hoặc 'zh-cn' và 'zh-tw') bị
    gộp về CÙNG 1 key -- khi ghi vào new_row ở process_ted2025(), key trùng
    sẽ bị ghi đè âm thầm, mất hẳn 1 ngôn ngữ và làm giảm sai tổng số ngôn ngữ
    distinct đếm được (vd TED-2025 thực có 113 ngôn ngữ nhưng ra file chỉ còn
    107, vì 6 mã có region subtag bị gộp mất).

    Cách sửa: giữ nguyên region/variant subtag gốc (nếu có) trong key, theo
    đúng quy ước mà chính NTREX-128 đang dùng cho các biến thể của nó
    (vd "por-BR_Latn", "eng-US_Latn") -> đảm bảo mỗi mã nguồn luôn ánh xạ
    ra 1 key riêng biệt, không đụng nhau.
    """
    iso3 = to_iso639_3(code)
    script = detect_script(samples)

    normalized = code.strip().lower()
    region = None
    if "-" in normalized:
        region = normalized.split("-", 1)[1].upper()
    elif "_" in normalized:
        region = normalized.split("_", 1)[1].upper()

    return f"{iso3}-{region}_{script}" if region else f"{iso3}_{script}"


def process_ted2025():
    src_path = ALIGNMENT_DIR / "ted-2025" / "multi_way.jsonl"
    if not src_path.exists():
        print(f"  [!] không thấy {src_path}, bỏ qua ted-2025.")
        return

    SAMPLE_LIMIT = 300
    samples_by_lang = defaultdict(list)
    # mỗi row: {"id": "<talk_id>_<timestamp>", "_langs": {lang_2_chữ: text}}
    rows = []
    n_records_valid = 0       # record có >= 1 ngôn ngữ (trước khi lọc theo ngưỡng)
    n_records_filtered_out = 0  # record bị loại vì không vượt ngưỡng

    with open(src_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(src_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="  ted-2025/multi_way (đọc + gộp id)"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            talk_id = rec.get("talk_id")
            timestamp = rec.get("timestamp")
            new_id = f"{talk_id}_{timestamp}"

            para_data = rec.get("para_data", {}) or {}
            lang_list = rec.get("lang_list") or list(para_data.keys())

            row_langs = {}
            for lang in lang_list:
                text = (para_data.get(lang) or "").strip()
                if not text:
                    continue
                row_langs[lang] = text
                # Mẫu câu dùng để detect script (detect_script()/to_lang_script())
                # luôn lấy từ MỌI record đọc được, KHÔNG phụ thuộc bộ lọc theo
                # ngưỡng bên dưới -- để việc detect script không bị nghèo mẫu chỉ
                # vì record đó có bị lọc bỏ khỏi ted.json hay không.
                if len(samples_by_lang[lang]) < SAMPLE_LIMIT:
                    samples_by_lang[lang].append(text)

            if not row_langs:
                continue
            n_records_valid += 1

            # BỘ LỌC THEO NGƯỠNG SỐ NGÔN NGỮ / SAMPLE -- CHỈ ÁP DỤNG CHO TED-2025
            # (process_flores() và process_ntrex() không có bước lọc tương ứng).
            # Chỉ giữ lại record nếu số ngôn ngữ (không rỗng) của nó lớn hơn
            # TED2025_MIN_LANGS_PER_SAMPLE. Mặc định ngưỡng = 0 -> mọi record có
            # >= 1 ngôn ngữ đều thoả (0 < len(row_langs)), tức không lọc gì thêm,
            # giữ nguyên hành vi cũ.
            if len(row_langs) <= TED2025_MIN_LANGS_PER_SAMPLE:
                n_records_filtered_out += 1
                continue

            rows.append({"id": new_id, "_langs": row_langs})

    print(f"  Đọc được {n_records_valid} bản ghi TED-2025 hợp lệ (có >= 1 ngôn ngữ).")
    if TED2025_MIN_LANGS_PER_SAMPLE > 0:
        print(f"  [lọc TED-2025] Ngưỡng TED2025_MIN_LANGS_PER_SAMPLE = "
              f"{TED2025_MIN_LANGS_PER_SAMPLE} -> loại {n_records_filtered_out} "
              f"bản ghi có số ngôn ngữ <= ngưỡng.")
    print(f"  Tổng cộng {len(rows)} bản ghi TED-2025 sau khi lọc.")

    print("  Đang xác định mã lang_script cho từng ngôn ngữ (langcodes + unicodedataplus)...")
    lang_code_map = {}
    target_to_source = {}  # target lang_script -> mã TED gốc đã chiếm key đó
    n_collisions = 0
    for lang, samples in tqdm(samples_by_lang.items(), desc="  Xác định lang_script"):
        target = to_lang_script(lang, samples)
        if target in target_to_source and target_to_source[target] != lang:
            # BUG ĐÃ SỬA: trước đây 2 mã TED-2025 khác nhau có thể vô tình
            # map ra cùng 1 key rồi ghi đè lên nhau ở process_ted2025() bên
            # dưới (new_row[...] = text), làm mất hẳn 1 ngôn ngữ mà không hề
            # có cảnh báo nào. Giờ luôn phát hiện + báo động rõ ràng, đồng
            # thời TỰ ĐỘNG thêm hậu tố mã gốc để đảm bảo mỗi mã nguồn luôn có
            # 1 key riêng biệt (distinct) -- không bao giờ đè mất dữ liệu.
            n_collisions += 1
            print(f"  [!] COLLISION: '{lang}' và '{target_to_source[target]}' cùng map -> "
                  f"'{target}'. Tự thêm hậu tố mã gốc để tách 2 ngôn ngữ này ra.")
            target = f"{target}-{lang.upper()}"
        target_to_source[target] = lang
        lang_code_map[lang] = target

    if n_collisions:
        print(f"  [!] Tổng cộng {n_collisions} collision đã được tự động tách ra "
              f"(xem log ở trên để biết chính xác cặp mã nào).")

    results = []
    for row in tqdm(rows, desc="  Chuẩn hoá bản ghi ted-2025"):
        new_row = {"id": row["id"]}
        for lang, text in row["_langs"].items():
            new_row[lang_code_map.get(lang, lang)] = text
        results.append(new_row)

    save_json(results, OUT_DIR / "ted.json", desc="ted.json")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1/3: FLORES-200 (gộp dev + devtest, bỏ prefix sentence_) ===")
    process_flores()

    print("\n=== 2/3: NTREX-128 ===")
    process_ntrex()

    print("\n=== 3/3: TED-2025 (mã 2 chữ -> lang_script, id = talk_id_timestamp) ===")
    process_ted2025()

    print("\nHoàn tất chuẩn hoá alignment dataset.")


if __name__ == "__main__":
    main()