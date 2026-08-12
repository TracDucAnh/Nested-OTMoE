"""
process_data.py
================
Chuẩn hoá 3 bộ dữ liệu alignment (flores-200, ntrex-128, bible) về CÙNG 1
định dạng N-way, lưu trữ theo kiểu NTREX (mỗi bản ghi = 1 dict gồm "id" +
các cột "{iso639_3}_{iso15924}": "câu dịch").

Input  (đọc từ):
    data/alignment/flores-200/dev.json
    data/alignment/flores-200/devtest.json
    data/alignment/ntrex-128/test.json
    data/alignment/bible/<lang1>-<lang2>/*.json      (2-way, pivot = en)

Output (ghi ra):
    data/processed_alignment/flores.json   (gộp dev + devtest, bỏ prefix "sentence_")
    data/processed_alignment/ntrex.json
    data/processed_alignment/bible.json    (2-way -> N-way qua pivot "en")

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
# 3. BIBLE: 2-way (en-xx) -> N-way qua pivot "en" + đổi mã 2 chữ -> lang_script
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
    # nhiều mã trong bible-uedin (OPUS) vốn đã là ISO 639-3 sẵn (vd: acu, agr)
    return code


def detect_script(samples: list) -> str:
    """Phát hiện script THỰC TẾ từ nội dung câu bằng thư viện unicodedataplus
    (đếm script của từng ký tự chữ cái trong mẫu câu, lấy script chiếm đa số),
    KHÔNG suy ra script từ mã ngôn ngữ."""
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
    return f"{to_iso639_3(code)}_{detect_script(samples)}"


def process_bible():
    bible_root = ALIGNMENT_DIR / "bible"
    if not bible_root.exists():
        print(f"  [!] không thấy {bible_root}, bỏ qua bible.")
        return

    pair_dirs = sorted(p for p in bible_root.iterdir() if p.is_dir() and "-" in p.name)
    if not pair_dirs:
        print(f"  [!] {bible_root} không có cặp ngôn ngữ nào, bỏ qua bible.")
        return

    # tự phát hiện ngôn ngữ pivot: ngôn ngữ đứng đầu ("lang1") xuất hiện nhiều nhất
    lang1_counts = Counter(p.name.split("-", 1)[0] for p in pair_dirs)
    pivot_lang = lang1_counts.most_common(1)[0][0]
    print(f"  Phát hiện ngôn ngữ pivot: '{pivot_lang}' "
          f"({lang1_counts[pivot_lang]}/{len(pair_dirs)} cặp)")

    SAMPLE_LIMIT = 300
    samples_by_lang = defaultdict(list)
    # pivot_text -> {"_row_id": int, "_langs": {lang_code: text}}
    # LƯU Ý: id được tách RIÊNG khỏi dict ngôn ngữ, vì "id" cũng là mã ISO 639-1
    # của tiếng Indonesia -> nếu để chung 1 dict, key ngôn ngữ "id" sẽ ghi đè lên
    # id số nguyên (row["id"] = câu tiếng Indonesia), gây lỗi khi sort() vì trộn
    # lẫn kiểu int và str.
    rows_by_pivot_text = {}
    next_id = 0

    for pdir in tqdm(pair_dirs, desc="Bible: gộp các cặp ngôn ngữ (2-way -> N-way)"):
        lang1, lang2 = pdir.name.split("-", 1)
        if lang1 != pivot_lang:  # phòng khi thư mục lưu ngược chiều
            lang1, lang2 = lang2, lang1

        for split_file in sorted(pdir.glob("*.json")):
            try:
                with open(split_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except Exception as e:
                print(f"    [lỗi] không đọc được {split_file}: {e}")
                continue

            for rec in records:
                tr = rec.get("translation", {})
                pivot_text = (tr.get(pivot_lang) or "").strip()
                other_text = (tr.get(lang2) or "").strip()
                if not pivot_text or not other_text:
                    continue

                if len(samples_by_lang[pivot_lang]) < SAMPLE_LIMIT:
                    samples_by_lang[pivot_lang].append(pivot_text)
                if len(samples_by_lang[lang2]) < SAMPLE_LIMIT:
                    samples_by_lang[lang2].append(other_text)

                # câu tiếng Anh (pivot) làm khoá gộp N-way
                entry = rows_by_pivot_text.get(pivot_text)
                if entry is None:
                    entry = {"_row_id": next_id, "_langs": {pivot_lang: pivot_text}}
                    rows_by_pivot_text[pivot_text] = entry
                    next_id += 1
                entry["_langs"][lang2] = other_text

    print(f"  Tổng cộng {len(rows_by_pivot_text)} câu đã gộp theo pivot '{pivot_lang}'.")

    print("  Đang xác định mã lang_script cho từng ngôn ngữ (langcodes + unicodedataplus)...")
    lang_code_map = {}
    for lang, samples in tqdm(samples_by_lang.items(), desc="  Xác định lang_script"):
        lang_code_map[lang] = to_lang_script(lang, samples)

    results = []
    for entry in tqdm(rows_by_pivot_text.values(), desc="  Chuẩn hoá bản ghi bible"):
        new_row = {"id": entry["_row_id"]}
        for lang, text in entry["_langs"].items():
            new_row[lang_code_map.get(lang, lang)] = text
        results.append(new_row)

    results.sort(key=lambda r: r["id"])
    save_json(results, OUT_DIR / "bible.json", desc="bible.json")


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1/3: FLORES-200 (gộp dev + devtest, bỏ prefix sentence_) ===")
    process_flores()

    print("\n=== 2/3: NTREX-128 ===")
    process_ntrex()

    print("\n=== 3/3: Bible (2-way -> N-way qua pivot, 2 ký tự -> lang_script) ===")
    process_bible()

    print("\nHoàn tất chuẩn hoá alignment dataset.")


if __name__ == "__main__":
    main()