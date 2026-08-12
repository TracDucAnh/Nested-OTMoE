"""
download_data.py
=================
Tải các bộ dữ liệu cho OT-MOE và lưu ra JSON, tổ chức theo cấu trúc:

    data/
      alignment/
        flores-200/    <- facebook/flores  (config "all")
        ntrex-128/     <- mteb/NTREX
        bible/         <- Helsinki-NLP/bible_para (corpus "bible-uedin" trên OPUS)
                          TOÀN BỘ ngôn ngữ có trong corpus, tự động lấy qua OPUS-API
                          (không hardcode danh sách cặp)
      downstream/
        mmmlu/
        xnli/
        xquad/         <- google/xquad, ĐỌC TRỰC TIẾP FILE PARQUET (nhánh
                          refs/convert/parquet trên HF Hub), KHÔNG dùng
                          load_dataset() thông thường -- xem ghi chú "LƯU Ý
                          LỖI XQuAD" bên dưới.
        tatoeba/       <- mteb/tatoeba-bitext-mining, MỖI CẶP NGÔN NGỮ 1 FILE JSON
                          (thay cho OPUS-100 trước đây). Danh sách cặp được lấy tự
                          động qua datasets.get_dataset_config_names(), không hardcode.

CƠ CHẾ SKIP (bỏ qua nếu đã tải):
    Mặc định, trước khi tải bất kỳ đơn vị dữ liệu nào (1 config/locale/ngôn ngữ/
    cặp ngôn ngữ), script sẽ kiểm tra xem thư mục output tương ứng đã có sẵn file
    .json hay chưa. Nếu có rồi thì bỏ qua (không gọi load_dataset lại, không ghi
    đè), rất hữu ích khi script bị ngắt giữa chừng và cần chạy lại. Muốn tải lại
    toàn bộ (ghi đè dữ liệu cũ) thì thêm cờ --force khi chạy.

LƯU Ý LỖI XQuAD ("Feature type 'List' not found"):
    Gần đây Hugging Face đã cập nhật metadata (README/dataset_info.json) của
    google/xquad sang kiểu feature mới "List" — kiểu này CHỈ được thư viện
    `datasets` bản >=4.0.0 hiểu. Nhưng project này lại cần ghim
    `datasets<4.0.0` vì FLORES-200/Bible cần trust_remote_code=True (đã bị
    GỠ BỎ ở datasets 4.0). Kẹt giữa 2 yêu cầu trái ngược này, nên với XQuAD,
    script KHÔNG gọi load_dataset("google/xquad", ...) như bình thường (sẽ
    lỗi "Feature type 'List' not found"), mà đọc THẲNG các file .parquet đã
    được HF tự động chuyển đổi (nhánh refs/convert/parquet của repo) bằng
    pyarrow — cách này bỏ qua hoàn toàn phần metadata bị lỗi. Xem hàm
    load_dataset_via_parquet() bên dưới.

Script này được đặt ở ROOT của project (ngang hàng với thư mục data/, ví dụ
OT-MOE/download_data.py), đúng như cấu trúc project hiện tại của bạn, nên:
    ALIGNMENT_DIR  = data/alignment
    DOWNSTREAM_DIR = data/downstream

Cài đặt:
    pip install -r requirements.txt

    LƯU Ý: FLORES-200 (facebook/flores) và Bible (Helsinki-NLP/bible_para)
    là dataset kiểu "loading script" cũ, cần trust_remote_code=True. Từ
    `datasets` bản 4.0 trở lên, cơ chế này đã bị GỠ BỎ hoàn toàn (sẽ báo lỗi
    "trust_remote_code is not supported anymore"). Vì vậy requirements.txt
    ghim `datasets<4.0.0` — đừng tự ý nâng cấp `datasets` lên bản mới hơn
    nếu vẫn muốn tải 2 bộ này. Tatoeba (mteb/tatoeba-bitext-mining) là
    parquet chuẩn nên không bị ảnh hưởng bởi giới hạn này. XQuAD cũng không
    bị ảnh hưởng nữa vì đã chuyển sang đọc parquet trực tiếp (xem ghi chú ở
    trên).

    Bộ Bible cần thêm gói `requests` (dùng để gọi OPUS-API lấy danh sách
    ngôn ngữ động, xem hàm get_bible_uedin_languages() bên dưới). XQuAD cũng
    dùng `requests` (tải file parquet) và `pyarrow` (đọc parquet) — cả hai
    đều đã có sẵn vì là dependency của `datasets`, chỉ `requests` cần cài
    thêm nếu chưa có:
        pip install requests

Chạy:
    python download_data.py                       # tải tất cả (tự skip phần đã có)
    python download_data.py --only flores ntrex bible tatoeba
    python download_data.py --only xquad            # chỉ tải XQuAD
    python download_data.py --list                 # xem danh sách các bộ hỗ trợ
    python download_data.py --force                # tải lại toàn bộ, ghi đè dữ liệu cũ
"""

import io
import os
import json
import argparse
import itertools
from pathlib import Path

import requests
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import login, list_repo_files
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# 1. HUGGING FACE TOKEN
# --------------------------------------------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)
    print("[+] Hugging Face authentication successful.")
else:
    print(
        "[!] HF_TOKEN đang trống — nếu dataset nào yêu cầu "
        "đăng nhập/gated thì việc tải sẽ lỗi."
    )

# --------------------------------------------------------------------------
# 2. ĐƯỜNG DẪN THƯ MỤC
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent        # .../OT-MOE (root project)
DOWNSTREAM_DIR = SCRIPT_DIR / "data" / "downstream"  # OT-MOE/data/downstream
ALIGNMENT_DIR = SCRIPT_DIR / "data" / "alignment"    # OT-MOE/data/alignment

# --------------------------------------------------------------------------
# 3. CẤU HÌNH NGÔN NGỮ / SUBSET CHO TỪNG DATASET
#    (tuỳ chỉnh trực tiếp các list dưới đây nếu muốn tải nhiều/ít hơn)
# --------------------------------------------------------------------------

# NTREX-128: dataset chỉ có 1 config "default" -> không cần list ngôn ngữ.

# Bible / bible-uedin (Helsinki-NLP/bible_para): KHÔNG hardcode danh sách cặp
# ngôn ngữ nữa. Mặc định script sẽ tự gọi OPUS-API để lấy TOÀN BỘ mã ngôn ngữ
# có trong corpus bible-uedin (xem get_bible_uedin_languages() bên dưới), rồi
# sinh tất cả cặp (C(n,2)) và thử tải từng cặp — cặp nào OPUS không có dữ
# liệu song song thì bỏ qua.
#
# Nếu bạn muốn GIỚI HẠN lại (ví dụ chỉ quan tâm một số ngôn ngữ cụ thể, để đỡ
# tốn thời gian/dung lượng) thì điền danh sách MÃ NGÔN NGỮ (không phải cặp)
# vào đây, ví dụ: ["en", "vi", "fr", "de", "es", "zh"]. Để None nếu muốn lấy
# toàn bộ ngôn ngữ có sẵn.
BIBLE_LANGUAGES_OVERRIDE = None  # ví dụ: ["en", "vi", "fr", "de", "es", "zh"]

# PIVOT: thay vì thử TOÀN BỘ C(n,2) ~ 5000 cặp (rất chậm và phần lớn sẽ bị
# bỏ qua vì OPUS không có đủ mọi cặp chéo), mặc định chỉ tải các cặp
# (pivot, X) cho mọi ngôn ngữ X còn lại — tức lấy 1 ngôn ngữ làm gốc/anchor,
# giống cách NTREX-128 lấy English làm nguồn. Số request giảm từ ~5000
# xuống còn ~n-1 (n = số ngôn ngữ trong bible-uedin, khoảng 101 request).
#
# Đặt BIBLE_PIVOT_LANGUAGE = None nếu vẫn muốn tải TOÀN BỘ mọi cặp chéo
# (n-way đầy đủ, không qua pivot) như trước.
BIBLE_PIVOT_LANGUAGE = "en"  # None để tải full C(n,2) cặp

# MMMLU: 14 locale được OpenAI dịch (xem README của openai/MMMLU)
MMMLU_LOCALES = [
    "AR_XY", "BN_BD", "DE_DE", "ES_LA", "FR_FR", "HI_IN", "ID_ID",
    "IT_IT", "JA_JP", "KO_KR", "PT_BR", "SW_KE", "YO_NG", "ZH_CN",
]

# XNLI: 15 config theo mã ngôn ngữ (có thể thay bằng ["all_languages"]
# nếu muốn 1 file duy nhất chứa tất cả ngôn ngữ - file sẽ rất nặng)
XNLI_LANGUAGES = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi",
    "ru", "sw", "th", "tr", "ur", "vi", "zh",
]

# XQuAD: 12 config, dạng "xquad.<lang>"
XQUAD_LANGUAGES = [
    "ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh",
]

# Tatoeba (mteb/tatoeba-bitext-mining): KHÔNG hardcode danh sách cặp nữa.
# Script tự gọi datasets.get_dataset_config_names() để lấy toàn bộ config
# (mỗi config là 1 cặp ngôn ngữ, dạng "xxx-eng") rồi tải từng cặp.

# Giới hạn số dòng tải về cho mỗi split (đặt None để tải toàn bộ).
# Hữu ích khi chỉ muốn test nhanh trước khi tải full (Tatoeba/XNLI rất lớn).
MAX_EXAMPLES_PER_SPLIT = None  # ví dụ: 5000


# --------------------------------------------------------------------------
# 4. HÀM TIỆN ÍCH
# --------------------------------------------------------------------------
# Đặt True bằng cờ --force khi chạy để tải lại toàn bộ, ghi đè dữ liệu cũ.
# Mặc định là False -> mọi phần dữ liệu đã có sẵn (đã có file .json) sẽ được
# bỏ qua thay vì tải lại.
FORCE_REDOWNLOAD = False


def output_already_exists(out_dir: Path) -> bool:
    """Kiểm tra nhanh xem 1 đơn vị dữ liệu (1 config/locale/ngôn ngữ/cặp ngôn
    ngữ, ứng với 1 thư mục output) đã được tải trước đó hay chưa, bằng cách
    xem thư mục đó đã tồn tại và có ít nhất 1 file .json hay không.

    Dùng để skip TOÀN BỘ lệnh load_dataset() cho đơn vị đó (tiết kiệm băng
    thông/thời gian), thay vì chỉ skip lúc ghi file.
    """
    if not out_dir.exists():
        return False
    return any(out_dir.glob("*.json"))


def save_split_as_json(dataset_split, out_path: Path, desc: str):
    """Chuyển 1 split của HF Dataset thành list[dict] và ghi ra file JSON,
    có thanh tiến trình tqdm chạy theo từng dòng dữ liệu.

    Nếu file out_path đã tồn tại và FORCE_REDOWNLOAD=False (mặc định), hàm sẽ
    bỏ qua (không ghi đè) -> đây là lớp skip "chi tiết" (theo từng split),
    bổ sung cho lớp skip "thô" (theo từng đơn vị dữ liệu) ở output_already_exists().
    """
    if out_path.exists() and not FORCE_REDOWNLOAD:
        print(f"    [skip] {out_path} đã tồn tại -> bỏ qua (dùng --force để tải lại).")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    n = len(dataset_split)
    if MAX_EXAMPLES_PER_SPLIT is not None:
        n = min(n, MAX_EXAMPLES_PER_SPLIT)

    for i in tqdm(range(n), desc=desc, unit="dòng"):
        records.append(dataset_split[i])

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"    -> đã lưu {len(records)} dòng vào {out_path}")


def safe_load_dataset(repo_id: str, config: str = None, **kwargs):
    """Wrapper quanh load_dataset() để bắt lỗi gọn gàng và in ra chỗ lỗi."""
    try:
        if config:
            return load_dataset(repo_id, config, **kwargs)
        return load_dataset(repo_id, **kwargs)
    except Exception as e:
        print(f"[LỖI] Không tải được {repo_id} (config={config}): {e}")
        return None


def load_dataset_via_parquet(repo_id: str, config_name: str) -> dict:
    """Đọc thẳng dữ liệu của 1 config từ các file .parquet đã được Hugging
    Face tự động chuyển đổi (nhánh 'refs/convert/parquet' của repo), BỎ QUA
    hoàn toàn dataset_info.json/README YAML của repo gốc.

    Dùng cho XQuAD vì metadata của google/xquad hiện dùng kiểu feature mới
    "List" (chỉ datasets>=4.0.0 hiểu được), trong khi project cần ghim
    datasets<4.0.0 cho FLORES-200/Bible -> load_dataset() thông thường sẽ
    lỗi "Feature type 'List' not found" dù dữ liệu vẫn tải được bình thường.
    Đọc thẳng parquet bằng pyarrow tránh được lỗi này hoàn toàn.

    Trả về dict {split_name: [row_dict, ...]} (list[dict] kiểu Python thuần,
    sẵn sàng json.dump), hoặc {} nếu không tìm thấy/không tải được.
    """
    revision = "refs/convert/parquet"
    try:
        files = list_repo_files(repo_id, repo_type="dataset", revision=revision)
    except Exception as e:
        print(f"[LỖI] Không lấy được danh sách file parquet của {repo_id} "
              f"(revision={revision}): {e}")
        return {}

    prefix = f"{config_name}/"
    parquet_files = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not parquet_files:
        print(f"[LỖI] Không tìm thấy file parquet nào cho {repo_id} "
              f"(config={config_name}) trên nhánh {revision}.")
        return {}

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    by_split: dict[str, list] = {}
    for rel_path in parquet_files:
        # dạng thường gặp: "<config>/<split>-00000-of-00001.parquet"
        split = rel_path[len(prefix):].split("/")[0].split("-")[0]
        url = (f"https://huggingface.co/datasets/{repo_id}/resolve/"
               f"{revision.replace('/', '%2F')}/{rel_path}")
        try:
            resp = requests.get(url, headers=headers, timeout=120)
            resp.raise_for_status()
        except Exception as e:
            print(f"[LỖI] Không tải được file parquet {url}: {e}")
            continue
        table = pq.read_table(io.BytesIO(resp.content))
        by_split.setdefault(split, []).extend(table.to_pylist())

    return by_split


def save_records_as_json(records: list, out_path: Path, desc: str):
    """Giống save_split_as_json() nhưng nhận thẳng list[dict] (dùng cho dữ
    liệu đọc qua load_dataset_via_parquet(), không phải HF Dataset object).
    Cũng tự skip nếu file đã tồn tại (trừ khi FORCE_REDOWNLOAD=True)."""
    if out_path.exists() and not FORCE_REDOWNLOAD:
        print(f"    [skip] {out_path} đã tồn tại -> bỏ qua (dùng --force để tải lại).")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if MAX_EXAMPLES_PER_SPLIT is not None:
        records = records[:MAX_EXAMPLES_PER_SPLIT]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"    -> đã lưu {len(records)} dòng vào {out_path}")


def get_bible_uedin_languages():
    """Lấy TOÀN BỘ mã ngôn ngữ có trong corpus bible-uedin trên OPUS, thông
    qua OPUS-API (https://opus.nlpl.eu/opusapi), thay vì hardcode sẵn danh
    sách. Trả về list rỗng nếu gọi API thất bại (mất mạng, đổi API, ...).
    """
    api_url = "https://opus.nlpl.eu/opusapi"
    try:
        resp = requests.get(
            api_url,
            params={"languages": "True", "corpus": "bible-uedin"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[LỖI] Không gọi được OPUS-API để lấy danh sách ngôn ngữ "
              f"bible-uedin: {e}")
        return []

    # Cấu trúc JSON trả về của OPUS-API có thể thay đổi theo phiên bản, nên
    # xử lý vài trường hợp phổ biến: {"languages": [...]}, hoặc list thẳng.
    if isinstance(data, dict):
        langs = data.get("languages") or data.get("language") or []
    elif isinstance(data, list):
        langs = data
    else:
        langs = []

    # Mỗi phần tử có thể là string mã ngôn ngữ, hoặc dict {"language": "en"}
    cleaned = []
    for item in langs:
        if isinstance(item, str):
            cleaned.append(item)
        elif isinstance(item, dict):
            code = item.get("language") or item.get("code") or item.get("name")
            if code:
                cleaned.append(code)

    return sorted(set(cleaned))


# --------------------------------------------------------------------------
# 5. ALIGNMENT: FLORES-200, NTREX-128, Bible (bible-uedin)
# --------------------------------------------------------------------------
def download_flores200():
    print("\n=== FLORES-200 -> alignment/flores-200/ ===")
    # LƯU Ý: FLORES-200 gốc chỉ công khai 2 split "dev" (997 câu) và
    # "devtest" (1012 câu). Split "test" bị Meta giữ kín (hidden test set)
    # để chấm leaderboard nội bộ -> KHÔNG tồn tại bản public nào để tải, kể
    # cả trên openlanguagedata/flores_plus (bản kế thừa) cũng chỉ có dev +
    # devtest. Vì vậy vòng lặp for split in ds.keys() bên dưới đã tự động
    # lấy HẾT mọi split có sẵn rồi, không cần (và không thể) lấy thêm "train".
    #
    # Dataset này cũng đã chuyển sang GATED: cần đăng nhập bằng HF_TOKEN của
    # tài khoản đã bấm "Accept" điều khoản tại
    # https://huggingface.co/datasets/facebook/flores , nếu không sẽ lỗi
    # permission khi tải.
    out_dir = ALIGNMENT_DIR / "flores-200"
    if not FORCE_REDOWNLOAD and output_already_exists(out_dir):
        print(f"  [skip] flores-200 đã có dữ liệu tại {out_dir} -> bỏ qua toàn bộ "
              f"(dùng --force để tải lại).")
        return

    ds = safe_load_dataset("facebook/flores", "all", trust_remote_code=True)
    if ds is None:
        print("    -> Nếu lỗi permission/gated: vào "
              "https://huggingface.co/datasets/facebook/flores , đăng nhập "
              "và bấm Accept điều khoản bằng đúng tài khoản ứng với HF_TOKEN.")
        return

    found_splits = list(ds.keys())
    print(f"  Các split có sẵn: {found_splits} "
          f"({', '.join(f'{s}={len(ds[s])} dòng' for s in found_splits)})")

    for split in tqdm(found_splits, desc="FLORES-200 splits"):
        save_split_as_json(
            ds[split], out_dir / f"{split}.json", desc=f"  flores-200/{split}"
        )


def download_ntrex128():
    print("\n=== NTREX-128 -> alignment/ntrex-128/ ===")
    # LƯU Ý: mteb/NTREX chỉ có DUY NHẤT 1 split "test" (~2000 dòng). Đây là
    # bộ eval MT, không có train/dev -> vòng lặp bên dưới đã lấy hết những gì
    # có sẵn.
    out_dir = ALIGNMENT_DIR / "ntrex-128"
    if not FORCE_REDOWNLOAD and output_already_exists(out_dir):
        print(f"  [skip] ntrex-128 đã có dữ liệu tại {out_dir} -> bỏ qua toàn bộ "
              f"(dùng --force để tải lại).")
        return

    ds = safe_load_dataset("mteb/NTREX")
    if ds is None:
        return

    found_splits = list(ds.keys())
    print(f"  Các split có sẵn: {found_splits} "
          f"({', '.join(f'{s}={len(ds[s])} dòng' for s in found_splits)})")

    for split in tqdm(found_splits, desc="NTREX-128 splits"):
        save_split_as_json(
            ds[split], out_dir / f"{split}.json", desc=f"  ntrex-128/{split}"
        )


def download_bible():
    print("\n=== Bible / bible-uedin -> alignment/bible/ ===")
    out_root = ALIGNMENT_DIR / "bible"

    languages = BIBLE_LANGUAGES_OVERRIDE or get_bible_uedin_languages()
    if not languages:
        print("[!] Không lấy được danh sách ngôn ngữ bible-uedin -> bỏ qua bible.")
        print("    -> Kiểm tra kết nối mạng tới opus.nlpl.eu, hoặc set thủ công")
        print("       biến BIBLE_LANGUAGES_OVERRIDE ở đầu file để bỏ qua bước gọi API.")
        return

    if BIBLE_PIVOT_LANGUAGE:
        if BIBLE_PIVOT_LANGUAGE not in languages:
            print(f"[!] Pivot '{BIBLE_PIVOT_LANGUAGE}' không có trong danh sách "
                  f"ngôn ngữ bible-uedin lấy được -> vẫn thử tải (OPUS có thể "
                  f"vẫn hỗ trợ), nhưng kiểm tra lại mã ngôn ngữ nếu lỗi hết.")
        # Chỉ lấy các cặp (pivot, X) -> n-1 cặp thay vì C(n,2), nhanh hơn
        # nhiều và vẫn đủ dùng cho mục đích alignment qua 1 ngôn ngữ gốc.
        pairs = [
            (BIBLE_PIVOT_LANGUAGE, lang)
            for lang in languages
            if lang != BIBLE_PIVOT_LANGUAGE
        ]
        print(f"  bible-uedin có {len(languages)} ngôn ngữ -> dùng pivot "
              f"'{BIBLE_PIVOT_LANGUAGE}', sẽ thử {len(pairs)} cặp "
              f"({BIBLE_PIVOT_LANGUAGE}-X). Đặt BIBLE_PIVOT_LANGUAGE = None "
              f"ở đầu file nếu muốn tải TOÀN BỘ C(n,2) cặp chéo thay vì qua pivot.")
    else:
        pairs = list(itertools.combinations(languages, 2))
        print(f"  bible-uedin có {len(languages)} ngôn ngữ -> sẽ thử toàn bộ "
              f"{len(pairs)} cặp (một số cặp có thể không có dữ liệu song song, "
              f"script sẽ tự bỏ qua các cặp đó). Việc này sẽ khá lâu — đặt "
              f"BIBLE_PIVOT_LANGUAGE = 'en' (hoặc mã khác) nếu muốn tải nhanh "
              f"hơn qua 1 ngôn ngữ pivot.")

    ok, skipped, already = 0, 0, 0
    for lang1, lang2 in tqdm(pairs, desc="Bible language pairs"):
        pair_name = f"{lang1}-{lang2}"

        if not FORCE_REDOWNLOAD and output_already_exists(out_root / pair_name):
            already += 1
            continue

        try:
            ds = load_dataset(
                "Helsinki-NLP/bible_para",
                lang1=lang1,
                lang2=lang2,
                trust_remote_code=True,
            )
        except Exception:
            # Không phải cặp nào trong C(n,2) cũng thực sự có bitext trên
            # OPUS -> bỏ qua lặng lẽ để không spam log cho hàng nghìn cặp.
            skipped += 1
            continue

        ok += 1
        for split in ds.keys():
            save_split_as_json(
                ds[split],
                out_root / pair_name / f"{split}.json",
                desc=f"  bible/{pair_name}/{split}",
            )

    print(f"  -> Hoàn tất bible: {ok} cặp tải mới, {already} cặp đã có sẵn (skip), "
          f"{skipped} cặp bỏ qua (không tồn tại trên OPUS).")


# --------------------------------------------------------------------------
# 6. DOWNSTREAM: MMMLU, XNLI, XQuAD, Tatoeba
# --------------------------------------------------------------------------
def download_mmmlu():
    print("\n=== MMMLU -> downstream/mmmlu/ ===")
    out_root = DOWNSTREAM_DIR / "mmmlu"
    for locale in tqdm(MMMLU_LOCALES, desc="MMMLU locales"):
        if not FORCE_REDOWNLOAD and output_already_exists(out_root / locale):
            print(f"  [skip] mmmlu/{locale} đã có dữ liệu -> bỏ qua.")
            continue

        ds = safe_load_dataset("openai/MMMLU", locale)
        if ds is None:
            continue
        for split in ds.keys():
            save_split_as_json(
                ds[split],
                out_root / locale / f"{split}.json",
                desc=f"  mmmlu/{locale}/{split}",
            )


def download_xnli():
    print("\n=== XNLI -> downstream/xnli/ ===")
    out_root = DOWNSTREAM_DIR / "xnli"
    for lang in tqdm(XNLI_LANGUAGES, desc="XNLI languages"):
        if not FORCE_REDOWNLOAD and output_already_exists(out_root / lang):
            print(f"  [skip] xnli/{lang} đã có dữ liệu -> bỏ qua.")
            continue

        ds = safe_load_dataset("facebook/xnli", lang)
        if ds is None:
            continue
        for split in ds.keys():
            save_split_as_json(
                ds[split],
                out_root / lang / f"{split}.json",
                desc=f"  xnli/{lang}/{split}",
            )


def download_xquad():
    print("\n=== XQuAD -> downstream/xquad/ ===")
    # LƯU Ý: KHÔNG dùng safe_load_dataset()/load_dataset() thông thường ở
    # đây vì metadata của google/xquad hiện dùng kiểu feature "List" (chỉ
    # datasets>=4.0.0 hiểu), trong khi project ghim datasets<4.0.0 cho
    # FLORES-200/Bible -> sẽ lỗi "Feature type 'List' not found". Thay vào
    # đó đọc thẳng file parquet qua load_dataset_via_parquet() (xem ghi chú
    # đầu file). google/xquad có ĐÚNG 12 config (11 ngôn ngữ dịch + tiếng
    # Anh gốc) -- XQUAD_LANGUAGES ở trên đã liệt kê đủ.
    out_root = DOWNSTREAM_DIR / "xquad"
    repo_id = "google/xquad"

    for lang in tqdm(XQUAD_LANGUAGES, desc="XQuAD languages"):
        if not FORCE_REDOWNLOAD and output_already_exists(out_root / lang):
            print(f"  [skip] xquad/{lang} đã có dữ liệu -> bỏ qua.")
            continue

        splits = load_dataset_via_parquet(repo_id, f"xquad.{lang}")
        if not splits:
            continue
        for split, records in splits.items():
            save_records_as_json(
                records,
                out_root / lang / f"{split}.json",
                desc=f"  xquad/{lang}/{split}",
            )


def download_tatoeba():
    print("\n=== Tatoeba (mteb/tatoeba-bitext-mining) -> downstream/tatoeba/ ===")
    out_root = DOWNSTREAM_DIR / "tatoeba"
    repo_id = "mteb/tatoeba-bitext-mining"

    # Không hardcode danh sách cặp: lấy toàn bộ config (mỗi config là 1 cặp
    # ngôn ngữ, dạng "xxx-eng") trực tiếp từ Hugging Face Hub.
    try:
        configs = get_dataset_config_names(repo_id)
    except Exception as e:
        print(f"[LỖI] Không lấy được danh sách cặp ngôn ngữ của {repo_id}: {e}")
        return

    print(f"  Tìm thấy {len(configs)} cặp ngôn ngữ trong {repo_id}.")

    for pair in tqdm(configs, desc="Tatoeba language pairs"):
        if not FORCE_REDOWNLOAD and output_already_exists(out_root / pair):
            print(f"  [skip] tatoeba/{pair} đã có dữ liệu -> bỏ qua.")
            continue

        ds = safe_load_dataset(repo_id, pair)
        if ds is None:
            continue
        # Mỗi cặp ngôn ngữ -> 1 file JSON riêng cho mỗi split (thường chỉ có
        # split "test", nên thực chất là 1 file JSON / cặp ngôn ngữ).
        for split in ds.keys():
            save_split_as_json(
                ds[split],
                out_root / pair / f"{split}.json",
                desc=f"  tatoeba/{pair}/{split}",
            )


# --------------------------------------------------------------------------
# 7. MAIN
# --------------------------------------------------------------------------
DATASET_REGISTRY = {
    "flores": download_flores200,
    "ntrex": download_ntrex128,
    "bible": download_bible,
    "mmmlu": download_mmmlu,
    "xnli": download_xnli,
    "xquad": download_xquad,
    "tatoeba": download_tatoeba,
}


def main():
    parser = argparse.ArgumentParser(description="Tải dữ liệu cho OT-MOE")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
        help="Chỉ tải các dataset được liệt kê (mặc định: tải tất cả)",
    )
    parser.add_argument(
        "--list", action="store_true", help="In danh sách dataset hỗ trợ rồi thoát"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải lại toàn bộ, ghi đè dữ liệu đã có sẵn thay vì skip "
             "(mặc định: tự động skip phần dữ liệu đã tải trước đó).",
    )
    args = parser.parse_args()

    if args.list:
        print("Các dataset hỗ trợ:")
        for k in DATASET_REGISTRY:
            print(f"  - {k}")
        return

    global FORCE_REDOWNLOAD
    FORCE_REDOWNLOAD = args.force

    targets = args.only if args.only else list(DATASET_REGISTRY.keys())

    print(f"Sẽ tải: {targets}")
    print(f"ALIGNMENT_DIR  = {ALIGNMENT_DIR}")
    print(f"DOWNSTREAM_DIR = {DOWNSTREAM_DIR}")
    print(f"Chế độ: {'FORCE tải lại toàn bộ (ghi đè)' if FORCE_REDOWNLOAD else 'tự động SKIP phần đã có sẵn'}")

    for name in targets:
        DATASET_REGISTRY[name]()

    print("\nHoàn tất.")


if __name__ == "__main__":
    main()