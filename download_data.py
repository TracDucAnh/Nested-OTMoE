"""
download_data.py
=================
Tải các bộ dữ liệu cho OT-MOE và lưu ra JSON, tổ chức theo cấu trúc:

    data/
      alignment/
        flores-200/    <- facebook/flores  (config "all")
        ntrex-128/     <- mteb/NTREX
        ted-2025/      <- Tải TRỰC TIẾP file .zip từ Google Drive (bằng gdown),
                          KHÔNG qua Hugging Face. File .zip được tải tạm vào
                          alignment/ (ngang hàng ted-2025/), sau đó NỘI DUNG bên
                          trong zip được giải nén thẳng vào alignment/ted-2025/
                          (nếu zip có 1 thư mục gốc chung, thư mục đó bị bỏ đi
                          để tránh lồng thêm 1 cấp, ví dụ tránh
                          alignment/ted-2025/ted-2025/...). File .zip tạm bị
                          xoá sau khi giải nén xong.
      downstream/
        mmmlu/
        xnli/
        xquad/         <- google/xquad, ĐỌC TRỰC TIẾP FILE PARQUET (nhánh
                          refs/convert/parquet trên HF Hub), KHÔNG dùng
                          load_dataset() thông thường -- xem ghi chú "LƯU Ý
                          LỖI XQuAD" bên dưới.
      english_task/    <- các bộ task TIẾNG ANH (dùng cho hướng alternate training:
                          alignment -> task -> alignment), mỗi bộ 1 thư mục, mỗi split
                          1 file JSON:
        snli/          <- stanfordnlp/snli (config "plain_text"; train/validation/test)
                          LƯU Ý: giữ nguyên dữ liệu gốc, các dòng không có nhãn gold
                          có label = -1 (chưa lọc).
        squad/         <- rajpurkar/squad (config "plain_text", SQuAD v1.1 -- cùng
                          phiên bản gốc với XQuAD; train/validation)
        mmlu/          <- cais/mmlu (config "all", gộp 57 subject; auxiliary_train/
                          test/validation/dev). auxiliary_train rất lớn (~100k dòng).

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
    `datasets<4.0.0` vì FLORES-200 cần trust_remote_code=True (đã bị
    GỠ BỎ ở datasets 4.0). Kẹt giữa 2 yêu cầu trái ngược này, nên với XQuAD,
    script KHÔNG gọi load_dataset("google/xquad", ...) như bình thường (sẽ
    lỗi "Feature type 'List' not found"), mà đọc THẲNG các file .parquet đã
    được HF tự động chuyển đổi (nhánh refs/convert/parquet của repo) bằng
    pyarrow — cách này bỏ qua hoàn toàn phần metadata bị lỗi. Xem hàm
    load_dataset_via_parquet() bên dưới.

Script này được đặt ở ROOT của project (ngang hàng với thư mục data/, ví dụ
OT-MOE/download_data.py), đúng như cấu trúc project hiện tại của bạn, nên:
    ALIGNMENT_DIR    = data/alignment
    DOWNSTREAM_DIR   = data/downstream
    ENGLISH_TASK_DIR = data/english_task

CƠ CHẾ FALLBACK CHO SNLI / SQuAD / MMLU:
    Các bộ này có trường dạng list (answers, choices) nên có nguy cơ gặp lại lỗi
    metadata "Feature type 'List' not found" như XQuAD. Vì vậy chúng được tải
    bằng load_splits_with_parquet_fallback(): thử load_dataset() trước; nếu lỗi
    thì tự đọc thẳng file .parquet bằng pyarrow (thử nhánh refs/convert/parquet,
    rồi tới nhánh main của repo).

Cài đặt:
    pip install -r requirements.txt

    LƯU Ý: FLORES-200 (facebook/flores) là dataset kiểu "loading script" cũ,
    cần trust_remote_code=True. Từ `datasets` bản 4.0 trở lên, cơ chế này đã
    bị GỠ BỎ hoàn toàn (sẽ báo lỗi "trust_remote_code is not supported
    anymore"). Vì vậy requirements.txt ghim `datasets<4.0.0` — đừng tự ý
    nâng cấp `datasets` lên bản mới hơn nếu vẫn muốn tải bộ này. XQuAD không
    bị ảnh hưởng bởi giới hạn này vì đã chuyển sang đọc parquet trực tiếp
    (xem ghi chú ở trên).

    XQuAD dùng `requests` (tải file parquet) và `pyarrow` (đọc parquet) —
    cả hai đều đã có sẵn vì là dependency của `datasets`.

    Bộ TED-2025 cần thêm gói `gdown` (dùng để tải file .zip công khai từ
    Google Drive theo link chia sẻ, xem hàm download_ted2025() bên dưới):
        pip install gdown

Chạy:
    python download_data.py                       # tải tất cả (tự skip phần đã có)
    python download_data.py --only flores ntrex ted2025
    python download_data.py --only xquad            # chỉ tải XQuAD
    python download_data.py --only ted2025          # chỉ tải TED-2025 (từ Google Drive)
    python download_data.py --only snli squad mmlu  # chỉ tải 3 bộ task tiếng Anh
                                                    # (lưu ý: "mmlu" khác "mmmlu")
    python download_data.py --list                 # xem danh sách các bộ hỗ trợ
    python download_data.py --force                # tải lại toàn bộ, ghi đè dữ liệu cũ
"""

import io
import os
import re
import json
import shutil
import zipfile
import argparse
from pathlib import Path

import requests
import pyarrow.parquet as pq
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import login, list_repo_files
from dotenv import load_dotenv

try:
    import gdown
except ImportError:
    # gdown chỉ cần thiết cho download_ted2025(); nếu chưa cài, các dataset
    # khác vẫn chạy bình thường -- download_ted2025() sẽ tự báo lỗi rõ ràng
    # và hướng dẫn `pip install gdown` khi được gọi tới.
    gdown = None

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
ENGLISH_TASK_DIR = SCRIPT_DIR / "data" / "english_task"  # OT-MOE/data/english_task

# --------------------------------------------------------------------------
# 3. CẤU HÌNH NGÔN NGỮ / SUBSET CHO TỪNG DATASET
#    (tuỳ chỉnh trực tiếp các list dưới đây nếu muốn tải nhiều/ít hơn)
# --------------------------------------------------------------------------

# NTREX-128: dataset chỉ có 1 config "default" -> không cần list ngôn ngữ.

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

# Giới hạn số dòng tải về cho mỗi split (đặt None để tải toàn bộ).
# Hữu ích khi chỉ muốn test nhanh trước khi tải full (XNLI rất lớn).
MAX_EXAMPLES_PER_SPLIT = None  # ví dụ: 5000

# TED-2025: link chia sẻ Google Drive của file .zip (dataset không nằm trên
# Hugging Face, tải trực tiếp bằng gdown). Link phải ở chế độ chia sẻ công
# khai ("Anyone with the link") thì gdown mới tải được mà không cần đăng nhập.
TED2025_GDRIVE_URL = (
    "https://drive.google.com/file/d/1bSr5bDC7kvl2oMx7-65vku3O_ziIjqZr/view?usp=sharing"
)


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


def output_dir_has_any_files(out_dir: Path) -> bool:
    """Giống output_already_exists(), nhưng kiểm tra BẤT KỲ file nào (không
    riêng .json) đã tồn tại trong thư mục hay chưa -- dùng cho các dataset
    không được lưu ra JSON mà chỉ giải nén thô (ví dụ TED-2025), vì nội dung
    giải nén có thể là .txt/.xml/... tuỳ theo dataset gốc, không riêng .json.
    """
    if not out_dir.exists():
        return False
    return any(p.is_file() for p in out_dir.rglob("*"))


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


def load_dataset_via_parquet(
    repo_id: str, config_name: str, revision: str = "refs/convert/parquet"
) -> dict:
    """Đọc thẳng dữ liệu của 1 config từ các file .parquet đã được Hugging
    Face tự động chuyển đổi (nhánh 'refs/convert/parquet' của repo), BỎ QUA
    hoàn toàn dataset_info.json/README YAML của repo gốc.

    Dùng cho XQuAD vì metadata của google/xquad hiện dùng kiểu feature mới
    "List" (chỉ datasets>=4.0.0 hiểu được), trong khi project cần ghim
    datasets<4.0.0 cho FLORES-200 -> load_dataset() thông thường sẽ
    lỗi "Feature type 'List' not found" dù dữ liệu vẫn tải được bình thường.
    Đọc thẳng parquet bằng pyarrow tránh được lỗi này hoàn toàn.

    Tham số `revision` mặc định là nhánh tự động chuyển đổi của HF
    ('refs/convert/parquet', dùng cho XQuAD). Với các repo vốn đã là parquet
    (như SNLI/SQuAD/MMLU), nhánh convert có thể không tồn tại -> truyền
    revision="main" để đọc thẳng file parquet gốc của repo.

    Trả về dict {split_name: [row_dict, ...]} (list[dict] kiểu Python thuần,
    sẵn sàng json.dump), hoặc {} nếu không tìm thấy/không tải được.
    """
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


def load_splits_with_parquet_fallback(repo_id: str, config: str = None) -> dict:
    """Tải tất cả split của 1 dataset, có cơ chế dự phòng khi load_dataset()
    lỗi (ví dụ lỗi metadata "Feature type 'List' not found" với datasets<4.0):

      1. Thử load_dataset() bình thường.
      2. Nếu lỗi: đọc thẳng file parquet ở nhánh 'refs/convert/parquet'.
      3. Nếu vẫn không có: đọc file parquet gốc ở nhánh 'main' (dành cho các
         repo vốn đã là parquet nên HF không tạo nhánh convert).

    Trả về dict {split_name: HF Dataset hoặc list[dict]}, hoặc {} nếu thất bại.
    Dùng save_any_split() để ghi từng phần tử ra JSON.
    """
    ds = safe_load_dataset(repo_id, config)
    if ds is not None:
        return {split: ds[split] for split in ds.keys()}

    if not config:
        return {}

    for revision in ("refs/convert/parquet", "main"):
        print(f"    -> Thử đọc thẳng file parquet của {repo_id} "
              f"(config={config}, revision={revision}) ...")
        splits = load_dataset_via_parquet(repo_id, config, revision=revision)
        if splits:
            return splits
    return {}


def save_any_split(split_data, out_path: Path, desc: str):
    """Ghi 1 split ra JSON, dù nó là HF Dataset (từ load_dataset) hay
    list[dict] (từ load_dataset_via_parquet)."""
    if isinstance(split_data, list):
        save_records_as_json(split_data, out_path, desc=desc)
    else:
        save_split_as_json(split_data, out_path, desc=desc)


def is_macos_zip_junk(member_path: str) -> bool:
    """Kiểm tra 1 entry trong file zip có phải rác do macOS tự sinh khi nén
    (qua Finder/Archive Utility) hay không, để loại bỏ khi giải nén:
        - Thư mục "__MACOSX/" (và mọi thứ nằm bên trong nó)
        - File resource-fork ẩn, tên bắt đầu bằng "._" (ví dụ "._en.txt")
        - File ".DS_Store" (metadata thư mục của Finder)
    """
    parts = [p for p in member_path.split("/") if p]
    if not parts:
        return False
    if parts[0] == "__MACOSX":
        return True
    basename = parts[-1]
    if basename.startswith("._") or basename == ".DS_Store":
        return True
    return False


def extract_gdrive_file_id(url_or_id: str) -> str:
    """Tách file ID từ 1 link chia sẻ Google Drive. Hỗ trợ các dạng phổ biến:
        https://drive.google.com/file/d/<ID>/view?usp=sharing
        https://drive.google.com/open?id=<ID>
        https://drive.google.com/uc?id=<ID>
    Nếu không khớp dạng nào ở trên, coi như chuỗi truyền vào đã là ID trần
    và trả về nguyên văn (fallback).

    Cố tình tự tách ID thay vì dùng gdown.download(url=..., fuzzy=True), vì
    tham số "fuzzy" chỉ xuất hiện ở các bản gdown khá mới (>=4.4) -- dùng
    gdown.download(id=<ID>, ...) hoạt động ổn định với MỌI bản gdown, kể cả
    bản cũ không có "fuzzy".
    """
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id.strip()


# --------------------------------------------------------------------------
# 5. ALIGNMENT: FLORES-200, NTREX-128, TED-2025
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


def download_ted2025():
    """Tải bộ TED-2025 từ 1 file .zip công khai trên Google Drive (không
    nằm trên Hugging Face) bằng gdown, rồi giải nén.

    Quy trình:
      1. Tự tách file ID từ link chia sẻ (dạng ".../file/d/<ID>/view?...")
         bằng regex, rồi gọi gdown.download(id=<ID>, ...) để tải file .zip
         về alignment/ted-2025.zip. Cố tình KHÔNG dùng tham số
         gdown.download(url=..., fuzzy=True) vì "fuzzy" chỉ có ở các bản
         gdown khá mới (>=4.4) -- tự tách ID rồi dùng id=... hoạt động ổn
         định với mọi bản gdown, kể cả bản cũ.
      2. Giải nén PHẲNG (KHÔNG giữ cấu trúc thư mục con trong zip): mọi file
         thật trong zip (bất kể đang nằm ở thư mục con nào) đều được đặt
         TRỰC TIẾP vào alignment/ted-2025/, chỉ giữ lại tên file -- tránh
         hoàn toàn tình trạng lồng thêm thư mục con thừa. Các entry rác do
         macOS tự sinh khi nén (thư mục "__MACOSX/", file resource-fork
         "._xxx", file ".DS_Store") bị loại bỏ, không giải nén. Nếu 2 file
         trong zip trùng tên (do trước đó nằm ở 2 thư mục con khác nhau),
         script tự thêm hậu tố số vào tên để không bị ghi đè mất dữ liệu,
         kèm cảnh báo.
      3. Xoá file .zip tạm sau khi giải nén xong.
    """
    print("\n=== TED-2025 (Google Drive) -> alignment/ted-2025/ ===")

    if gdown is None:
        print("[LỖI] Chưa cài gói 'gdown'. Cài bằng: pip install gdown")
        return

    out_dir = ALIGNMENT_DIR / "ted-2025"
    zip_path = ALIGNMENT_DIR / "ted-2025.zip"

    if not FORCE_REDOWNLOAD and output_dir_has_any_files(out_dir):
        print(f"  [skip] ted-2025 đã có dữ liệu tại {out_dir} -> bỏ qua toàn bộ "
              f"(dùng --force để tải lại).")
        return

    ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)

    file_id = extract_gdrive_file_id(TED2025_GDRIVE_URL)
    print(f"  Đang tải file .zip từ Google Drive (id={file_id}) về {zip_path} ...")
    try:
        gdown.download(id=file_id, output=str(zip_path), quiet=False)
    except Exception as e:
        print(f"[LỖI] Không tải được file TED-2025 từ Google Drive: {e}")
        print("    -> Nếu lỗi liên quan tới gdown quá cũ, thử nâng cấp: "
              "pip install -U gdown")
        return

    if not zip_path.exists():
        print(f"[LỖI] gdown chạy xong nhưng không thấy file {zip_path} -> có thể "
              f"link đã hết hạn/hết quyền chia sẻ công khai.")
        return

    print(f"  Đang giải nén (phẳng, bỏ rác __MACOSX) {zip_path} vào {out_dir} ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [n for n in zf.namelist() if n.strip("/")]

            # Bỏ qua entry thư mục và toàn bộ rác do macOS tự sinh khi nén:
            # thư mục "__MACOSX/", file resource-fork "._xxx", ".DS_Store".
            real_files = [
                n for n in names
                if not n.endswith("/") and not is_macos_zip_junk(n)
            ]

            extracted = 0
            used_names = set()
            for member in tqdm(real_files, desc="  Giải nén ted-2025"):
                # Giải nén PHẲNG: chỉ lấy tên file (Path(...).name), bỏ toàn
                # bộ đường dẫn thư mục con trong zip -- đúng yêu cầu "không
                # phải folder", mọi file nằm thẳng trong alignment/ted-2025/.
                base_name = Path(member).name
                if not base_name:
                    continue
                target_name = base_name
                if target_name in used_names:
                    # Trùng tên (2 thư mục con trong zip cùng có file tên
                    # giống nhau) -> thêm hậu tố số để không ghi đè mất dữ
                    # liệu, đồng thời cảnh báo cho người dùng biết.
                    stem = Path(base_name).stem
                    suffix = Path(base_name).suffix
                    i = 1
                    while target_name in used_names:
                        target_name = f"{stem}__{i}{suffix}"
                        i += 1
                    print(f"    [!] Trùng tên file '{base_name}' (nguồn: {member}) "
                          f"-> đổi thành '{target_name}' để tránh ghi đè.")
                used_names.add(target_name)

                target_path = out_dir / target_name
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted += 1
    except zipfile.BadZipFile:
        print(f"[LỖI] {zip_path} không phải file zip hợp lệ -- có thể Google "
              f"Drive trả về trang lỗi/trang cảnh báo quét virus thay vì file "
              f"thật (thường gặp với file rất lớn). Thử tải thủ công bằng "
              f"trình duyệt để kiểm tra link.")
        return
    except Exception as e:
        print(f"[LỖI] Giải nén thất bại: {e}")
        return

    print(f"  -> Đã giải nén {extracted} file (phẳng, không thư mục con) vào {out_dir}")

    try:
        zip_path.unlink()
        print(f"  -> Đã xoá file .zip tạm {zip_path}")
    except Exception as e:
        print(f"[!] Không xoá được file .zip tạm {zip_path}: {e}")


# --------------------------------------------------------------------------
# 6. DOWNSTREAM: MMMLU, XNLI, XQuAD
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
    # FLORES-200 -> sẽ lỗi "Feature type 'List' not found". Thay vào
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


# --------------------------------------------------------------------------
# 7. ENGLISH TASK: SNLI, SQuAD, MMLU
#    (task tiếng Anh tương ứng XNLI / XQuAD / MMMLU, dùng cho alternate training)
# --------------------------------------------------------------------------
def _download_english_task(display_name: str, folder: str, repo_id: str, config: str):
    """Hàm chung cho các bộ task tiếng Anh: mỗi bộ -> english_task/<folder>/,
    mỗi split 1 file <split>.json. Tự skip nếu thư mục đã có file .json."""
    print(f"\n=== {display_name} -> english_task/{folder}/ ===")
    out_dir = ENGLISH_TASK_DIR / folder
    if not FORCE_REDOWNLOAD and output_already_exists(out_dir):
        print(f"  [skip] {folder} đã có dữ liệu tại {out_dir} -> bỏ qua toàn bộ "
              f"(dùng --force để tải lại).")
        return

    splits = load_splits_with_parquet_fallback(repo_id, config)
    if not splits:
        print(f"[LỖI] Không tải được {repo_id} (config={config}) bằng cả 2 cách.")
        return

    print(f"  Các split có sẵn: {list(splits.keys())} "
          f"({', '.join(f'{s}={len(d)} dòng' for s, d in splits.items())})")

    for split, data in splits.items():
        save_any_split(data, out_dir / f"{split}.json", desc=f"  {folder}/{split}")


def download_snli():
    # LƯU Ý: SNLI gốc có các dòng không có nhãn gold (label = -1, ~1-2% mỗi
    # split). Script giữ nguyên dữ liệu gốc, KHÔNG lọc -- hãy lọc label != -1
    # lúc train/eval nếu cần.
    _download_english_task("SNLI", "snli", "stanfordnlp/snli", "plain_text")


def download_squad():
    # LƯU Ý: rajpurkar/squad là SQuAD v1.1 (train ~87.6k, validation ~10.6k),
    # cùng phiên bản gốc mà XQuAD được dịch ra từ đó. Nếu cần SQuAD v2.0
    # (có câu không trả lời được) thì đổi sang "rajpurkar/squad_v2".
    _download_english_task("SQuAD", "squad", "rajpurkar/squad", "plain_text")


def download_mmlu():
    # LƯU Ý: config "all" gộp 57 subject vào 1 bộ (cột "subject" cho biết
    # môn nào), gồm 4 split: auxiliary_train (~99.8k dòng, tập train phụ gộp từ
    # ARC/RACE/OBQA/...), test (~14k), validation (~1.5k), dev (285, few-shot).
    # auxiliary_train nặng -- đặt MAX_EXAMPLES_PER_SPLIT nếu chỉ cần test nhanh.
    _download_english_task("MMLU", "mmlu", "cais/mmlu", "all")


# --------------------------------------------------------------------------
# 8. MAIN
# --------------------------------------------------------------------------
DATASET_REGISTRY = {
    "flores": download_flores200,
    "ntrex": download_ntrex128,
    "ted2025": download_ted2025,
    "mmmlu": download_mmmlu,
    "xnli": download_xnli,
    "xquad": download_xquad,
    "snli": download_snli,
    "squad": download_squad,
    "mmlu": download_mmlu,
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
    print(f"ENGLISH_TASK_DIR = {ENGLISH_TASK_DIR}")
    print(f"Chế độ: {'FORCE tải lại toàn bộ (ghi đè)' if FORCE_REDOWNLOAD else 'tự động SKIP phần đã có sẵn'}")

    for name in targets:
        DATASET_REGISTRY[name]()

    print("\nHoàn tất.")


if __name__ == "__main__":
    main()