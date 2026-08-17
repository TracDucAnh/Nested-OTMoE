#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_mt.py — Tai bitext song ngu tu 3 NGUON KHAC NHAU tuy cap ngon ngu,
xuat thanh JSON gom DUNG 3 truong: id, <ma ngon ngu 1>, <ma ngon ngu 2>.

NGUON THEO TUNG CAP
    vi-zh   VLSP 2022 (HuggingFace, dataset GATED — can dang nhap + duoc cap
            quyen truy cap): VLSP2023-MT/ViBidirectionMT-Eval, thu muc VLSP2022/
    sw-ar   CCMatrix (qua allenai/nllb tren HuggingFace, nhanh (B) — cap
            (arb_Arab, swh_Latn) khong co trong NLLB_PAIRS nhung co trong
            CCMATRIX_PAIRS). Script tu dong nhan dien nhanh va tai tu statmt.org.
    ht-fr   NLLB mined bitext (nhu truoc, KHONG loc, lay nguyen cau goc)
    fr-wo   NLLB mined bitext (nhu truoc, KHONG loc, lay nguyen cau goc)
    hi-ur   NLLB mined bitext (nhu truoc, KHONG loc, lay nguyen cau goc)

    (JW300 van con ho tro trong ham process_jw300() cho ai muon dung lai voi
    cap khac trong tuong lai, nhung khong con duoc dung cho sw-ar nua.)

LOC DU LIEU
    KHONG loc gi ca cho ca 5 cap — lay nguyen cau goc tu nguon, chi ghep theo
    dong (line-aligned). Neu tong so dong vuot --max-rows (mac dinh 20,000,
    du cho zero-shot eval), script SAMPLE NGAU NHIEN (khong lay N dong dau)
    voi --seed co dinh (mac dinh 42) de tai lap duoc ket qua. Nhanh NLLB
    dung reservoir sampling (mot luot doc toan bo stream); nhanh VLSP2022/
    JW300 dung random.sample tren du lieu da doc het vao bo nho.

===============================================================================
CITATION (NLLB) — cap sw-ar lay tu nhanh CCMatrix trong cung dataset nay
===============================================================================
CCMatrix (Schwenk et al.):
@inproceedings{schwenk-etal-2021-ccmatrix,
  title = {CCMatrix: Mining Billions of High-Quality Parallel Sentences on the Web},
  author = {Schwenk, Holger and Wenzek, Guillaume and Edunov, Sergey and
            Grave, Edouard and Joulin, Armand and Fan, Angela},
  booktitle = {ACL-IJCNLP 2021}, year = {2021},
  url = {https://aclanthology.org/2021.acl-long.507/}
}

@article{article,
author = {Costa-jussa, Marta and Cross, James and Çelebi, Onur and Elbayad, Maha
and Heafield, Kenneth and Heffernan, Kevin and Kalbassi, Elahe and Licht, Daniel
and Maillard, Jean and Sun, Anna and Wang, Skyler and Wenzek, Guillaume and
Youngblood, Al and Akula, Bapi and Barrault, Loïc and Gonzalez, Gabriel and
Hansanti, Prangthip and Hoffman, John and Wang, Jeff},
year = {2024}, month = {06}, title = {Scaling neural machine translation to 200 languages},
volume = {630}, journal = {Nature}, doi = {10.1038/s41586-024-07335-x}
}

===============================================================================
CACH DUNG
===============================================================================
    python download_mt.py                    # ca 5 cap, sample 20K/cap, seed=42
    python download_mt.py vi-zh              # mot cap (vi du: vi-zh)
    python download_mt.py --list             # liet ke cap da dang ky roi thoat
    python download_mt.py --max-rows 20000   # so dong SAMPLE NGAU NHIEN giu lai / cap
    python download_mt.py --seed 42          # seed cho sample ngau nhien
    python download_mt.py --hf-token hf_xxx  # token HF neu chua `huggingface-cli login`
    python download_mt.py --vlsp-split dev   # dung split dev/test thay vi train

DAU RA
    data/mt_translation/{pair}.json
    vi du cap vi-zh: {"id": "vlsp2022_vi-zh_0", "vi": "...", "zh": "..."}

YEU CAU
    pip install requests huggingface_hub
    Voi vi-zh: truy cap https://huggingface.co/datasets/VLSP2023-MT/ViBidirectionMT-Eval,
    dang nhap, bam "Agree" de duoc cap quyen (dataset gated), roi chay
    `huggingface-cli login` (hoac truyen --hf-token) truoc khi chay script nay.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import random
import re
import sys
import zipfile
from typing import Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
load_dotenv()

try:
    import requests
except ImportError:
    sys.exit("Thieu thu vien 'requests'. Chay: pip install requests")



# =============================================================================
# DANG KY CAP NGON NGU — moi cap co "source" rieng
# =============================================================================
PAIRS: Dict[str, dict] = {
    "vi-zh": {"source": "vlsp2022", "short": ("vi", "zh"),
              "hf_repo": "VLSP2023-MT/ViBidirectionMT-Eval", "hf_subdir": "VLSP2022"},
    "ht-fr": {"source": "nllb", "nllb": ("hat_Latn", "fra_Latn"), "short": ("ht", "fr")},
    # sw-ar: da kiem tra nllb_lang_pairs.py (allenai/nllb tren HuggingFace) —
    # cap (arb_Arab, swh_Latn) KHONG co trong NLLB_PAIRS (nhanh mining chinh)
    # nhung CO trong CCMATRIX_PAIRS (nhanh du phong CCMatrix). Vi vay chuyen
    # sang nguon "nllb": ham classify() trong script se tu nhan dien day la
    # nhanh (B)/CCMatrix va tai tu statmt_base (khong can JW300 nua).
    "sw-ar": {"source": "nllb", "nllb": ("arb_Arab", "swh_Latn"), "short": ("sw", "ar")},
    "fr-wo": {"source": "nllb", "nllb": ("fra_Latn", "wol_Latn"), "short": ("fr", "wo")},
    "hi-ur": {"source": "nllb", "nllb": ("hin_Deva", "urd_Arab"), "short": ("hi", "ur")},
}


# =============================================================================
# THAM SO MAC DINH
# =============================================================================
OUTDIR_DEFAULT = os.path.join("data", "mt_translation")
MAX_ROWS_DEFAULT = 10_000   # zero-shot eval: 20K mau/cap la du
SEED_DEFAULT = 42

HF_RAW = "https://huggingface.co/datasets/allenai/nllb/raw/main/"

# QUAN TRONG - LY DO LOI CU:
# Script ban dau doan cung URL "https://object.pouta.csc.fi/OPUS-JW300/v1/moses/..."
# nhung thu muc "v1/moses/" KHONG TON TAI tren server (JW300 ban moses chi co o
# version "v1b", ban da vien lai loi alignment). Ngoai ra, cac bucket cua OPUS
# doi ten version theo tung lan release nen doan cung version la cach lam de gay
# 404 ve sau. Cach dung va ben vung la goi OPUS-API chinh thuc (opus.nlpl.eu),
# API nay tra ve URL tai hien hanh dang JSON - day cung la cach ma cong cu
# opustools/opus_read dung ben trong. Neu API khong goi duoc (vi du bi chan
# mang), moi fallback sang doan URL bucket tinh (v1b roi v1).
OPUS_API_URL = "https://opus.nlpl.eu/opusapi/"
JW300_URL_VERSIONS = ["v1b", "v1"]
JW300_URL_TMPL = "https://object.pouta.csc.fi/OPUS-JW300/{ver}/moses/{a}-{b}.txt.zip"


def query_opus_api(corpus: str, source: str, target: str,
                    preprocessing: str = "moses", version: str = "latest") -> List[str]:
    """Tra OPUS-API de lay (cac) URL tai that su dang co hieu luc cho 1 cap
    ngon ngu. Tra ve danh sach rong neu khong tim thay hoac loi mang."""
    try:
        r = requests.get(OPUS_API_URL, params={
            "corpus": corpus, "source": source, "target": target,
            "preprocessing": preprocessing, "version": version,
        }, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"    [OPUS-API loi: {type(e).__name__}: {e}]")
        return []
    urls = [c["url"] for c in data.get("corpora", []) if c.get("url")]
    if urls:
        print(f"    [OPUS-API] tim thay {len(urls)} URL cho {corpus} {source}-{target}")
    return urls

# Schema nhanh (A) NLLB — doc tu _generate_full_examples trong nllb.py.
A_NCOLS = 9
A_SRC, A_TGT = 0, 1
# Schema nhanh (B) CCMatrix/statmt — (laser_score, src, tgt)
B_NCOLS = 3
B_SRC, B_TGT = 1, 2


def write_output(pair: str, key0: str, key1: str, records: List[dict], args) -> Optional[dict]:
    if not records:
        print("  !! Khong con dong nao. Bo qua.")
        return None
    for i in range(min(3, len(records))):
        print(f"  Vi du {i}: {records[i][key0][:58]!r}")
        print(f"        -> {records[i][key1][:58]!r}")
    out_json = os.path.join(args.outdir, f"{pair}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"  => {out_json}  ({len(records):,} cap)")
    return {"pair": pair, "n": len(records), "keys": (key0, key1)}


# =============================================================================
# NGUON 1: NLLB mined bitext (ht-fr, fr-wo, hi-ur) — KHONG LOC
# =============================================================================
def _fetch(name: str) -> str:
    r = requests.get(HF_RAW + name, timeout=60)
    r.raise_for_status()
    return r.text


def load_registry():
    """Tra ve (allenai_base, statmt_base, NLLB_PAIRS, CCMATRIX_PAIRS, mapping)."""
    script = _fetch("nllb.py")
    m_a = re.search(r'_ALLENAI_URL\s*=\s*"([^"]+)"', script)
    m_s = re.search(r'_STATMT_URL\s*=\s*"([^"]+)"', script)
    if not m_a or not m_s:
        raise RuntimeError("Khong tim thay _ALLENAI_URL / _STATMT_URL trong nllb.py")

    src = _fetch("nllb_lang_pairs.py")

    def block(var: str) -> str:
        m = re.search(rf"^{var}\s*=\s*[\[\{{]", src, re.MULTILINE)
        if not m:
            return ""
        rest = src[m.end():]
        nxt = re.search(r"^[A-Z_]+\s*=", rest, re.MULTILINE)
        return rest[:nxt.start()] if nxt else rest

    def tuples(text: str) -> Set[Tuple[str, str]]:
        return set(re.findall(
            r'\(\s*"([a-z]{3}_[A-Za-z]+)"\s*,\s*"([a-z]{3}_[A-Za-z]+)"\s*\)', text))

    nllb_pairs = tuples(block("NLLB_PAIRS"))
    cc_pairs = tuples(block("CCMATRIX_PAIRS"))
    mapping = dict(re.findall(
        r'"([a-z]{3}_[A-Za-z]+)"\s*:\s*"([^"]+)"', block("CCMATRIX_MAPPING")))
    return m_a.group(1), m_s.group(1), nllb_pairs, cc_pairs, mapping


def classify(cfg: dict, nllb_pairs, cc_pairs):
    """Tra ve ('A'|'B'|'?', thu tu canonical cua cap)."""
    a, b = cfg["nllb"]
    for order in ((a, b), (b, a)):
        if order in nllb_pairs:
            return "A", order
    for order in ((a, b), (b, a)):
        if order in cc_pairs:
            return "B", order
    return "?", None


def open_stream(urls: List[str]):
    """Thu lan luot cac URL, in ro ma HTTP cua tung lan thu."""
    for url in urls:
        try:
            r = requests.get(url, stream=True, timeout=120)
            if r.status_code == 200:
                print(f"    [200] {url}")
                return r, url
            print(f"    [{r.status_code}] {url}")
            r.close()
        except requests.RequestException as e:
            print(f"    [{type(e).__name__}] {url}")
    return None, None


def process_pair_nllb(pair: str, cfg: dict, reg, args) -> Optional[dict]:
    allenai_base, statmt_base, nllb_pairs, cc_pairs, mapping = reg
    print(f"\n{'=' * 70}\n[{pair}] (NLLB, khong loc)\n{'=' * 70}")

    branch, order = classify(cfg, nllb_pairs, cc_pairs)
    if branch == "?":
        print(f"  !! Cap {cfg['nllb'][0]}-{cfg['nllb'][1]} khong co trong "
              f"NLLB_PAIRS lan CCMATRIX_PAIRS. Kiem tra lai ma ngon ngu.")
        return None

    if branch == "A":
        print("  Nhanh (A) NLLB — 9 cot.")
        urls = [f"{allenai_base}{order[0]}-{order[1]}.gz",
                f"{allenai_base}{order[1]}-{order[0]}.gz"]
    else:
        print("  Nhanh (B) CCMatrix/statmt — 3 cot.")
        cc0, cc1 = mapping.get(order[0]), mapping.get(order[1])
        if not cc0 or not cc1:
            print(f"  !! Thieu CCMATRIX_MAPPING cho {order}. Bo qua.")
            return None
        urls = [f"{statmt_base}{cc0}-{cc1}.bitextf.tsv.gz",
                f"{statmt_base}{cc1}-{cc0}.bitextf.tsv.gz"]

    resp, used = open_stream(urls)
    if resp is None:
        print(f"  !! Khong tai duoc cap {pair} (xem ma HTTP ben tren). Bo qua.")
        return None

    fname = used.rsplit("/", 1)[-1]
    if branch == "A":
        col_first = fname.split(".gz")[0].split("-")[0]
    else:
        cc_first = fname.split(".bitextf")[0].split("-")[0]
        inv = {v: k for k, v in mapping.items()}
        col_first = inv.get(cc_first, order[0])

    a_nllb, b_nllb = cfg["nllb"]
    code2short = {a_nllb: cfg["short"][0], b_nllb: cfg["short"][1]}
    other = b_nllb if col_first == a_nllb else a_nllb
    key0, key1 = code2short[col_first], code2short[other]
    print(f"  Cot 0 -> '{key0}' | Cot 1 -> '{key1}'")

    reader = io.TextIOWrapper(gzip.GzipFile(fileobj=resp.raw),
                              encoding="utf-8", errors="replace")

    # Reservoir sampling (thuat toan R) — sample NGAU NHIEN DEU toi da
    # args.max_rows dong tu toan bo stream (khong biet truoc tong so dong),
    # chi doc MOT LUOT, deterministic theo args.seed.
    rng = random.Random(args.seed)
    k = args.max_rows
    reservoir: List[Tuple[str, str]] = []
    n_read = 0
    n_badcol = 0
    n_valid = 0
    ncols = A_NCOLS if branch == "A" else B_NCOLS
    src_idx, tgt_idx = (A_SRC, A_TGT) if branch == "A" else (B_SRC, B_TGT)

    for line in reader:
        n_read += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) != ncols:
            n_badcol += 1
            continue

        # Lay nguyen cau goc — khong normalize, khong loc, khong khu trung lap.
        s0, s1 = parts[src_idx], parts[tgt_idx]
        if n_valid < k:
            reservoir.append((s0, s1))
        else:
            j = rng.randint(0, n_valid)
            if j < k:
                reservoir[j] = (s0, s1)
        n_valid += 1

    resp.close()

    records: List[dict] = [
        {"id": f"nllb_{pair}_{i}", key0: s0, key1: s1}
        for i, (s0, s1) in enumerate(reservoir)
    ]
    print(f"  Doc {n_read:,} dong ({n_valid:,} dong hop le) -> sample ngau nhien "
          f"(seed={args.seed}) giu {len(records):,}"
          + (f" (bo {n_badcol:,} dong sai so cot)" if n_badcol else ""))

    return write_output(pair, key0, key1, records, args)


# =============================================================================
# NGUON 2: VLSP 2022 (vi-zh) — HuggingFace dataset GATED — CO LOC
# =============================================================================
def process_vlsp(pair: str, cfg: dict, args) -> Optional[dict]:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        sys.exit("Thieu thu vien 'huggingface_hub'. Chay: pip install huggingface_hub")

    lang0, lang1 = cfg["short"]
    repo_id, subdir = cfg["hf_repo"], cfg["hf_subdir"]
    print(f"\n{'=' * 70}\n[{pair}] (VLSP2022, HuggingFace: {repo_id}, co loc)\n{'=' * 70}")

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    api = HfApi(token=token)
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"  !! Khong liet ke duoc file trong repo '{repo_id}': {type(e).__name__}: {e}")
        print("     Dataset nay bi GATED. Vao trang HuggingFace cua dataset, dang nhap,")
        print("     bam 'Agree' de xin quyen truy cap, roi chay `huggingface-cli login`")
        print("     (hoac truyen --hf-token hf_xxx) truoc khi chay lai script.")
        return None

    cand = [f for f in files if f.startswith(subdir + "/")]
    if not cand:
        print(f"  !! Khong tim thay file nao trong thu muc '{subdir}/' cua repo.")
        print(f"     Danh sach file trong repo: {files}")
        return None

    split = args.vlsp_split.lower()

    def pick(lang: str) -> Optional[str]:
        exact = [f for f in cand if f.lower().endswith(f".{lang}") and split in f.lower()]
        if exact:
            return exact[0]
        loose = [f for f in cand if f.lower().endswith(f".{lang}")]
        return loose[0] if loose else None

    f0, f1 = pick(lang0), pick(lang1)
    if not f0 or not f1:
        print(f"  !! Khong xac dinh duoc file cho '.{lang0}' / '.{lang1}' (split='{split}').")
        print(f"     File tim thay trong '{subdir}/': {cand}")
        return None
    print(f"  File: {f0}  <->  {f1}")

    try:
        p0 = hf_hub_download(repo_id=repo_id, filename=f0, repo_type="dataset", token=token)
        p1 = hf_hub_download(repo_id=repo_id, filename=f1, repo_type="dataset", token=token)
    except Exception as e:
        print(f"  !! Loi tai file tu HuggingFace: {type(e).__name__}: {e}")
        return None

    with open(p0, encoding="utf-8", errors="replace") as fh0:
        lines0 = [ln.rstrip("\n") for ln in fh0]
    with open(p1, encoding="utf-8", errors="replace") as fh1:
        lines1 = [ln.rstrip("\n") for ln in fh1]

    if len(lines0) != len(lines1):
        print(f"  !! So dong khong khop: {lang0}={len(lines0):,} vs "
              f"{lang1}={len(lines1):,}. Cat theo min.")
    n = min(len(lines0), len(lines1))

    # Sample NGAU NHIEN toi da args.max_rows dong (khong con lay N dong dau),
    # deterministic theo args.seed.
    rng = random.Random(args.seed)
    if n > args.max_rows:
        idxs: List[int] = sorted(rng.sample(range(n), args.max_rows))
    else:
        idxs = list(range(n))

    records: List[dict] = []
    for i in idxs:
        s0, s1 = lines0[i].strip(), lines1[i].strip()
        rid = f"vlsp2022_{pair}_{len(records)}"
        records.append({"id": rid, lang0: s0, lang1: s1})

    print(f"  Doc {len(lines0):,} dong -> sample ngau nhien (seed={args.seed}) "
          f"giu {len(records):,} (khong loc)")
    return write_output(pair, lang0, lang1, records, args)


# =============================================================================
# NGUON 3: JW300 (sw-ar) — OPUS, tai truc tiep — CO LOC
# =============================================================================
def process_jw300(pair: str, cfg: dict, args) -> Optional[dict]:
    lang0, lang1 = cfg["short"]
    print(f"\n{'=' * 70}\n[{pair}] (JW300/OPUS, co loc)\n{'=' * 70}")

    # 1) Uu tien: tra URL tai hien hanh qua OPUS-API chinh thuc (khong doan version).
    print("  Dang tra OPUS-API de lay URL tai hien hanh...")
    urls = query_opus_api("JW300", lang0, lang1, preprocessing="moses", version="latest")
    urls += query_opus_api("JW300", lang1, lang0, preprocessing="moses", version="latest")
    # 2) Du phong: doan cac URL bucket tinh (v1b roi v1) phong khi API khong goi duoc.
    for ver in JW300_URL_VERSIONS:
        urls.append(JW300_URL_TMPL.format(ver=ver, a=lang0, b=lang1))
        urls.append(JW300_URL_TMPL.format(ver=ver, a=lang1, b=lang0))
    # Khu trung lap nhung van giu thu tu uu tien.
    seen: Set[str] = set()
    urls = [u for u in urls if not (u in seen or seen.add(u))]
    resp, used = open_stream(urls)
    if resp is None:
        print(f"  !! Khong tai duoc JW300 cho {pair} (xem ma HTTP ben tren). Bo qua.")
        return None

    data = resp.content
    resp.close()

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        print("  !! File tai ve khong phai zip hop le.")
        return None

    names = zf.namelist()

    def find(lang: str) -> Optional[str]:
        cands = [n for n in names if n.lower().endswith(f".{lang}")]
        return cands[0] if cands else None

    n0, n1 = find(lang0), find(lang1)
    if not n0 or not n1:
        print(f"  !! Khong tim thay file '.{lang0}' / '.{lang1}' trong zip.")
        print(f"     File co trong zip: {names}")
        return None
    print(f"  File trong zip: {n0}  <->  {n1}")

    lines0 = zf.read(n0).decode("utf-8", errors="replace").splitlines()
    lines1 = zf.read(n1).decode("utf-8", errors="replace").splitlines()

    if len(lines0) != len(lines1):
        print(f"  !! So dong khong khop: {lang0}={len(lines0):,} vs "
              f"{lang1}={len(lines1):,}. Cat theo min.")
    n = min(len(lines0), len(lines1))

    # Sample NGAU NHIEN toi da args.max_rows dong (khong con lay N dong dau),
    # deterministic theo args.seed.
    rng = random.Random(args.seed)
    if n > args.max_rows:
        idxs: List[int] = sorted(rng.sample(range(n), args.max_rows))
    else:
        idxs = list(range(n))

    records: List[dict] = []
    for i in idxs:
        s0, s1 = lines0[i].strip(), lines1[i].strip()
        rid = f"jw300_{pair}_{len(records)}"
        records.append({"id": rid, lang0: s0, lang1: s1})

    print(f"  Doc {len(lines0):,} dong -> sample ngau nhien (seed={args.seed}) "
          f"giu {len(records):,} (khong loc)")
    return write_output(pair, lang0, lang1, records, args)


# =============================================================================
# MAIN
# =============================================================================
def main() -> int:
    p = argparse.ArgumentParser(
        description="Tai bitext cho 5 cap ngon ngu tu 3 nguon khac nhau (VLSP2022, "
                     "JW300, NLLB) thanh JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Vi du:\n"
               "  python download_mt.py                  # 20K mau/cap, seed=42\n"
               "  python download_mt.py vi-zh\n"
               "  python download_mt.py --max-rows 20000 --seed 42\n"
               "  python download_mt.py --hf-token hf_xxx\n")
    p.add_argument("pairs", nargs="*", help="cap can xu ly (mac dinh: tat ca)")
    p.add_argument("--list", action="store_true",
                   help="liet ke cap da dang ky (kem nguon) roi thoat")
    p.add_argument("--outdir", default=OUTDIR_DEFAULT)
    p.add_argument("--max-rows", type=int, default=MAX_ROWS_DEFAULT,
                   help=f"so dong SAMPLE NGAU NHIEN giu lai cho moi cap "
                        f"(mac dinh {MAX_ROWS_DEFAULT:,})")
    p.add_argument("--seed", type=int, default=SEED_DEFAULT,
                   help=f"seed cho sample ngau nhien, de tai lap ket qua "
                        f"(mac dinh {SEED_DEFAULT})")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token cho dataset gated VLSP2022 (hoac dung "
                        "`huggingface-cli login` / bien moi truong HF_TOKEN truoc)")
    p.add_argument("--vlsp-split", default="train",
                   help="split VLSP2022 can lay: train/dev/test (mac dinh: train)")
    args = p.parse_args()

    selected = args.pairs or list(PAIRS.keys())
    unknown = [x for x in selected if x not in PAIRS]
    if unknown:
        print(f"Cap khong ton tai: {unknown}", file=sys.stderr)
        print(f"Cap hop le: {list(PAIRS.keys())}", file=sys.stderr)
        return 1

    need_nllb_registry = any(PAIRS[x]["source"] == "nllb" for x in selected)
    reg = None
    if need_nllb_registry:
        try:
            reg = load_registry()
        except Exception as e:
            print(f"Khong doc duoc registry NLLB: {type(e).__name__}: {e}", file=sys.stderr)
            return 1

    if args.list:
        print(f"{'CAP':<8}{'NGUON':<12}CHI TIET")
        for k, v in PAIRS.items():
            src = v["source"]
            if src == "nllb":
                branch, _ = classify(v, reg[2], reg[3]) if reg else ("?", None)
                detail = f"NLLB nhanh {branch} ({v['nllb'][0]}-{v['nllb'][1]})"
            elif src == "vlsp2022":
                detail = f"HuggingFace: {v['hf_repo']} ({v['hf_subdir']}/, gated)"
            else:
                detail = "OPUS JW300 (object.pouta.csc.fi)"
            print(f"{k:<8}{src:<12}{detail}")
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Thu muc dau ra: {os.path.abspath(args.outdir)}")
    print(f"max_rows={args.max_rows:,} (gioi han moi cap)")
    print(f"Se xu ly {len(selected)} cap: {selected}")

    summary = []
    for pair in selected:
        cfg = PAIRS[pair]
        try:
            if cfg["source"] == "nllb":
                res = process_pair_nllb(pair, cfg, reg, args)
            elif cfg["source"] == "vlsp2022":
                res = process_vlsp(pair, cfg, args)
            elif cfg["source"] == "jw300":
                res = process_jw300(pair, cfg, args)
            else:
                res = None
            if res:
                res["source"] = cfg["source"]
                summary.append(res)
        except KeyboardInterrupt:
            print("\nDung theo yeu cau nguoi dung.")
            return 130
        except Exception as e:
            print(f"  !! Loi khi xu ly {pair}: {type(e).__name__}: {e}")

    print(f"\n{'=' * 70}\nTONG KET\n{'=' * 70}")
    print(f"{'CAP':<10}{'NGUON':<12}{'TRUONG':<10}{'SO CAP':>12}")
    print("-" * 50)
    for s in summary:
        print(f"{s['pair']:<10}{s['source']:<12}"
              f"{s['keys'][0] + '/' + s['keys'][1]:<10}{s['n']:>12,}")
    print("-" * 50)
    print(f"Thanh cong {len(summary)}/{len(selected)} cap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())