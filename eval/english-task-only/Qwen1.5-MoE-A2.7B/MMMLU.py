"""
eval/english-task-only/Qwen1.5-MoE-A2.7B/MMMLU.py

Zero-shot cross-lingual MMMLU (Multilingual MMLU) evaluation cho model
Qwen1.5-MoE-A2.7B-MMLU-Task-Only.

Boi canh
--------
Model KHONG duoc finetune tren MMMLU. Model chi duoc LoRA-finetune tren MMLU (tieng Anh),
theo dung format prompt/answer trong `training/english-task-only/Qwen1.5-MoE-A2.7-MMLU.py`
(xem lai file do de doi chieu):

    The following are multiple choice questions (with answers) about <subject>.

    <question>
    A. <choice A>
    B. <choice B>
    C. <choice C>
    D. <choice D>
    Answer: <letter><eos>

Checkpoint duoc push len HF Hub CHI la 1 LoRA adapter (`model.save_pretrained()` cua mot
PeftModel), KHONG phai full model da merge — dung y het `save_checkpoint()` / `push_to_hub()`
trong file training (docstring cua file training con noi ro cach load lai: base model +
`PeftModel.from_pretrained(base, "ducanhdinh/Qwen1.5-MoE-A2.7B-MMLU-Task-Only")`). Vi vay de
eval, script nay:

    1) Load base model goc (Qwen/Qwen1.5-MoE-A2.7B) tu HF.
    2) Load LoRA adapter tu repo checkpoint (--adapter_repo_id) bang PeftModel.from_pretrained.
    3) (Mac dinh) merge_and_unload() de suy luan nhanh hon, giong nhu 1 model thuong.

Eval la ZERO-SHOT CROSS-LINGUAL: giu NGUYEN prompt/header tieng Anh dung luc train MMLU, CHI
thay question/4 choice bang van ban cua tung ngon ngu MMMLU (AR_XY, DE_DE, ZH_CN, ...) — model
chua tung thay cac ngon ngu nay luc train.

KHAC BIET DUY NHAT so voi `MMMLU_eval.py` mau (vanilla, chua finetune):
  - `build_prompt()` o day THEM lai dong header "The following are multiple choice questions
    (with answers) [about <subject>].\n\n" truoc cau hoi — dung HET format cua `build_prompt()`
    trong file training (bao gom ca truong hop subject rong thi bo "about ..."). Ban vanilla
    KHONG co header nay vi model goc chua finetune khong can. Model finetune roi thi da hoc
    sinh "<letter><eos>" ngay sau "Answer:" theo dung prompt CO header nay, nen giu prompt luc
    eval GIONG luc train se phan anh dung kha nang zero-shot-transfer (MMLU -> MMMLU) cua model,
    thay vi danh gia mot prompt format ma model chua tung thay.
  - Candidate scoring " A"/" B"/" C"/" D" GIU NGUYEN nhu ban vanilla — day cung CHINH XAC la
    format label ma model da duoc train (build_full_text: f"{prompt} {letter}{eos}").
  - Toan bo phan con lai (doc test.json theo tung ngon ngu, OOM-safe dynamic batching
    chia-doi-khi-OOM, resume/checkpoint qua mmmlu_results.json, bao cao theo subject) GIU
    NGUYEN 100% logic cua ban mau, vi day la co che ha tang khong phu thuoc vao model.

Du lieu: chi doc `test.json` trong tung thu muc ngon ngu duoi `data/downstream/mmmlu/<LANG>/`
(vi du AR_XY, DE_DE, ZH_CN, ...), voi cac truong "Question", "A", "B", "C", "D", "Answer",
"Subject" — dung cau truc nhu MMMLU_eval.py mau.

Cach chay
---------
    python eval/english-task-only/Qwen1.5-MoE-A2.7B/MMMLU.py \
        --base_model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --adapter_repo_id ducanhdinh/Qwen1.5-MoE-A2.7B-MMLU-Task-Only \
        --data_root data/downstream/mmmlu \
        --batch_size 8 \
        --output_dir eval/english-task-only/Qwen1.5-MoE-A2.7B/results/mmmlu

    # Chi vai ngon ngu, debug nhanh:
    python eval/english-task-only/Qwen1.5-MoE-A2.7B/MMMLU.py --languages AR_XY VI_VN --max_examples 20

Resume: chay lai voi cung --output_dir se tu dong bo qua cac ngon ngu da co ket qua trong
mmmlu_results.json (dung --overwrite de eval lai tu dau).

Ghi chu
-------
- Neu repo adapter la private, truyen --hf_token hoac set bien moi truong HF_TOKEN /
  HUGGINGFACE_HUB_TOKEN / HUGGING_FACE_HUB_TOKEN.
- Can GPU du VRAM de hold base model Qwen1.5-MoE-A2.7B (~14.3B tong tham so, ~2.7B active/
  token) o bf16 (~28GB).
- --no_merge_adapter de giu adapter tach rieng (PeftModel) thay vi merge_and_unload().
"""

import argparse
import gc
import json
import os
import tempfile

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

LETTERS = ["A", "B", "C", "D"]
CANDIDATES = [" A", " B", " C", " D"]  # index-aligned voi LETTERS; khop CHINH XAC label_word luc train

_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_hf_token(cli_token):
    if cli_token:
        return cli_token
    for var in _HF_TOKEN_ENV_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return None


# --------------------------------------------------------------------------
# OOM-safe dynamic batching helpers (giong het ban mau)
# --------------------------------------------------------------------------
def is_oom_error(err: BaseException) -> bool:
    """True neu `err` giong loi CUDA / CPU out-of-memory."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(err, oom_cls):
        return True
    if not isinstance(err, RuntimeError):
        return False
    msg = str(err).lower()
    return any(
        s in msg
        for s in (
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "not enough memory",
        )
    )


def clear_memory():
    """Best-effort giai phong GPU/CPU memory truoc batch tiep theo."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class DynamicBatcher:
    """Giu bound batch-size cho 1 lan chay.

    Moi chunk du lieu ngoai cung LUON bat dau lai tu `initial_batch_size` -- 1 lan OOM tren 1
    chunk KHONG lam giam kich thuoc bat dau cua chunk TIEP THEO. Viec chia doi khi OOM (toi
    `min_batch_size`) chi xay ra BEN TRONG 1 chunk (xem `score_chunk_with_oom_retry`) va bi bo
    di ngay khi chunk do xong; khong lan sang cac chunk sau."""

    def __init__(self, initial_batch_size: int, min_batch_size: int = 1):
        self.initial_batch_size = max(1, initial_batch_size)
        self.min_batch_size = max(1, min_batch_size)


def score_chunk_with_oom_retry(model, tokenizer, chunk, build_prompt_fn, device, min_batch_size=1):
    """Score `chunk` (list example), chia doi de quy khi OOM.

    Tra ve list (example, pred_index_or_None) khop voi `chunk`. `pred_index` chi la None khi
    ngay ca 1 example don le cung khong score duoc (OOM lien tuc o batch_size=1) -- example do
    bi bo qua thay vi lam crash toan bo lan chay."""
    if not chunk:
        return []

    prompts = [build_prompt_fn(ex) for ex in chunk]

    # Bat exception, ghi nhan, roi de scope `except` ket thuc (Python tu clear `e`/traceback tai
    # do) TRUOC khi goi clear_memory() + de quy -- de memory duoc giai phong THUC SU truoc moi
    # lan retry (xem giai thich chi tiet trong MMMLU_eval.py mau).
    oom = False
    try:
        scores = score_candidates_batch(model, tokenizer, prompts, device)
    except RuntimeError as e:
        if not is_oom_error(e):
            raise
        oom = True

    if not oom:
        preds = scores.argmax(axis=1)
        return list(zip(chunk, preds))

    clear_memory()

    if len(chunk) <= min_batch_size:
        q_preview = str(chunk[0].get("Question", ""))[:80]
        print(f"[OOM][WARN] batch_size=1 van OOM, bo qua example: {q_preview!r}")
        return [(ex, None) for ex in chunk]

    new_size = max(min_batch_size, len(chunk) // 2)
    print(
        f"[OOM] batch_size={len(chunk)} that bai -> chia doi thanh {new_size} va retry "
        f"(chunk ngoai cung tiep theo van bat dau lai tu --batch_size day du)"
    )
    mid = len(chunk) // 2
    left = score_chunk_with_oom_retry(model, tokenizer, chunk[:mid], build_prompt_fn, device, min_batch_size)
    clear_memory()
    right = score_chunk_with_oom_retry(model, tokenizer, chunk[mid:], build_prompt_fn, device, min_batch_size)
    return left + right


def build_prompt(ex: dict) -> str:
    """GIONG HET build_prompt() trong training/english-task-only/Qwen1.5-MoE-A2.7-MMLU.py: co
    them dong header ve subject (neu co) truoc cau hoi. Day chinh la format model da duoc SFT
    de sinh "<letter><eos>" ngay sau "Answer:", nen phai giu dung khi eval zero-shot tren MMMLU."""
    subject = str(ex.get("Subject") or "").strip()
    if subject:
        header = (
            f"The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n\n"
        )
    else:
        header = "The following are multiple choice questions (with answers).\n\n"
    choice_lines = "\n".join(f"{letter}. {ex[letter]}" for letter in LETTERS)
    return f"{header}{ex['Question']}\n{choice_lines}\nAnswer:"


def load_test_data(lang_dir: str):
    path = os.path.join(lang_dir, "test.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "examples", "rows", "test"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Khong nhan dang duoc cau truc test.json tai {path}")
    return data


@torch.no_grad()
def score_candidates_batch(model, tokenizer, prompts, device):
    """Log-likelihood length-normalized cua tung candidate cho 1 batch prompt.
    Tra ve numpy array shape (len(prompts), len(CANDIDATES))."""
    b = len(prompts)
    num_cand = len(CANDIDATES)

    all_texts = []
    context_lens = []
    for p in prompts:
        ctx_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
        for cand in CANDIDATES:
            all_texts.append(p + cand)
            context_lens.append(len(ctx_ids))

    enc = tokenizer(all_texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)

    seq_lens = attention_mask.sum(dim=1).tolist()
    n = input_ids.shape[0]
    scores = torch.empty(n, dtype=torch.float32)

    for i in range(n):
        ctx_len = context_lens[i]
        real_len = int(seq_lens[i])
        if real_len <= ctx_len:
            scores[i] = float("-inf")
            continue
        token_ids = input_ids[i, ctx_len:real_len]
        pred_log_probs = log_probs[i, ctx_len - 1 : real_len - 1, :]
        gathered = pred_log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        scores[i] = gathered.mean().item()

    return scores.view(b, num_cand).numpy()


# --------------------------------------------------------------------------
# Resume / checkpoint helpers (giong het ban mau)
# --------------------------------------------------------------------------
def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Ghi `data` vao `path` mot cach atomic: ghi ra file tam trong cung thu muc, roi
    os.replace() vao vi tri thuc -- de crash/kill giua luc ghi khong bao gio de lai file
    ket qua bi hong (quan trong vi file nay cung la resume checkpoint)."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_mmmlu_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_write_json(path: str, obj) -> None:
    _atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def atomic_write_csv(path: str, df: pd.DataFrame) -> None:
    _atomic_write_bytes(path, df.to_csv(index=False).encode("utf-8"))


def compute_subject_breakdown(records):
    """Gop cac record cua 1 ngon ngu thanh {subject: {"correct", "total"}}.

    `total` bao gom ca example bi skip (OOM lien tuc), tinh la sai -- khop dung semantics cua
    CSV subject (trung binh boolean `correct` tren tat ca record, ke ca skip)."""
    breakdown = {}
    for r in records:
        subj = r.get("subject") or "unknown"
        entry = breakdown.setdefault(subj, {"correct": 0, "total": 0})
        entry["total"] += 1
        if r.get("correct"):
            entry["correct"] += 1
    return breakdown


def merge_subject_breakdown(subject_totals: dict, breakdown: dict) -> None:
    """Gop in-place subject breakdown cua 1 ngon ngu vao tong chay qua tat ca ngon ngu, dung de
    xay mmmlu_subject_results.csv."""
    for subj, stats in breakdown.items():
        entry = subject_totals.setdefault(subj, {"correct": 0, "total": 0})
        entry["correct"] += stats["correct"]
        entry["total"] += stats["total"]


def load_existing_results(json_path: str):
    """Doc mmmlu_results.json cua lan chay truoc (neu co) de resume.

    Tra ve (results, subject_totals, langs_missing_breakdown):
      - results: {lang: {...}} cho tung ngon ngu da eval xong o lan chay truoc -- se bi SKIP
        o lan chay nay.
      - subject_totals: {subject: {"correct", "total"}} gop tren tat ca ngon ngu da xong do.
      - langs_missing_breakdown: ngon ngu tim thay trong file nhung tu ban script cu, khong co
        field "subject_breakdown". Van tinh la xong va bi skip, nhung khong gop vao
        mmmlu_subject_results.csv duoc vi record goc da mat."""
    if not os.path.exists(json_path):
        return {}, {}, []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Khong doc duoc file ket qua cu {json_path} ({e}); bat dau lai tu dau.")
        return {}, {}, []

    results = {}
    subject_totals = {}
    langs_missing_breakdown = []
    for lang, stats in prev.get("per_language", {}).items():
        results[lang] = dict(stats)
        breakdown = stats.get("subject_breakdown")
        if breakdown:
            merge_subject_breakdown(subject_totals, breakdown)
        else:
            langs_missing_breakdown.append(lang)
    return results, subject_totals, langs_missing_breakdown


def write_all_outputs(output_dir, model_desc, results, subject_totals, total_skipped):
    """(Ghi lai) mmmlu_results.csv, mmmlu_subject_results.csv, mmmlu_results.json tu state
    trong memory hien tai. Goi sau MOI ngon ngu (khong chi 1 lan cuoi) de crash khong bao gio
    mat qua 1 ngon ngu dang chay, va lan chay sau co the resume bang doc mmmlu_results.json."""
    overall_correct = sum(r["accuracy"] * r["n_examples"] for r in results.values())
    overall_total = sum(r["n_examples"] for r in results.values())
    overall_micro_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    macro_acc = sum(r["accuracy"] for r in results.values()) / len(results) if results else 0.0

    lang_rows = [
        {"language": lang, "accuracy": r["accuracy"], "n_examples": r["n_examples"], "n_skipped": r["n_skipped"]}
        for lang, r in results.items()
    ]
    lang_df = pd.DataFrame(lang_rows, columns=["language", "accuracy", "n_examples", "n_skipped"])
    if not lang_df.empty:
        lang_df = lang_df.sort_values("language")
    csv_path = os.path.join(output_dir, "mmmlu_results.csv")
    atomic_write_csv(csv_path, lang_df)

    subj_csv_path = os.path.join(output_dir, "mmmlu_subject_results.csv")
    if subject_totals:
        subj_rows = [
            {
                "subject": subj,
                "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
                "n_examples": stats["total"],
            }
            for subj, stats in subject_totals.items()
        ]
        subj_df = pd.DataFrame(subj_rows).sort_values("subject")
        atomic_write_csv(subj_csv_path, subj_df)

    summary = {
        "model": model_desc,
        "per_language": results,
        "overall_micro_accuracy": overall_micro_acc,
        "macro_average_accuracy": macro_acc,
        "total_skipped": total_skipped,
    }
    json_path = os.path.join(output_dir, "mmmlu_results.json")
    atomic_write_json(json_path, summary)

    return overall_micro_acc, macro_acc, csv_path, subj_csv_path, json_path


def evaluate_language(model, tokenizer, lang, data_root, device, batch_size, max_examples=None, min_batch_size=1):
    lang_dir = os.path.join(data_root, lang)
    data = load_test_data(lang_dir)
    if max_examples is not None:
        data = data[:max_examples]

    correct, total, skipped = 0, 0, 0
    records = []
    batcher = DynamicBatcher(initial_batch_size=batch_size, min_batch_size=min_batch_size)

    pbar = tqdm(total=len(data), desc=f"MMMLU[{lang}]")
    idx = 0
    while idx < len(data):
        cur_bs = batcher.initial_batch_size
        chunk = data[idx : idx + cur_bs]

        pair_results = score_chunk_with_oom_retry(
            model, tokenizer, chunk, build_prompt, device, min_batch_size
        )

        for ex, pred in pair_results:
            gold_letter = str(ex["Answer"]).strip()
            total += 1
            if pred is None:
                skipped += 1
                records.append(
                    {
                        "subject": ex.get("Subject", ""),
                        "question": ex["Question"],
                        "gold": gold_letter,
                        "pred": None,
                        "correct": False,
                        "skipped": True,
                    }
                )
                continue
            pred_letter = LETTERS[int(pred)]
            is_correct = pred_letter == gold_letter
            correct += is_correct
            records.append(
                {
                    "subject": ex.get("Subject", ""),
                    "question": ex["Question"],
                    "gold": gold_letter,
                    "pred": pred_letter,
                    "correct": is_correct,
                    "skipped": False,
                }
            )

        idx += len(chunk)
        pbar.update(len(chunk))
        clear_memory()
    pbar.close()

    if skipped:
        print(f"[{lang}] WARNING: {skipped} example bi skip vi OOM lien tuc tai batch_size=1.")

    acc = correct / total if total > 0 else 0.0
    subject_breakdown = compute_subject_breakdown(records)
    return acc, total, skipped, records, subject_breakdown


def main():
    parser = argparse.ArgumentParser(description="Zero-shot MMMLU eval cho Qwen1.5-MoE-A2.7B-MMLU-Task-Only (LoRA)")

    # Model: base model goc + LoRA adapter checkpoint da push len HF Hub luc train MMLU.
    parser.add_argument("--base_model_name_or_path", default="Qwen/Qwen1.5-MoE-A2.7B",
                         help="Model goc (PHAI khop voi model dung luc train LoRA).")
    parser.add_argument("--adapter_repo_id", default="ducanhdinh/Qwen1.5-MoE-A2.7B-MMLU-Task-Only",
                         help="HF Hub repo (hoac duong dan local) chua LoRA adapter da push luc train MMLU.")
    parser.add_argument("--adapter_revision", default=None,
                         help="Branch/tag/commit cu the cua adapter repo, None = mac dinh (main).")
    parser.add_argument("--merge_adapter", action="store_true", default=True,
                         help="Merge LoRA vao base weights truoc khi eval (nhanh hon, mac dinh True).")
    parser.add_argument("--no_merge_adapter", dest="merge_adapter", action="store_false",
                         help="Giu adapter tach rieng (PeftModel) thay vi merge_and_unload().")
    parser.add_argument("--hf_token", default=None, help="HF token (repo private); mac dinh doc tu bien moi truong.")

    # Data / eval
    parser.add_argument("--data_root", default="data/downstream/mmmlu")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=8,
                         help="Batch size moi chunk bat dau; chia doi rieng cho chunk do khi OOM.")
    parser.add_argument("--min_batch_size", type=int, default=1,
                         help="Khong chia batch nho hon gia tri nay truoc khi skip example.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", default="eval/english-task-only/Qwen1.5-MoE-A2.7B/results/mmmlu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Bo qua mmmlu_results.json cu (neu co) va eval lai tat ca ngon ngu tu dau (mac "
             "dinh: ngon ngu da co trong file do se duoc skip, coi la da xong).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hf_token = load_hf_token(args.hf_token)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    json_path = os.path.join(args.output_dir, "mmmlu_results.json")

    # ------------------------------------------------------------------
    # Resume: mmmlu_results.json cua --output_dir nay dong thoi la checkpoint.
    # ------------------------------------------------------------------
    if args.overwrite:
        results, subject_totals, langs_missing_breakdown = {}, {}, []
    else:
        results, subject_totals, langs_missing_breakdown = load_existing_results(json_path)

    if results:
        print(f"Tim thay ket qua da co cho {len(results)} ngon ngu trong {json_path}.")
    if langs_missing_breakdown:
        print(
            f"[WARN] {len(langs_missing_breakdown)} trong so do duoc luu boi phien ban script cu, "
            f"khong co subject breakdown ({langs_missing_breakdown}); van bi skip, nhung se KHONG "
            f"gop vao mmmlu_subject_results.csv. Dung --overwrite neu can tinh lai."
        )

    total_skipped = sum(r.get("n_skipped", 0) for r in results.values())

    languages = args.languages or sorted(
        d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))
    )

    already_done = [lang for lang in languages if lang in results]
    to_run = [lang for lang in languages if lang not in results]

    if already_done:
        print(f"Skip {len(already_done)} ngon ngu da xong: {already_done}")
    if to_run:
        print(f"Ngon ngu can eval ({len(to_run)}): {to_run}")
    else:
        print("Khong con gi de eval -- tat ca ngon ngu yeu cau deu da co ket qua.")

    # Chi load model (nang) khi thuc su con viec phai lam.
    if to_run:
        print(f"Loading tokenizer + base model: {args.base_model_name_or_path} (device={device}, dtype={args.dtype})")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # bat buoc cho logic slicing log-likelihood o tren

        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype_map[args.dtype],
            device_map="auto" if device == "cuda" else None,
            token=hf_token,
        )
        if device == "cpu":
            base_model.to(device)

        print(f"Loading LoRA adapter checkpoint: {args.adapter_repo_id}"
              f"{f' (revision={args.adapter_revision})' if args.adapter_revision else ''}")
        model = PeftModel.from_pretrained(
            base_model,
            args.adapter_repo_id,
            revision=args.adapter_revision,
            token=hf_token,
            is_trainable=False,
        )
        if args.merge_adapter:
            print("Merging LoRA adapter vao base weights (merge_and_unload) ...")
            model = model.merge_and_unload()
        model.eval()

    model_desc = f"{args.base_model_name_or_path} + LoRA[{args.adapter_repo_id}]"

    for lang in to_run:
        acc, total, skipped, records, subject_breakdown = evaluate_language(
            model, tokenizer, lang, args.data_root, device, args.batch_size, args.max_examples, args.min_batch_size
        )
        results[lang] = {
            "accuracy": acc,
            "n_examples": total,
            "n_skipped": skipped,
            "subject_breakdown": subject_breakdown,
        }
        total_skipped += skipped
        print(f"[{lang}] accuracy = {acc:.4f}  ({total} examples, {skipped} skipped)")

        merge_subject_breakdown(subject_totals, subject_breakdown)

        if args.save_predictions:
            for r in records:
                r["language"] = lang
            atomic_write_csv(
                os.path.join(args.output_dir, f"mmmlu_predictions_{lang}.csv"), pd.DataFrame(records)
            )

        _, _, _, _, checkpoint_path = write_all_outputs(
            args.output_dir, model_desc, results, subject_totals, total_skipped
        )
        print(f"  -> checkpointed to {checkpoint_path}")

        clear_memory()

    overall_micro_acc, macro_acc, csv_path, subj_csv_path, json_path = write_all_outputs(
        args.output_dir, model_desc, results, subject_totals, total_skipped
    )

    print("=" * 60)
    print(f"Overall (micro, weighted by #examples) accuracy: {overall_micro_acc:.4f}")
    print(f"Macro-average (mean over languages) accuracy:    {macro_acc:.4f}")
    if total_skipped:
        print(f"Total skipped examples (persistent OOM): {total_skipped}")

    print(f"\nSaved: {csv_path}")
    if subject_totals:
        print(f"Saved: {subj_csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()