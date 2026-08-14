"""
Zero-shot MMMLU (Multilingual MMLU) evaluation for Qwen1.5-MoE-A2.7B via
log-likelihood scoring.

Method
------
Standard MMLU zero-shot prompt:

    {Question}
    A. {A}
    B. {B}
    C. {C}
    D. {D}
    Answer:

We score the 4 single-letter continuations " A", " B", " C", " D" with
teacher-forced log-likelihood (no free generation) and take the argmax as
the prediction, compared against the gold `Answer` field.

Only `test.json` is used for every language folder under
`data/downstream/mmmlu/<LANG>/` (e.g. AR_XY, DE_DE, ZH_CN, ...).

OOM-safe dynamic batching
--------------------------
Batches are processed with `--batch_size` as the *starting* size. If a batch
raises a CUDA/CPU out-of-memory error, we:
  1. free whatever memory we can (gc.collect + torch.cuda.empty_cache),
  2. split the offending batch in half and retry each half recursively
     (halving again if needed), and
  3. remember the smaller size so future batches for this run start smaller
     too (it will slowly grow back after a streak of OOM-free batches).
If a single example (batch size 1) still OOMs, that example is skipped
(logged and marked in the output) instead of crashing the whole run.

Usage
-----
    python MMMLU_eval.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --data_root data/downstream/mmmlu \
        --batch_size 8 \
        --output_dir eval/Qwen1.5-MoE-A2.7B/results

Besides per-language + overall accuracy (same convention as XNLI_eval.py),
this script also dumps a per-subject breakdown (aggregated across all
languages) to `mmmlu_subject_results.csv`, since MMLU-style benchmarks are
commonly reported both by language and by subject.
"""

import argparse
import gc
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]
CANDIDATES = [" A", " B", " C", " D"]  # index-aligned with LETTERS


# --------------------------------------------------------------------------
# OOM-safe dynamic batching helpers
# --------------------------------------------------------------------------
def is_oom_error(err: BaseException) -> bool:
    """True if `err` looks like a CUDA / CPU out-of-memory error."""
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
    """Best-effort release of GPU/CPU memory before the next batch."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class DynamicBatcher:
    """Tracks a "safe" batch size across a run.

    Halves on OOM (down to `min_batch_size`) and slowly grows back by 1 after
    a streak of `grow_after` OOM-free batches, up to the original requested
    size. This means one OOM doesn't permanently cripple throughput for the
    rest of the evaluation, but we also don't immediately jump back into
    the same OOM.
    """

    def __init__(self, initial_batch_size: int, min_batch_size: int = 1, grow_after: int = 20):
        self.initial_batch_size = max(1, initial_batch_size)
        self.min_batch_size = max(1, min_batch_size)
        self.grow_after = grow_after
        self.size = self.initial_batch_size
        self._success_streak = 0

    def shrink(self):
        self.size = max(self.min_batch_size, self.size // 2)
        self._success_streak = 0

    def note_success(self):
        self._success_streak += 1
        if self._success_streak >= self.grow_after and self.size < self.initial_batch_size:
            self.size = min(self.initial_batch_size, self.size + 1)
            self._success_streak = 0


def score_chunk_with_oom_retry(model, tokenizer, chunk, build_prompt_fn, device, batcher, min_batch_size=1):
    """Score `chunk` (a list of examples), recursively halving on OOM.

    Returns a list of (example, pred_index_or_None) pairs aligned with
    `chunk`. `pred_index` is None only when even a single example could not
    be scored (persistent OOM at batch size 1) -- that example is skipped
    rather than crashing the run.
    """
    if not chunk:
        return []

    prompts = [build_prompt_fn(ex) for ex in chunk]

    # NOTE on the fix below: we deliberately do NOT call clear_memory() or
    # recurse from *inside* the `except` block. In Python 3, an `except X as e`
    # clause keeps `e` (and therefore its traceback) alive for the entire
    # duration of that block. The traceback holds a reference to every stack
    # frame between where the exception was raised and where it was caught --
    # including score_candidates_batch's frame, with its still-allocated GPU
    # tensors (input_ids, attention_mask, logits, log_probs, ...). While `e`
    # is alive, gc.collect()/torch.cuda.empty_cache() cannot reclaim that
    # memory, so clear_memory() was effectively a no-op. Worse, because the
    # old code recursed *inside* the except block, every OOM'd ancestor call
    # in the recursion tree kept its own `e`/traceback (and its own failed
    # batch's tensors) alive simultaneously, all the way down -- so by the
    # time batch_size reached 1, GPU memory was still clogged with every
    # larger failed batch above it, and even a single tiny example could OOM.
    #
    # The fix: catch the exception, record that it happened, then let the
    # `except` block end (Python auto-clears `e`/the traceback at that
    # point). Only after we're back to a clean scope do we call
    # clear_memory() and recurse -- so the memory is actually freed before
    # each retry.
    oom = False
    try:
        scores = score_candidates_batch(model, tokenizer, prompts, device)
    except RuntimeError as e:
        if not is_oom_error(e):
            raise
        oom = True
    # `e` and its traceback are now out of scope and cleared.

    if not oom:
        preds = scores.argmax(axis=1)
        return list(zip(chunk, preds))

    clear_memory()

    if len(chunk) <= min_batch_size:
        q_preview = str(chunk[0].get("Question", ""))[:80]
        print(f"[OOM][WARN] batch_size=1 still OOM, skipping example: {q_preview!r}")
        return [(ex, None) for ex in chunk]

    new_size = max(min_batch_size, len(chunk) // 2)
    print(f"[OOM] batch_size={len(chunk)} failed -> halving to {new_size} and retrying")
    batcher.shrink()
    mid = len(chunk) // 2
    left = score_chunk_with_oom_retry(model, tokenizer, chunk[:mid], build_prompt_fn, device, batcher, min_batch_size)
    clear_memory()
    right = score_chunk_with_oom_retry(model, tokenizer, chunk[mid:], build_prompt_fn, device, batcher, min_batch_size)
    return left + right


def build_prompt(ex: dict) -> str:
    return (
        f"{ex['Question']}\n"
        f"A. {ex['A']}\n"
        f"B. {ex['B']}\n"
        f"C. {ex['C']}\n"
        f"D. {ex['D']}\n"
        f"Answer:"
    )


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
        raise ValueError(f"Unrecognized test.json structure at {path}")
    return data


@torch.no_grad()
def score_candidates_batch(model, tokenizer, prompts, device):
    """Length-normalized log-likelihood of each candidate for a batch of prompts.
    Returns numpy array of shape (len(prompts), len(CANDIDATES))."""
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
    log_probs = F.log_softmax(outputs.logits, dim=-1)

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
        cur_bs = batcher.size
        chunk = data[idx : idx + cur_bs]

        pair_results = score_chunk_with_oom_retry(
            model, tokenizer, chunk, build_prompt, device, batcher, min_batch_size
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
        batcher.note_success()
        clear_memory()
    pbar.close()

    if skipped:
        print(f"[{lang}] WARNING: {skipped} example(s) skipped due to persistent OOM at batch_size=1.")

    acc = correct / total if total > 0 else 0.0
    return acc, total, skipped, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--data_root", default="data/downstream/mmmlu")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="Starting batch size; shrinks/grows dynamically on OOM.")
    parser.add_argument("--min_batch_size", type=int, default=1, help="Never split batches smaller than this before skipping an example.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", default="eval/Qwen1.5-MoE-A2.7B/results")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_predictions", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    print(f"Loading model: {args.model_name_or_path} (device={device}, dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    if device == "cpu":
        model.to(device)

    languages = args.languages or sorted(
        d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))
    )
    print(f"Languages to evaluate ({len(languages)}): {languages}")

    results = {}
    all_records = []
    total_skipped = 0
    for lang in languages:
        acc, total, skipped, records = evaluate_language(
            model, tokenizer, lang, args.data_root, device, args.batch_size, args.max_examples, args.min_batch_size
        )
        results[lang] = {"accuracy": acc, "n_examples": total, "n_skipped": skipped}
        total_skipped += skipped
        print(f"[{lang}] accuracy = {acc:.4f}  ({total} examples, {skipped} skipped)")
        for r in records:
            r["language"] = lang
        all_records.extend(records)

        if args.save_predictions:
            pd.DataFrame(records).to_csv(
                os.path.join(args.output_dir, f"mmmlu_predictions_{lang}.csv"), index=False
            )

        # free memory before moving on to the next language
        clear_memory()

    overall_correct = sum(r["accuracy"] * r["n_examples"] for r in results.values())
    overall_total = sum(r["n_examples"] for r in results.values())
    overall_micro_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    macro_acc = sum(r["accuracy"] for r in results.values()) / len(results) if results else 0.0

    print("=" * 60)
    print(f"Overall (micro, weighted by #examples) accuracy: {overall_micro_acc:.4f}")
    print(f"Macro-average (mean over languages) accuracy:    {macro_acc:.4f}")
    if total_skipped:
        print(f"Total skipped examples (persistent OOM): {total_skipped}")

    df = pd.DataFrame(
        [
            {"language": lang, "accuracy": r["accuracy"], "n_examples": r["n_examples"], "n_skipped": r["n_skipped"]}
            for lang, r in results.items()
        ]
    ).sort_values("language")
    csv_path = os.path.join(args.output_dir, "mmmlu_results.csv")
    df.to_csv(csv_path, index=False)

    # Bonus: per-subject breakdown aggregated across all languages
    if all_records:
        subj_df = pd.DataFrame(all_records)
        subj_summary = (
            subj_df.groupby("subject")["correct"]
            .agg(accuracy="mean", n_examples="count")
            .reset_index()
            .sort_values("subject")
        )
        subj_csv_path = os.path.join(args.output_dir, "mmmlu_subject_results.csv")
        subj_summary.to_csv(subj_csv_path, index=False)
        print(f"Saved: {subj_csv_path}")

    summary = {
        "model": args.model_name_or_path,
        "per_language": results,
        "overall_micro_accuracy": overall_micro_acc,
        "macro_average_accuracy": macro_acc,
        "total_skipped": total_skipped,
    }
    json_path = os.path.join(args.output_dir, "mmmlu_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()