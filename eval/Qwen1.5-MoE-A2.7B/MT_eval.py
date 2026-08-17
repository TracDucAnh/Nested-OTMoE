#!/usr/bin/env python3
"""
Machine-translation evaluation of Qwen1.5-MoE-A2.7B on 5 language pairs:
    vi -> zh    sw -> ar    ht -> fr    fr -> wo    hi -> ur

For every pair, records are loaded from `data/mt_translation/<src>-<tgt>.json`
(a flat list of {"id", "<src>", "<tgt>"} objects, normalized format shared
across all pairs in this repo). The model is prompted zero-shot to translate
each source sentence. Qwen1.5-MoE-A2.7B is the base (non-instruct) checkpoint,
so this script uses the raw completion-style prompt by default (same
zero-shot base-model protocol as XQuAD_eval.py in this eval/ directory)
instead of the tokenizer's chat template. If you point --model_name_or_path
at a chat-tuned checkpoint instead (e.g. Qwen1.5-MoE-A2.7B-Chat), drop
--no_chat_template to use `add_generation_prompt=True` formatting.

Translations are scored at the corpus level with:
    - spBLEU : sacrebleu BLEU with FLORES-200 SentencePiece tokenization
               (tokenize="flores200"). Requires internet access on first run
               so sacrebleu can download the FLORES-200 SPM model.
    - chrF++ : sacrebleu chrF with word_order=2 (character + word n-grams).

Length-sorted batching
-----------------------
Within each pair, examples not yet translated are sorted by source-sentence
length before being sliced into batches. This groups similarly-long
sentences together so far less padding is wasted per batch (a batch of one
very long sentence + many short ones pads every short sentence up to the
long one's length) -- purely a speed optimization, output order is restored
afterwards.

OOM-safe dynamic batching
--------------------------
Same protocol as MMMLU_eval.py in this eval/ directory. Every chunk of data
is first attempted at the full `--batch_size`. If a chunk raises a CUDA/CPU
out-of-memory error, we free memory (gc.collect + torch.cuda.empty_cache)
and split the offending chunk in half, retrying each half recursively
(halving again if needed). This shrinking is purely local to the chunk that
OOM'd: it does NOT lower the starting size for the next chunk -- each new
chunk always starts fresh at the full `--batch_size`. If a single example
(batch size 1) still OOMs, that example is skipped (logged and marked
"skipped": true in the predictions file) instead of crashing the run.

Resume support
--------------
`<pair>_predictions.json` in `--output_dir` doubles as a checkpoint. As soon
as a batch finishes translating, every example finished so far (both
resumed-from-checkpoint and newly translated) is written into that file
immediately (atomically, via a temp file + os.replace, so a crash mid-write
never corrupts it). On the next run, any example id already present in that
file is skipped and NOT retranslated -- only the remaining ids are sent to
the model. Once every requested pair is fully checkpointed, the (large)
model is not even loaded.

Usage (run from eval/Qwen1.5-MoE-A2.7B/):
    python MT_eval.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --data_root ../../data/mt_translation \
        --output_dir ./result \
        --batch_size 1024 \
        --max_new_tokens 256 \
        --no_chat_template

Only a subset of pairs:
    python MT_eval.py --pairs vi-zh hi-ur

Quick debug run on a handful of examples per pair:
    python MT_eval.py --limit 20

Requires (on top of the shared requirements.txt): sacrebleu>=2.4.0
    pip install sacrebleu
"""

import argparse
import csv
import gc
import json
import os
import re
import sys
import tempfile
import time
from collections import OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

try:
    import sacrebleu
except ImportError:
    sys.exit(
        "sacrebleu is required for spBLEU / chrF++ scoring but is not installed.\n"
        "Install it with:  pip install sacrebleu"
    )


# ----------------------------------------------------------------------------- #
# Language pairs available in data/mt_translation/<pair>.json
# Pair string is "<src>-<tgt>" and doubles as the filename stem; the json
# records use the raw ISO codes ("vi", "zh", ...) as keys, so field names are
# derived from the pair string itself -- nothing here is hardcoded per file.
# ----------------------------------------------------------------------------- #
MT_PAIRS = ["vi-zh", "sw-ar", "ht-fr", "fr-wo", "hi-ur"]

# Human-readable language names, used only to build the natural-language
# instruction prompt (e.g. "Translate from Vietnamese to Chinese").
LANG_NAMES = {
    "vi": "Vietnamese",
    "zh": "Chinese",
    "sw": "Swahili",
    "ar": "Arabic",
    "ht": "Haitian Creole",
    "fr": "French",
    "wo": "Wolof",
    "hi": "Hindi",
    "ur": "Urdu",
}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


# ----------------------------------------------------------------------------- #
# Prompting
# ----------------------------------------------------------------------------- #
INSTRUCTION_TEMPLATE = (
    "Translate the following text from {src_name} to {tgt_name}. "
    "Output only the translation, with no explanation, notes, or additional text.\n\n"
    "{src_name}: {text}"
)


def build_prompt(tokenizer, text: str, src_code: str, tgt_code: str, use_chat_template: bool = True) -> str:
    src_name, tgt_name = lang_name(src_code), lang_name(tgt_code)
    content = INSTRUCTION_TEMPLATE.format(src_name=src_name, tgt_name=tgt_name, text=text)
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Raw completion-style fallback: end the prompt on the target-language label
    # so a base model naturally continues with the translation.
    return f"{content}\n{tgt_name}:"


# ----------------------------------------------------------------------------- #
# Data loading
# ----------------------------------------------------------------------------- #
def load_mt_pair(path: str, src_code: str, tgt_code: str):
    """
    Loads a single pair file and returns a flat list of
    {"id", <src_code>, <tgt_code>} records. Accepts either a flat list (the
    normalized format used in this repo) or a {"data": [...]} wrapper.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        for key in ("data", "records", "examples"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise ValueError(f"Unrecognized MT file format: {path}")

    for r in raw:
        if src_code not in r or tgt_code not in r:
            raise ValueError(
                f"Record {r.get('id', '?')} in {path} is missing '{src_code}' or '{tgt_code}' field."
            )
        if "id" not in r:
            raise ValueError(f"Record missing 'id' field in {path}; ids are required for checkpoint/resume.")
    return raw


# ----------------------------------------------------------------------------- #
# OOM-safe dynamic batching helpers (same protocol as MMMLU_eval.py)
# ----------------------------------------------------------------------------- #
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
    """Holds the batch-size bounds for a run.

    Every new outer chunk of data starts fresh at `initial_batch_size` -- an
    OOM on one chunk does NOT lower the starting size for the *next* chunk.
    The halving on OOM (down to `min_batch_size`) only happens *within* a
    single chunk's divide-and-conquer retry (see
    `translate_chunk_with_oom_retry`) and is discarded once that chunk is
    done; it never leaks into subsequent chunks.
    """

    def __init__(self, initial_batch_size: int, min_batch_size: int = 1):
        self.initial_batch_size = max(1, initial_batch_size)
        self.min_batch_size = max(1, min_batch_size)


def translate_chunk_with_oom_retry(model, tokenizer, chunk, src_code, tgt_code,
                                    use_chat_template, max_new_tokens, device, min_batch_size=1):
    """Translate `chunk` (a list of records), recursively halving on OOM.

    Returns a list of (record, translation_or_None) pairs aligned with
    `chunk`. `translation` is None only when even a single example could
    not be translated (persistent OOM at batch size 1) -- that example is
    skipped rather than crashing the run.
    """
    if not chunk:
        return []

    prompts = [
        build_prompt(tokenizer, r[src_code], src_code, tgt_code, use_chat_template) for r in chunk
    ]

    # See the long comment in MMMLU_eval.py's score_chunk_with_oom_retry for
    # why we deliberately catch-then-exit-the-except-block before clearing
    # memory / recursing, rather than doing it from inside `except`: an
    # `except X as e` clause keeps `e` (and its traceback, which pins every
    # GPU tensor in the failed call's stack frames) alive for the whole
    # block, so clear_memory() would otherwise be a no-op.
    oom = False
    try:
        preds = translate_batch(model, tokenizer, prompts, max_new_tokens, device)
    except RuntimeError as e:
        if not is_oom_error(e):
            raise
        oom = True
    # `e` and its traceback are now out of scope and cleared.

    if not oom:
        return list(zip(chunk, preds))

    clear_memory()

    if len(chunk) <= min_batch_size:
        text_preview = str(chunk[0].get(src_code, ""))[:80]
        print(f"[OOM][WARN] batch_size=1 still OOM, skipping example: {text_preview!r}")
        return [(ex, None) for ex in chunk]

    new_size = max(min_batch_size, len(chunk) // 2)
    print(
        f"[OOM] batch_size={len(chunk)} failed -> halving to {new_size} and retrying "
        f"(next outer chunk still starts fresh at the full --batch_size)"
    )
    mid = len(chunk) // 2
    left = translate_chunk_with_oom_retry(
        model, tokenizer, chunk[:mid], src_code, tgt_code, use_chat_template,
        max_new_tokens, device, min_batch_size,
    )
    clear_memory()
    right = translate_chunk_with_oom_retry(
        model, tokenizer, chunk[mid:], src_code, tgt_code, use_chat_template,
        max_new_tokens, device, min_batch_size,
    )
    return left + right


# ----------------------------------------------------------------------------- #
# Generation
# ----------------------------------------------------------------------------- #
def clean_translation(text: str) -> str:
    """Strip whitespace/quotes and drop stray language-label echoes or rambling."""
    text = text.strip()
    # Model sometimes echoes a "Chinese:" / "French:" style label before the
    # actual translation -- strip a single leading label if present.
    text = re.sub(r"^[A-Za-zÀ-ÖØ-öø-ÿ ]{2,20}:\s*", "", text, count=1)
    # If the model keeps rambling after the translation (extra paragraph,
    # notes, etc.), keep only the first paragraph.
    text = text.split("\n\n")[0].strip()
    text = text.strip("\"'“”‘’ ")
    return text


@torch.no_grad()
def translate_batch(model, tokenizer, prompts, max_new_tokens, device):
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096
    ).to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_only = output_ids[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
    return [clean_translation(d) for d in decoded]


# ----------------------------------------------------------------------------- #
# Metrics (corpus-level spBLEU + chrF++)
# ----------------------------------------------------------------------------- #
def compute_corpus_metrics(hyps, refs):
    """hyps/refs: parallel lists of strings (one reference per hypothesis)."""
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200")
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)  # chrF++ (word_order=2)
    return bleu.score, chrf.score


# ----------------------------------------------------------------------------- #
# Checkpoint / resume helpers
# ----------------------------------------------------------------------------- #
def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Write `data` to `path` atomically: write to a temp file in the same
    directory, then os.replace() it into place. This means a crash or kill
    mid-write can never leave a truncated/corrupted predictions file behind
    -- important since this file also serves as the resume checkpoint."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_mt_", dir=directory)
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


def load_existing_predictions(pred_path: str) -> "OrderedDict[str, dict]":
    """Load a pair's `<pair>_predictions.json` from a previous (possibly
    interrupted) run, if any. Returns an OrderedDict keyed by example id --
    every id in it is skipped (not retranslated) this run."""
    if not os.path.exists(pred_path):
        return OrderedDict()
    try:
        with open(pred_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read checkpoint {pred_path} ({e}); starting this pair fresh.")
        return OrderedDict()
    out = OrderedDict()
    for rec in data:
        rid = rec.get("id")
        if rid is not None:
            out[rid] = rec
    return out


# ----------------------------------------------------------------------------- #
# Per-pair evaluation
# ----------------------------------------------------------------------------- #
def evaluate_pair(model, tokenizer, pair, records, src_code, tgt_code,
                   batch_size, max_new_tokens, device, use_chat_template,
                   pred_path, min_batch_size=1):
    id_to_index = {r["id"]: i for i, r in enumerate(records)}
    n = len(records)

    # Resume: anything already checkpointed from a previous run is skipped.
    # Drop any stale ids that no longer belong to this pair's current data
    # (e.g. the source file changed between runs).
    done = OrderedDict(
        (rid, rec) for rid, rec in load_existing_predictions(pred_path).items() if rid in id_to_index
    )
    if done:
        print(f"  [resume] {len(done)}/{n} example(s) already translated in a previous run; skipping them.")

    remaining = [r for r in records if r["id"] not in done]
    n_skipped_new = 0

    if remaining:
        # Sort by source-sentence length before batching: groups
        # similarly-long sentences together so batches waste far less
        # padding, which speeds up decoding noticeably. Original dataset
        # order is restored below when we checkpoint / compute metrics.
        remaining = sorted(remaining, key=lambda r: len(r[src_code]))

        batcher = DynamicBatcher(initial_batch_size=batch_size, min_batch_size=min_batch_size)
        idx = 0
        pbar = tqdm(total=len(remaining), desc=pair, unit="ex", leave=False)
        while idx < len(remaining):
            # Every new chunk starts fresh at the full requested batch size;
            # an OOM only shrinks *that* chunk's own retry, not later ones.
            cur_bs = batcher.initial_batch_size
            chunk = remaining[idx: idx + cur_bs]

            chunk_results = translate_chunk_with_oom_retry(
                model, tokenizer, chunk, src_code, tgt_code, use_chat_template,
                max_new_tokens, device, min_batch_size,
            )

            for record, pred in chunk_results:
                out_rec = OrderedDict()
                out_rec["id"] = record["id"]
                out_rec[src_code] = record[src_code]
                out_rec[tgt_code] = record[tgt_code]
                if pred is None:
                    out_rec[f"translated_{tgt_code}"] = ""
                    out_rec["skipped"] = True
                    n_skipped_new += 1
                else:
                    out_rec[f"translated_{tgt_code}"] = pred
                    out_rec["skipped"] = False
                done[record["id"]] = out_rec

            idx += len(chunk)
            pbar.update(len(chunk))

            # Checkpoint right after this batch: persist everything finished
            # so far (resumed + new), restored to original dataset order.
            # A crash/kill immediately after this line loses nothing except
            # work not yet attempted.
            ordered = sorted(done.values(), key=lambda r: id_to_index[r["id"]])
            atomic_write_json(pred_path, ordered)

            clear_memory()
        pbar.close()

        if n_skipped_new:
            print(f"  [{pair}] WARNING: {n_skipped_new} example(s) skipped this run due to persistent OOM at batch_size=1.")

    predictions = sorted(done.values(), key=lambda r: id_to_index[r["id"]])
    hyps = [r[f"translated_{tgt_code}"] for r in predictions]
    refs = [r[tgt_code] for r in predictions]
    n_skipped_total = sum(1 for r in predictions if r.get("skipped"))

    spbleu, chrfpp = compute_corpus_metrics(hyps, refs) if predictions else (0.0, 0.0)

    return {
        "num_examples": len(predictions),
        "num_skipped": n_skipped_total,
        "spBLEU": spbleu,
        "chrF++": chrfpp,
        "predictions": predictions,
    }


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="MT evaluation (spBLEU / chrF++) for Qwen1.5-MoE-A2.7B")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--data_root", type=str, default="../../data/mt_translation",
                         help="Path to data/mt_translation")
    parser.add_argument("--output_dir", type=str, default="./result")
    parser.add_argument("--pairs", type=str, nargs="+", default=MT_PAIRS,
                         help="Subset of language pairs to evaluate, e.g. vi-zh hi-ur")
    parser.add_argument("--batch_size", type=int, default=1024,
                         help="Batch size each chunk starts at; halves locally (per-chunk only) on OOM.")
    parser.add_argument("--min_batch_size", type=int, default=1,
                         help="Never split batches smaller than this before skipping an example.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of examples per pair, for quick debugging")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat-template formatting and use raw completion-style prompts instead (base-model style).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing <pair>_predictions.json checkpoints and retranslate every example from scratch.",
    )
    args = parser.parse_args()
    use_chat_template = not args.no_chat_template

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    # ------------------------------------------------------------------- #
    # Pre-scan every requested pair's data + existing checkpoint *before*
    # loading the model, so a fully-checkpointed run (everything already
    # translated in a previous session) never has to load it at all.
    # ------------------------------------------------------------------- #
    pair_records, pair_pred_paths = OrderedDict(), {}
    work_remaining = False
    for pair in args.pairs:
        src_code, tgt_code = pair.split("-")
        data_path = os.path.join(args.data_root, f"{pair}.json")
        if not os.path.exists(data_path):
            print(f"[WARN] Skipping '{pair}': {data_path} not found")
            continue

        records = load_mt_pair(data_path, src_code, tgt_code)
        if args.limit:
            records = records[: args.limit]
        pair_records[pair] = records

        pred_path = os.path.join(args.output_dir, f"{pair}_predictions.json")
        pair_pred_paths[pair] = pred_path

        if args.overwrite:
            n_done = 0
        else:
            ids_in_data = {r["id"] for r in records}
            n_done = sum(1 for rid in load_existing_predictions(pred_path) if rid in ids_in_data)

        if n_done < len(records):
            work_remaining = True
        if n_done:
            print(f"[{pair}] {n_done}/{len(records)} example(s) already checkpointed.")

    if args.overwrite:
        for pred_path in pair_pred_paths.values():
            if os.path.exists(pred_path):
                os.remove(pred_path)

    if not pair_records:
        print("No language pairs were found (no data). Exiting.")
        sys.exit(1)

    tokenizer = model = None
    if work_remaining:
        print(f"Loading tokenizer & model: {args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # required for batched causal-LM generation

        if use_chat_template and not getattr(tokenizer, "chat_template", None):
            print("[WARN] Tokenizer has no chat_template; falling back to raw completion-style prompts.")
            use_chat_template = False
        elif use_chat_template:
            print("[INFO] Using tokenizer chat_template to format prompts (Instruct-model mode).")

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype_map[args.dtype],
            trust_remote_code=args.trust_remote_code,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            model.to(device)
        model.eval()
    else:
        print("Nothing left to translate -- every requested pair is already fully checkpointed.")

    results = OrderedDict()
    t0 = time.time()

    for pair in tqdm(list(pair_records.keys()), desc="pairs"):
        src_code, tgt_code = pair.split("-")
        records = pair_records[pair]

        print(f"\n=== Evaluating pair: {pair} ({lang_name(src_code)} -> {lang_name(tgt_code)}) ===")
        pair_result = evaluate_pair(
            model, tokenizer, pair, records, src_code, tgt_code,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            use_chat_template=use_chat_template,
            pred_path=pair_pred_paths[pair],
            min_batch_size=args.min_batch_size,
        )
        results[pair] = pair_result
        print(
            f"  {pair}: spBLEU={pair_result['spBLEU']:.2f}  chrF++={pair_result['chrF++']:.2f}  "
            f"(n={pair_result['num_examples']}, skipped={pair_result['num_skipped']})"
        )
        # predictions.json for this pair was already checkpointed batch-by-
        # batch inside evaluate_pair -- nothing left to save here.

    # ------------------------------------------------------------------- #
    # Aggregate CSV report (overall = macro-average across pairs)
    # ------------------------------------------------------------------- #
    if not results:
        print("No language pairs were evaluated (no data found). Exiting.")
        sys.exit(1)

    csv_path = os.path.join(args.output_dir, "mt_summary.csv")
    fieldnames = ["pair", "src_lang", "tgt_lang", "num_examples", "num_skipped", "spBLEU", "chrF++"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair, r in results.items():
            src_code, tgt_code = pair.split("-")
            writer.writerow({
                "pair": pair,
                "src_lang": src_code,
                "tgt_lang": tgt_code,
                "num_examples": r["num_examples"],
                "num_skipped": r["num_skipped"],
                "spBLEU": round(r["spBLEU"], 2),
                "chrF++": round(r["chrF++"], 2),
            })
        overall_spbleu = sum(r["spBLEU"] for r in results.values()) / len(results)
        overall_chrfpp = sum(r["chrF++"] for r in results.values()) / len(results)
        writer.writerow({
            "pair": "overall",
            "src_lang": "",
            "tgt_lang": "",
            "num_examples": sum(r["num_examples"] for r in results.values()),
            "num_skipped": sum(r["num_skipped"] for r in results.values()),
            "spBLEU": round(overall_spbleu, 2),
            "chrF++": round(overall_chrfpp, 2),
        })

    print("\n" + "=" * 58)
    print(f"{'Pair':<10}{'#Examples':<12}{'spBLEU':<12}{'chrF++':<12}")
    print("-" * 58)
    for pair, r in results.items():
        print(f"{pair:<10}{r['num_examples']:<12}{r['spBLEU']:<12.2f}{r['chrF++']:<12.2f}")
    print("-" * 58)
    print(f"{'OVERALL':<10}{sum(r['num_examples'] for r in results.values()):<12}"
          f"{overall_spbleu:<12.2f}{overall_chrfpp:<12.2f}")
    print("=" * 58)
    total_skipped = sum(r["num_skipped"] for r in results.values())
    if total_skipped:
        print(f"Total skipped examples (persistent OOM): {total_skipped}")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"\nSaved per-pair predictions (checkpointed batch-by-batch) and mt_summary.csv to: {args.output_dir}")


if __name__ == "__main__":
    main()