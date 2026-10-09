#!/usr/bin/env python3
"""
Mixed-task router steering: identify task experts ONCE on a mix of XNLI + XQuAD + MMMLU,
then evaluate XNLI -> XQuAD -> MMMLU with that single shared set of steered experts.

Why this works
--------------
Delta (Bandarkar et al., ICLR 2026, Eq. 3) is just a difference of mean expert-activation
frequencies:

    Delta = mean_freq(English in-task prompts) - mean_freq(English FLORES)

so the "in-task" side can simply be a pool of prompts coming from several tasks. Here the
pool contains the English prompts of the three benchmarks (same prompt formats as in
XNLI_eval.py / XQuAD_eval.py / MMMLU_eval.py), with the SAME number of sequences per task
(min(--id_num_samples, smallest task)) so that no task dominates the average.
Expert k of layer l is selected iff Delta_mix[l, k] > tau, and the same experts are steered
for every benchmark.

This script re-uses the code of the three original eval scripts (it imports them), so the
steering block, the prompts, the scoring and the metrics are identical to the single-task runs.
Put mix_eval.py in the SAME folder as XNLI_eval.py, XQuAD_eval.py and MMMLU_eval.py.

Defaults
--------
    eval order  : XNLI -> XQuAD -> MMMLU
    batch sizes : 32    32       8          (--xnli_batch_size / --xquad_batch_size / --mmmlu_batch_size)
    outputs     : <output_dir>/xnli_results.csv, xquad_results.csv, mmmlu_results.csv
                  (+ mmmlu_subject_results.csv, mix_summary.json, delta_mix.pt)

Usage (run from the repo root)
------------------------------
    # (a) look at the mixed Delta only (no evaluation). Tau for a MIX is usually lower than for
    #     a single task, because task-specific experts get diluted -> check the printed counts.
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/mix_eval.py \
        --delta_path eval/rounter-steering/Qwen1.5-MoE-A2.7B/delta/mix.pt \
        --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/mix_soft --identify_only

    # (b) steered evaluation (soft, lambda=0.5) on all three benchmarks
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/mix_eval.py \
        --delta_path eval/rounter-steering/Qwen1.5-MoE-A2.7B/delta/mix.pt \
        --steering soft --steer_lambda 0.5 --tau 0.1 --target_layers 4 19 \
        --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/mix_soft_tau0.1

    # (c) baseline (no steering, no Delta needed)
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/mix_eval.py --steering none \
        --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/mix_baseline

Notes
-----
- MMMLU has no English folder, so its English in-task prompts come from
  --mmmlu_task_data_path (default: data/english_task/mmlu/test.json; accepts Question/A/B/C/D or
  question + choices records, or a folder -> test.json is picked when there is no train.json).
- `--benchmarks xquad mmmlu` evaluates a subset (Delta is still the 3-task mix, and is read from
  the cache --delta_path, so re-running a single benchmark is cheap).
- `--max_examples 20` is handy to smoke-test the whole pipeline.
"""

import argparse
import copy
import json
import os
import sys
import time
from collections import OrderedDict

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Re-use the three original scripts (they must live next to this file).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import MMMLU_eval  # noqa: E402
import XNLI_eval  # noqa: E402
import XQuAD_eval  # noqa: E402

ORDER = ("xnli", "xquad", "mmmlu")  # fixed evaluation order
# English in-task data used for the mixed Delta. XNLI / XQuAD have an English split under their
# downstream folder (None = use it); MMMLU has none, so it points to the English MMLU test split.
DEFAULT_TASK_DATA = {"xnli": None, "xquad": None, "mmmlu": "data/english_task/mmlu/test.json"}


# =============================================================================
# Args
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Mixed-task Delta + steered eval on XNLI -> XQuAD -> MMMLU")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--output_dir", default="eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/mix")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--benchmarks", nargs="+", default=list(ORDER), choices=list(ORDER),
                   help="which benchmarks to evaluate (always run in the order XNLI -> XQuAD -> MMMLU)")
    p.add_argument("--max_examples", type=int, default=None, help="debug: cap #examples per language")
    p.add_argument("--save_predictions", action="store_true", help="also dump per-example predictions (CSV)")

    g = p.add_argument_group("data (one block per benchmark)")
    g.add_argument("--xnli_root", default="data/downstream/xnli")
    g.add_argument("--xquad_root", default="data/downstream/xquad")
    g.add_argument("--mmmlu_root", default="data/downstream/mmmlu")
    for name in ORDER:
        g.add_argument(f"--{name}_task_data_path", default=DEFAULT_TASK_DATA[name],
                       help=f"English in-task data for {name.upper()} used to build the mixed Delta "
                            f"(default: {DEFAULT_TASK_DATA[name] or f'the English split under --{name}_root'})")
        g.add_argument(f"--{name}_id_lang", default=None,
                       help=f"folder name of the English split under --{name}_root")
        g.add_argument(f"--{name}_languages", nargs="+", default=None,
                       help=f"subset of {name.upper()} languages (default: all found)")

    b = p.add_argument_group("batch sizes")
    b.add_argument("--xnli_batch_size", type=int, default=32)
    b.add_argument("--xquad_batch_size", type=int, default=32)
    b.add_argument("--mmmlu_batch_size", type=int, default=8)
    b.add_argument("--mmmlu_min_batch_size", type=int, default=1,
                   help="MMMLU OOM-retry never splits batches below this size")
    b.add_argument("--max_new_tokens", type=int, default=32, help="XQuAD generation length")

    # --steering/--tau/--target_layers/--delta_path/--flores_path/--id_* ... (same flags as the single-task scripts)
    XNLI_eval.add_steering_args(p)
    args = p.parse_args()

    if args.task_data_path or args.id_lang:
        p.error("--task_data_path / --id_lang are single-task flags; in the mix use "
                "--xnli_*/--xquad_*/--mmmlu_task_data_path and --<task>_id_lang instead.")
    return args


# =============================================================================
# Mixed Delta: English in-task prompts of the three tasks, equal weight per task
# =============================================================================
def mix_signature(args):
    """String describing the mix; stored in the Delta cache metadata so a stale cache is never reused."""
    parts = []
    for name in ORDER:
        src = getattr(args, f"{name}_task_data_path") or \
            f"{getattr(args, f'{name}_root')}/{getattr(args, f'{name}_id_lang') or 'default-en'}"
        parts.append(f"{name}={src}")
    return "mix[equal-weight|" + ";".join(parts) + "]"


def build_mix_texts(args):
    """English in-task prompts of XNLI + XQuAD + MMMLU, n sequences per task (n = min over tasks)."""
    getters = OrderedDict([
        ("xnli", XNLI_eval.get_task_texts),
        ("xquad", XQuAD_eval.get_task_texts),
        ("mmmlu", MMMLU_eval.get_task_texts),
    ])
    per_task = OrderedDict()
    for name, fn in getters.items():
        sub = copy.copy(args)  # the per-task get_task_texts read data_root / task_data_path / id_lang
        sub.data_root = getattr(args, f"{name}_root")
        sub.task_data_path = getattr(args, f"{name}_task_data_path")
        sub.id_lang = getattr(args, f"{name}_id_lang")
        try:
            per_task[name] = fn(sub)  # already capped at --id_num_samples
        except (FileNotFoundError, KeyError) as e:
            hint = (f"pass --{name}_task_data_path <English {name.upper()}-style json> "
                    f"(or --{name}_id_lang <folder under --{name}_root>)")
            raise SystemExit(f"[mix] cannot build English in-task prompts for {name.upper()}: {e}\n[mix] -> {hint}")

    n = min(len(v) for v in per_task.values())
    if n == 0:
        raise SystemExit("[mix] one of the tasks has no English in-task prompts.")
    mixed = []
    for name, texts in per_task.items():
        texts = XNLI_eval.subsample(texts, n, args.seed)  # equal weight per task
        print(f"[mix] {name.upper():<6}: {len(texts)} English prompts (available {len(per_task[name])})")
        mixed.extend(texts)
    print(f"[mix] total {len(mixed)} in-task prompts ({n} per task)")
    return mixed


# =============================================================================
# Per-benchmark evaluation (thin wrappers around the original evaluate_language())
# =============================================================================
def _list_dirs(root):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def _save_csv(rows, columns, path):
    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df = df.sort_values("language")
    df.to_csv(path, index=False)  # rewritten after every language -> a crash loses at most one language


def _dump_predictions(records, args, bench, lang):
    if args.save_predictions:
        pd.DataFrame(records).to_csv(os.path.join(args.output_dir, f"{bench}_predictions_{lang}.csv"), index=False)


def run_xnli(args, model, tokenizer, device):
    tokenizer.padding_side = "right"  # log-likelihood slicing assumes right padding
    root = args.xnli_root
    langs = args.xnli_languages or _list_dirs(root)
    print(f"\n[XNLI] {len(langs)} languages, batch_size={args.xnli_batch_size}: {langs}")
    csv_path = os.path.join(args.output_dir, "xnli_results.csv")
    cols, rows = ["language", "accuracy", "n_examples"], []
    for lang in langs:
        acc, total, records = XNLI_eval.evaluate_language(
            model, tokenizer, lang, root, device, args.xnli_batch_size, args.max_examples)
        print(f"[XNLI][{lang}] accuracy = {acc:.4f}  ({total} examples)")
        rows.append({"language": lang, "accuracy": acc, "n_examples": total})
        _save_csv(rows, cols, csv_path)
        _dump_predictions(records, args, "xnli", lang)
        MMMLU_eval.clear_memory()
    n = sum(r["n_examples"] for r in rows)
    return {
        "csv": csv_path,
        "micro_accuracy": sum(r["accuracy"] * r["n_examples"] for r in rows) / n if n else 0.0,
        "macro_accuracy": sum(r["accuracy"] for r in rows) / len(rows) if rows else 0.0,
    }


def run_xquad(args, model, tokenizer, device):
    tokenizer.padding_side = "left"  # required for batched causal-LM generation
    root = args.xquad_root
    langs = args.xquad_languages or XQuAD_eval.XQUAD_LANGS
    print(f"\n[XQuAD] {len(langs)} languages, batch_size={args.xquad_batch_size}: {langs}")
    csv_path = os.path.join(args.output_dir, "xquad_results.csv")
    cols, rows = ["language", "em", "f1", "n_examples"], []
    for lang in langs:
        data_path = os.path.join(root, lang, "validation.json")
        if not os.path.exists(data_path):
            print(f"[XQuAD][WARN] skipping '{lang}': {data_path} not found")
            continue
        records = XQuAD_eval.load_xquad_split(data_path)
        r = XQuAD_eval.evaluate_language(
            model, tokenizer, lang, records,
            batch_size=args.xquad_batch_size, max_new_tokens=args.max_new_tokens,
            device=device, limit=args.max_examples)
        print(f"[XQuAD][{lang}] EM={r['em']:.2f}  F1={r['f1']:.2f}  (n={r['num_examples']})")
        rows.append({"language": lang, "em": r["em"], "f1": r["f1"], "n_examples": r["num_examples"]})
        _save_csv(rows, cols, csv_path)
        _dump_predictions(
            [{"id": k, "prediction": v["prediction"], "gold": " | ".join(v["gold"]), "em": v["em"], "f1": v["f1"]}
             for k, v in r["predictions"].items()], args, "xquad", lang)
        MMMLU_eval.clear_memory()
    n = sum(r["n_examples"] for r in rows)
    return {
        "csv": csv_path,
        "macro_em": sum(r["em"] for r in rows) / len(rows) if rows else 0.0,   # XTREME convention: macro over languages
        "macro_f1": sum(r["f1"] for r in rows) / len(rows) if rows else 0.0,
        "micro_em": sum(r["em"] * r["n_examples"] for r in rows) / n if n else 0.0,
        "micro_f1": sum(r["f1"] * r["n_examples"] for r in rows) / n if n else 0.0,
    }


def run_mmmlu(args, model, tokenizer, device):
    tokenizer.padding_side = "right"
    root = args.mmmlu_root
    langs = args.mmmlu_languages or _list_dirs(root)
    print(f"\n[MMMLU] {len(langs)} languages, batch_size={args.mmmlu_batch_size}: {langs}")
    csv_path = os.path.join(args.output_dir, "mmmlu_results.csv")
    subj_csv_path = os.path.join(args.output_dir, "mmmlu_subject_results.csv")
    cols, rows, subject_totals = ["language", "accuracy", "n_examples", "n_skipped"], [], {}
    for lang in langs:
        acc, total, skipped, records, subj = MMMLU_eval.evaluate_language(
            model, tokenizer, lang, root, device, args.mmmlu_batch_size, args.max_examples, args.mmmlu_min_batch_size)
        print(f"[MMMLU][{lang}] accuracy = {acc:.4f}  ({total} examples, {skipped} skipped)")
        rows.append({"language": lang, "accuracy": acc, "n_examples": total, "n_skipped": skipped})
        _save_csv(rows, cols, csv_path)
        MMMLU_eval.merge_subject_breakdown(subject_totals, subj)
        pd.DataFrame(
            [{"subject": s, "accuracy": v["correct"] / v["total"] if v["total"] else 0.0, "n_examples": v["total"]}
             for s, v in sorted(subject_totals.items())]
        ).to_csv(subj_csv_path, index=False)
        for rec in records:
            rec["language"] = lang
        _dump_predictions(records, args, "mmmlu", lang)
        MMMLU_eval.clear_memory()
    n = sum(r["n_examples"] for r in rows)
    return {
        "csv": csv_path,
        "micro_accuracy": sum(r["accuracy"] * r["n_examples"] for r in rows) / n if n else 0.0,
        "macro_accuracy": sum(r["accuracy"] for r in rows) / len(rows) if rows else 0.0,
        "total_skipped": sum(r["n_skipped"] for r in rows),
    }


RUNNERS = OrderedDict([("xnli", run_xnli), ("xquad", run_xquad), ("mmmlu", run_mmmlu)])


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    print(f"Loading model: {args.model_name_or_path} (device={device}, dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    if device == "cpu":
        model.to(device)

    # ---- 1) mixed Delta (computed once, BEFORE steering) + switch steering on ----------------
    # The signature replaces --task_data_path only inside the Delta-cache metadata; the real
    # per-task sources are read through --<task>_task_data_path / --<task>_id_lang.
    signature = mix_signature(args)
    args.task_data_path = signature
    controller, steering_cfg = XNLI_eval.setup_router_steering(
        args, model, tokenizer, device, lambda: build_mix_texts(args), tag="mix")
    steering_cfg = dict(steering_cfg)
    steering_cfg["delta_source"] = signature

    # ---- 2) evaluate XNLI -> XQuAD -> MMMLU with the same steered experts ----------------------
    summary = OrderedDict(model=args.model_name_or_path, steering_config=steering_cfg, benchmarks=OrderedDict())
    t0 = time.time()
    for name in ORDER:
        if name not in args.benchmarks:
            continue
        t1 = time.time()
        res = RUNNERS[name](args, model, tokenizer, device)
        res["seconds"] = round(time.time() - t1, 1)
        summary["benchmarks"][name] = res
        print(f"[{name.upper()}] saved: {res['csv']}")
        MMMLU_eval.clear_memory()

    summary["batch_sizes"] = {k: getattr(args, f"{k}_batch_size") for k in ORDER}
    summary["total_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(args.output_dir, "mix_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    for name, res in summary["benchmarks"].items():
        print(f"{name.upper():<6} " + "  ".join(f"{k}={v:.4f}" for k, v in res.items() if isinstance(v, float) and k != "seconds"))
    print(f"Saved: {os.path.join(args.output_dir, 'mix_summary.json')}")
    XNLI_eval.report_steering_usage(controller)


if __name__ == "__main__":
    main()