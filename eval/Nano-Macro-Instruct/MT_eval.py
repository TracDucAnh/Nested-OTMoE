#!/usr/bin/env python3
"""
Machine-translation evaluation of Nano-Macro-Instruct on 5 language pairs:
    vi -> zh    sw -> ar    ht -> fr    fr -> wo    hi -> ur

For every pair, records are loaded from `data/mt_translation/<src>-<tgt>.json`
(a flat list of {"id", "<src>", "<tgt>"} objects, normalized format shared
across all pairs in this repo). The model is prompted zero-shot to translate
each source sentence, using the tokenizer's chat template
(`add_generation_prompt=True`) since Nano-Macro-Instruct is instruct-tuned.
If the tokenizer has no chat_template (or --no_chat_template is passed), the
script falls back to a raw completion-style prompt.

Translations are scored at the corpus level with:
    - spBLEU : sacrebleu BLEU with FLORES-200 SentencePiece tokenization
               (tokenize="flores200"). Requires internet access on first run
               so sacrebleu can download the FLORES-200 SPM model.
    - chrF++ : sacrebleu chrF with word_order=2 (character + word n-grams).

Usage (run from eval/Nano-Macro-Instruct/):
    python MT_eval.py \
        --model_name_or_path Nano-Macro-Instruct \
        --data_root ../../data/mt_translation \
        --output_dir ./result \
        --batch_size 128 \
        --max_new_tokens 256

Only a subset of pairs:
    python MT_eval.py --pairs vi-zh hi-ur

Quick debug run on a handful of examples per pair:
    python MT_eval.py --limit 20

Disable chat-template formatting (base-model-style raw prompt):
    python MT_eval.py --no_chat_template

Requires (on top of the shared requirements.txt): sacrebleu>=2.4.0
    pip install sacrebleu
"""

import argparse
import csv
import json
import os
import re
import sys
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
    return raw


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
# Per-pair evaluation
# ----------------------------------------------------------------------------- #
def evaluate_pair(model, tokenizer, pair, records, src_code, tgt_code,
                   batch_size, max_new_tokens, device, use_chat_template, limit=None):
    if limit:
        records = records[:limit]

    n = len(records)
    predictions = []
    hyps, refs = [], []

    for start in tqdm(range(0, n, batch_size), desc=pair, unit="batch", leave=False):
        batch = records[start:start + batch_size]
        prompts = [
            build_prompt(tokenizer, r[src_code], src_code, tgt_code, use_chat_template) for r in batch
        ]
        preds = translate_batch(model, tokenizer, prompts, max_new_tokens, device)

        for record, pred in zip(batch, preds):
            hyps.append(pred)
            refs.append(record[tgt_code])
            out_rec = OrderedDict()
            out_rec["id"] = record["id"]
            out_rec[src_code] = record[src_code]
            out_rec[tgt_code] = record[tgt_code]
            out_rec[f"translated_{tgt_code}"] = pred
            predictions.append(out_rec)

    spbleu, chrfpp = compute_corpus_metrics(hyps, refs) if n else (0.0, 0.0)

    return {
        "num_examples": n,
        "spBLEU": spbleu,
        "chrF++": chrfpp,
        "predictions": predictions,
    }


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="MT evaluation (spBLEU / chrF++) for Nano-Macro-Instruct")
    parser.add_argument("--model_name_or_path", type=str, default="Nano-Macro-Instruct")
    parser.add_argument("--data_root", type=str, default="../../data/mt_translation",
                         help="Path to data/mt_translation")
    parser.add_argument("--output_dir", type=str, default="./result")
    parser.add_argument("--pairs", type=str, nargs="+", default=MT_PAIRS,
                         help="Subset of language pairs to evaluate, e.g. vi-zh hi-ur")
    parser.add_argument("--batch_size", type=int, default=128)
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
    args = parser.parse_args()
    use_chat_template = not args.no_chat_template

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

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

    results = OrderedDict()
    t0 = time.time()

    for pair in tqdm(args.pairs, desc="pairs"):
        src_code, tgt_code = pair.split("-")
        data_path = os.path.join(args.data_root, f"{pair}.json")
        if not os.path.exists(data_path):
            print(f"[WARN] Skipping '{pair}': {data_path} not found")
            continue

        print(f"\n=== Evaluating pair: {pair} ({lang_name(src_code)} -> {lang_name(tgt_code)}) ===")
        records = load_mt_pair(data_path, src_code, tgt_code)
        pair_result = evaluate_pair(
            model, tokenizer, pair, records, src_code, tgt_code,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            use_chat_template=use_chat_template,
            limit=args.limit,
        )
        results[pair] = pair_result
        print(
            f"  {pair}: spBLEU={pair_result['spBLEU']:.2f}  chrF++={pair_result['chrF++']:.2f}  "
            f"(n={pair_result['num_examples']})"
        )

        # Save per-pair predictions immediately (safe against crashes on later pairs)
        pred_path = os.path.join(args.output_dir, f"{pair}_predictions.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(pair_result["predictions"], f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------- #
    # Aggregate CSV report (overall = macro-average across pairs)
    # ------------------------------------------------------------------- #
    if not results:
        print("No language pairs were evaluated (no data found). Exiting.")
        sys.exit(1)

    csv_path = os.path.join(args.output_dir, "mt_summary.csv")
    fieldnames = ["pair", "src_lang", "tgt_lang", "num_examples", "spBLEU", "chrF++"]
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
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"\nSaved per-pair predictions and mt_summary.csv to: {args.output_dir}")


if __name__ == "__main__":
    main()