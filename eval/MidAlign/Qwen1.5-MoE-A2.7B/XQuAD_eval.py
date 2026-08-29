#!/usr/bin/env python3
"""
Zero-shot generative evaluation of the MidAlign baseline (LoRA) of
Qwen1.5-MoE-A2.7B on XQuAD.

This is the MidAlign-baseline counterpart of `eval/Qwen1.5-MoE-A2.7B/XQuAD_eval.py`
(base model) and a sibling of the plain-Finetuning version at
`eval/finetuning/Qwen1.5-MoE-A2.7B/XQuAD_eval.py`. The
scoring method (zero-shot generation, SQuAD-style EM / F1 with multilingual
normalization) is IDENTICAL to that script -- the only differences are:

  1. Model loading: we load the base model `Qwen/Qwen1.5-MoE-A2.7B` and then attach the
     LoRA adapter checkpoint that was pushed to the Hugging Face Hub by
     `training/MidAlign/Qwen1.5-MoE-A2.7B.py` (default repo id
     "ducanhdinh/Qwen1.5-MoE-A2.7B-MidAlign", same default as `--hub_model_id` in that
     training script).
  2. This file lives one level deeper (`eval/MidAlign/Qwen1.5-MoE-A2.7B/` instead of
     `eval/Qwen1.5-MoE-A2.7B/`), so the default `--data_root` gains one extra `../` to
     still reach `data/downstream/xquad` from the repo root, and result filenames get a
     `_lora` suffix (`{lang}_predictions_lora.json`, `summary_lora.json`), so a
     base-model run and a LoRA run never overwrite each other's output.

For every language available under `data/downstream/xquad/<lang>/validation.json`,
the model is prompted zero-shot (no in-context examples) to *generate* an answer
given the (context, question) pair. Generated answers are scored against the gold
answers with standard SQuAD-style Exact Match (EM) and token-level F1, using the
multilingual normalization scheme from the official XQuAD/MLQA evaluation scripts
(character-level matching for languages without whitespace word boundaries:
Chinese and Thai; whitespace-token matching otherwise).

Usage (run from eval/MidAlign/Qwen1.5-MoE-A2.7B/):
    python XQuAD_eval.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --lora_model_id ducanhdinh/Qwen1.5-MoE-A2.7B-MidAlign \
        --data_root ../../../data/downstream/xquad \
        --output_dir ./results/xquad \
        --batch_size 8 \
        --max_new_tokens 32

Only a subset of languages:
    python XQuAD_eval.py --languages en vi zh ar

Quick debug run on a handful of examples per language:
    python XQuAD_eval.py --limit 20

LoRA notes
----------
- `--lora_model_id` / `--lora_revision` select which Hub repo/commit the adapter is
  pulled from. If the repo is private, pass `--hf_token` or set `HF_TOKEN` (or put it
  in `--env_file`, default `.env`), same convention as the training script.
- `--merge_lora` merges the adapter into the base weights after loading (slightly
  faster generation); off by default so the run always reflects the adapter exactly as
  published, un-merged.
"""

import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter, OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


# ----------------------------------------------------------------------------- #
# Languages available in data/downstream/xquad/<lang>/validation.json
# ----------------------------------------------------------------------------- #
XQUAD_LANGS = ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"]

# Languages whose scripts are not whitespace-segmented: F1 is computed over
# characters instead of whitespace-split tokens (matches the official XQuAD /
# MLQA evaluation scripts).
MIXED_SEGMENTATION_LANGS = {"zh", "th"}

# Zero-shot instruction template. Kept in English on purpose: the instruction
# language is fixed while context/question vary by target language, which is
# the standard XTREME/XQuAD zero-shot cross-lingual transfer protocol -- we
# want to measure the model's ability to read/answer in each language, not
# its ability to follow instructions written in that language.
PROMPT_TEMPLATE = (
    "Answer the question using only the information in the context below. "
    "Give the shortest possible answer, copied verbatim from the context, "
    "with no explanation.\n\n"
    "Context: {context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# Same env var names / precedence used by training/MidAlign/Qwen1.5-MoE-A2.7B.py
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_hf_token(env_file, cli_token):
    """Uu tien: --hf_token (CLI) > bien moi truong da set san > file .env (qua dotenv).
    Can de load LoRA adapter neu repo tren Hugging Face Hub la private."""
    if cli_token:
        print("Using HF token passed via --hf_token.")
        return cli_token

    for var in _HF_TOKEN_ENV_VARS:
        if os.environ.get(var):
            print(f"Using HF token found in environment variable {var}.")
            return os.environ[var]

    if env_file and os.path.exists(env_file):
        if not DOTENV_AVAILABLE:
            print(
                f"[WARN] Found {env_file} but python-dotenv is not installed "
                f"(pip install python-dotenv --break-system-packages) -> cannot auto-read HF_TOKEN."
            )
            return None
        load_dotenv(env_file, override=False)
        for var in _HF_TOKEN_ENV_VARS:
            if os.environ.get(var):
                print(f"Loaded HF token from {env_file} (variable {var}).")
                return os.environ[var]
        print(f"[WARN] Loaded {env_file} but none of {_HF_TOKEN_ENV_VARS} were found inside.")
        return None

    print(
        f"No HF token found (no --hf_token, no env var, no {env_file}). "
        f"Continuing unauthenticated -- only works if the LoRA repo is public."
    )
    return None


def load_base_model_with_lora(args, device):
    """Load the base Qwen1.5-MoE-A2.7B model + tokenizer, then attach the LoRA adapter
    checkpoint published on the Hugging Face Hub by the MidAlign training script."""
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched causal-LM generation

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    print(f"Loading base model: {args.model_name_or_path} (device={device}, dtype={args.dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        base_model.to(device)

    hf_token = load_hf_token(args.env_file, args.hf_token)
    print(
        f"Attaching LoRA adapter from {args.lora_model_id} "
        f"(revision={args.lora_revision or 'main'}) ..."
    )
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_model_id,
        revision=args.lora_revision,
        token=hf_token,
    )

    if args.merge_lora:
        print("Merging LoRA weights into the base model ...")
        model = model.merge_and_unload()

    model.eval()
    return tokenizer, model


# ----------------------------------------------------------------------------- #
# Data loading
# ----------------------------------------------------------------------------- #
def load_xquad_split(path):
    """
    Loads a single-language XQuAD validation file and returns a flat list of
    {id, context, question, answers} records.

    Supports both:
      - a flat list of records (the processed format used in this repo), e.g.
            {
              "id": "56beb4343aeaaa14008c925c",
              "context": "...",
              "question": "...",
              "answers": {"text": ["136"], "answer_start": [557]}
            }
      - the original nested SQuAD format {"data": [{"paragraphs": [...]}]}.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict) and "data" in raw:
        records = []
        for article in raw["data"]:
            for paragraph in article["paragraphs"]:
                context = paragraph["context"]
                for qa in paragraph["qas"]:
                    answers = qa["answers"]
                    if isinstance(answers, dict):
                        norm_answers = answers
                    else:
                        norm_answers = {
                            "text": [a["text"] for a in answers],
                            "answer_start": [a["answer_start"] for a in answers],
                        }
                    records.append(
                        {
                            "id": qa["id"],
                            "context": context,
                            "question": qa["question"],
                            "answers": norm_answers,
                        }
                    )
    else:
        raise ValueError(f"Unrecognized XQuAD file format: {path}")

    return records


# ----------------------------------------------------------------------------- #
# SQuAD-style / XQuAD-style metrics (multilingual normalization)
# ----------------------------------------------------------------------------- #
def normalize_answer(text, lang):
    """Lowercase, strip punctuation, strip English articles, collapse whitespace."""

    def remove_articles(s):
        # Article stripping is only meaningful (and only applied by the
        # official eval scripts) for English.
        if lang == "en":
            return re.sub(r"\b(a|an|the)\b", " ", s)
        return s

    def white_space_fix(s):
        return " ".join(s.split())

    def remove_punc(s):
        exclude = set(string.punctuation + "¿？，。！？：；、《》「」『』…—“”‘’·")
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s):
        return s.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def tokenize_for_f1(text, lang):
    """Character-level tokens for whitespace-free scripts, else whitespace split."""
    if lang in MIXED_SEGMENTATION_LANGS:
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def compute_em(prediction, gold, lang):
    return int(normalize_answer(prediction, lang) == normalize_answer(gold, lang))


def compute_f1(prediction, gold, lang):
    pred_tokens = tokenize_for_f1(normalize_answer(prediction, lang), lang)
    gold_tokens = tokenize_for_f1(normalize_answer(gold, lang), lang)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # F1 is 1 only if both prediction and gold are empty, else 0.
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_example(prediction, gold_answers, lang):
    """Max EM / F1 over all gold answer variants for one example (SQuAD convention)."""
    em = max(compute_em(prediction, g, lang) for g in gold_answers)
    f1 = max(compute_f1(prediction, g, lang) for g in gold_answers)
    return em, f1


# ----------------------------------------------------------------------------- #
# Generation
# ----------------------------------------------------------------------------- #
def clean_generated_answer(text):
    """Model output sometimes rambles after the answer; keep just the answer span."""
    text = text.strip()
    text = text.split("\n")[0].strip()
    for marker in ["Question:", "Context:", "Explanation:"]:
        if marker in text:
            text = text.split(marker)[0].strip()
    text = text.strip("\"'“”‘’ ")
    return text


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens, device):
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
    return [clean_generated_answer(d) for d in decoded]


# ----------------------------------------------------------------------------- #
# Per-language evaluation
# ----------------------------------------------------------------------------- #
def evaluate_language(model, tokenizer, lang, records, batch_size, max_new_tokens, device, limit=None):
    if limit:
        records = records[:limit]

    predictions = OrderedDict()
    em_total, f1_total = 0.0, 0.0
    n = len(records)

    for start in tqdm(range(0, n, batch_size), desc=f"{lang}", unit="batch"):
        batch = records[start:start + batch_size]
        prompts = [PROMPT_TEMPLATE.format(context=r["context"], question=r["question"]) for r in batch]

        preds = generate_batch(model, tokenizer, prompts, max_new_tokens, device)

        for record, pred in zip(batch, preds):
            gold_answers = record["answers"]["text"]
            em, f1 = score_example(pred, gold_answers, lang)
            em_total += em
            f1_total += f1
            predictions[record["id"]] = {
                "prediction": pred,
                "gold": gold_answers,
                "em": em,
                "f1": f1,
            }

    return {
        "num_examples": n,
        "em": 100.0 * em_total / n if n else 0.0,
        "f1": 100.0 * f1_total / n if n else 0.0,
        "predictions": predictions,
    }


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Zero-shot generative XQuAD eval for the MidAlign baseline (LoRA) of Qwen1.5-MoE-A2.7B")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B",
                         help="Base model to load before attaching the LoRA adapter.")
    parser.add_argument("--lora_model_id", type=str, default="ducanhdinh/Qwen1.5-MoE-A2.7B-MidAlign",
                         help="Hugging Face Hub repo id of the LoRA adapter checkpoint "
                              "(same default as --hub_model_id in the MidAlign training script).")
    parser.add_argument("--lora_revision", type=str, default=None,
                         help="Specific Hub revision/commit/branch of the adapter, default = latest (main).")
    parser.add_argument("--merge_lora", action="store_true",
                         help="Merge the LoRA weights into the base model after loading.")
    parser.add_argument("--hf_token", type=str, default=None,
                         help="HF Hub token, needed if --lora_model_id is a private repo. "
                              "Overrides env var / .env if set.")
    parser.add_argument("--env_file", type=str, default=".env",
                         help="Path to a .env file containing HF_TOKEN, auto-loaded via python-dotenv.")
    parser.add_argument("--data_root", type=str, default="../../../data/downstream/xquad",
                         help="Path to data/downstream/xquad (one extra '../' vs. the base-model "
                              "script, since this file lives one directory deeper).")
    parser.add_argument("--output_dir", type=str, default="./results/xquad")
    parser.add_argument("--languages", type=str, nargs="+", default=XQUAD_LANGS,
                         help="Subset of XQuAD languages to evaluate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of examples per language, for quick debugging")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = load_base_model_with_lora(args, device)

    results = OrderedDict()
    t0 = time.time()

    for lang in tqdm(args.languages, desc="languages"):
        data_path = os.path.join(args.data_root, lang, "validation.json")
        if not os.path.exists(data_path):
            print(f"[WARN] Skipping '{lang}': {data_path} not found")
            continue

        print(f"\n=== Evaluating language: {lang} ===")
        records = load_xquad_split(data_path)
        lang_result = evaluate_language(
            model, tokenizer, lang, records,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=device,
            limit=args.limit,
        )
        results[lang] = lang_result
        print(f"  {lang}: EM={lang_result['em']:.2f}  F1={lang_result['f1']:.2f}  (n={lang_result['num_examples']})")

        # Save per-language predictions immediately (safe against crashes on later langs)
        with open(os.path.join(args.output_dir, f"{lang}_predictions_lora.json"), "w", encoding="utf-8") as f:
            json.dump(lang_result["predictions"], f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------- #
    # Aggregate report (overall = macro-average across languages, the
    # standard XTREME/XQuAD convention)
    # ------------------------------------------------------------------- #
    if not results:
        print("No languages were evaluated (no data found). Exiting.")
        sys.exit(1)

    em_scores = [r["em"] for r in results.values()]
    f1_scores = [r["f1"] for r in results.values()]
    overall_em = sum(em_scores) / len(em_scores)
    overall_f1 = sum(f1_scores) / len(f1_scores)

    summary = OrderedDict()
    summary["_meta"] = {
        "base_model": args.model_name_or_path,
        "lora_model_id": args.lora_model_id,
        "lora_revision": args.lora_revision or "main",
        "merged": args.merge_lora,
    }
    for lang, r in results.items():
        summary[lang] = {"num_examples": r["num_examples"], "em": round(r["em"], 2), "f1": round(r["f1"], 2)}
    summary["overall"] = {
        "num_examples": sum(r["num_examples"] for r in results.values()),
        "em": round(overall_em, 2),
        "f1": round(overall_f1, 2),
    }

    print("\n" + "=" * 46)
    print(f"{'Language':<10}{'#Examples':<12}{'EM':<10}{'F1':<10}")
    print("-" * 46)
    for lang, r in summary.items():
        if lang == "_meta":
            continue
        tag = lang.upper() if lang == "overall" else lang
        print(f"{tag:<10}{r['num_examples']:<12}{r['em']:<10.2f}{r['f1']:<10.2f}")
    print("=" * 46)
    print(f"Total time: {time.time() - t0:.1f}s")

    with open(os.path.join(args.output_dir, "summary_lora.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved per-language predictions and summary_lora.json to: {args.output_dir}")


if __name__ == "__main__":
    main()