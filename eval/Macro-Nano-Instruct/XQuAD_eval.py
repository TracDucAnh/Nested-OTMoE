#!/usr/bin/env python3
"""
Zero-shot generative evaluation of Nano-Macro-Instruct on XQuAD.

For every language available under `data/downstream/xquad/<lang>/validation.json`,
the model is prompted zero-shot (no in-context examples) to *generate* an answer
given the (context, question) pair. Nano-Macro-Instruct is a chat/instruct-tuned
model, so the prompt is built with the tokenizer's chat template
(`add_generation_prompt=True`) rather than the raw completion-style prompt used
for base models. If the tokenizer has no chat_template (or --no_chat_template is
passed), the script falls back automatically to the original raw prompt.

Generated answers are scored against the gold answers with standard SQuAD-style
Exact Match (EM) and token-level F1, using the multilingual normalization scheme
from the official XQuAD/MLQA evaluation scripts (character-level matching for
languages without whitespace word boundaries: Chinese and Thai; whitespace-token
matching otherwise).

Usage (run from eval/Nano-Macro-Instruct/):
    python XQuAD_eval.py \
        --model_name_or_path Nano-Macro-Instruct \
        --data_root ../../data/downstream/xquad \
        --output_dir ./results/xquad \
        --batch_size 8 \
        --max_new_tokens 32

Only a subset of languages:
    python XQuAD_eval.py --languages en vi zh ar

Quick debug run on a handful of examples per language:
    python XQuAD_eval.py --limit 20

Disable chat-template formatting (base-model-style raw prompt):
    python XQuAD_eval.py --no_chat_template
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
from tqdm import tqdm


# ----------------------------------------------------------------------------- #
# Languages available in data/downstream/xquad/<lang>/validation.json
# ----------------------------------------------------------------------------- #
XQUAD_LANGS = ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"]

# Languages whose scripts are not whitespace-segmented: F1 is computed over
# characters instead of whitespace-split tokens (matches the official XQuAD /
# MLQA evaluation scripts).
MIXED_SEGMENTATION_LANGS = {"zh", "th"}

# Instruction content (same across languages by design -- see XTREME/XQuAD
# zero-shot cross-lingual transfer protocol note below). Used as the "content"
# of the user turn when chat-templating, and as the raw completion-style
# prompt body when --no_chat_template is passed.
#
# Kept in English on purpose: the instruction language is fixed while
# context/question vary by target language, which is the standard
# XTREME/XQuAD zero-shot cross-lingual transfer protocol -- we want to
# measure the model's ability to read/answer in each language, not its
# ability to follow instructions written in that language.
INSTRUCTION_TEMPLATE = (
    "Answer the question using only the information in the context below. "
    "Give the shortest possible answer, copied verbatim from the context, "
    "with no explanation.\n\n"
    "Context: {context}\n\n"
    "Question: {question}"
)

# Raw completion-style prompt used only as a fallback for base models /
# --no_chat_template runs.
PROMPT_TEMPLATE = INSTRUCTION_TEMPLATE + "\n\nAnswer:"


def build_prompt(tokenizer, context: str, question: str, use_chat_template: bool = True) -> str:
    content = INSTRUCTION_TEMPLATE.format(context=context, question=question)
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return PROMPT_TEMPLATE.format(context=context, question=question)


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
def evaluate_language(model, tokenizer, lang, records, batch_size, max_new_tokens, device,
                       use_chat_template, limit=None):
    if limit:
        records = records[:limit]

    predictions = OrderedDict()
    em_total, f1_total = 0.0, 0.0
    n = len(records)

    for start in tqdm(range(0, n, batch_size), desc=f"{lang}", unit="batch"):
        batch = records[start:start + batch_size]
        prompts = [
            build_prompt(tokenizer, r["context"], r["question"], use_chat_template) for r in batch
        ]

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
    parser = argparse.ArgumentParser(description="Zero-shot generative XQuAD eval for Nano-Macro-Instruct")
    parser.add_argument("--model_name_or_path", type=str, default="Nano-Macro-Instruct")
    parser.add_argument("--data_root", type=str, default="../../data/downstream/xquad",
                         help="Path to data/downstream/xquad")
    parser.add_argument("--output_dir", type=str, default="./results/xquad")
    parser.add_argument("--languages", type=str, nargs="+", default=XQUAD_LANGS,
                         help="Subset of XQuAD languages to evaluate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of examples per language, for quick debugging")
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
            use_chat_template=use_chat_template,
            limit=args.limit,
        )
        results[lang] = lang_result
        print(f"  {lang}: EM={lang_result['em']:.2f}  F1={lang_result['f1']:.2f}  (n={lang_result['num_examples']})")

        # Save per-language predictions immediately (safe against crashes on later langs)
        with open(os.path.join(args.output_dir, f"{lang}_predictions.json"), "w", encoding="utf-8") as f:
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
        tag = lang.upper() if lang == "overall" else lang
        print(f"{tag:<10}{r['num_examples']:<12}{r['em']:<10.2f}{r['f1']:<10.2f}")
    print("=" * 46)
    print(f"Total time: {time.time() - t0:.1f}s")

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved per-language predictions and summary.json to: {args.output_dir}")


if __name__ == "__main__":
    main()