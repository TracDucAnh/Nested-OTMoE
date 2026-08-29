"""
Zero-shot XNLI evaluation for Qwen1.5-MoE-A2.7B + LoRA adapter (finetuned checkpoint)
via log-likelihood scoring.

This is the LoRA-finetuned counterpart of `eval/Qwen1.5-MoE-A2.7B/XNLI_eval.py`. The
scoring method (prompt, candidates, length-normalized log-likelihood, metrics) is
IDENTICAL to that script -- the only differences are:

  1. Model loading: we load the base model `Qwen/Qwen1.5-MoE-A2.7B` and then attach the
     LoRA adapter checkpoint that was pushed to the Hugging Face Hub by
     `training/finetuning/Qwen1.5-MoE-A2.7B.py` (default repo id
     "ducanhdinh/Qwen1.5-MoE-A2.7B-Finetuning", same default as `--hub_model_id` in that
     training script).
  2. This file lives one level deeper (`eval/finetuning/Qwen1.5-MoE-A2.7B/`), so the
     default `--output_dir` points at `eval/finetuning/Qwen1.5-MoE-A2.7B/results` and
     result filenames get a `_lora` suffix, so a base-model run and a LoRA run never
     overwrite each other's output.

Method
------
For every (premise, hypothesis) pair we build ONE prompt (English instruction,
premise/hypothesis kept in the original language) and score three candidate
continuations that correspond to the three XNLI labels:

    label 0 (entailment)   -> " True"
    label 1 (neutral)      -> " Neither"
    label 2 (contradiction)-> " False"

For each candidate we compute the *length-normalized* log-likelihood of the
continuation tokens given the prompt (teacher forcing, no sampling / no free
generation). The candidate with the highest average log-prob is the model's
prediction.

Only `test.json` is used for every language folder under
`data/downstream/xnli/<lang>/`.

Usage
-----
    python XNLI_eval.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --lora_model_id ducanhdinh/Qwen1.5-MoE-A2.7B-Finetuning \
        --data_root data/downstream/xnli \
        --batch_size 8 \
        --output_dir eval/finetuning/Qwen1.5-MoE-A2.7B/results

(run from the repo root, same convention as `eval/Qwen1.5-MoE-A2.7B/XNLI_eval.py`)

Notes
-----
- Requires a GPU with enough VRAM to hold the base model (Qwen1.5-MoE-A2.7B has
  ~14.3B total params / 2.7B activated, so budget ~28GB in bf16) plus the (small) LoRA
  adapter. Falls back to CPU automatically but will be very slow.
- `--lora_model_id` / `--lora_revision` select which Hub repo/commit the adapter is
  pulled from. If the repo is private, pass `--hf_token` or set `HF_TOKEN` (or put it in
  `--env_file`, default `.env`), same convention as the training script.
- `--merge_lora` merges the adapter into the base weights after loading (slightly
  faster forward passes); off by default so the run always reflects the adapter exactly
  as published, un-merged.
- `--languages` lets you restrict to a subset, e.g. `--languages en vi zh`.
- `--max_examples` is handy to smoke-test the pipeline on a few examples before
  launching the full run.
"""

import argparse
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# XNLI label id -> candidate continuation text (leading space matters for BPE tokenizers)
LABEL_NAMES = ["entailment", "neutral", "contradiction"]
CANDIDATES = [" True", " Neither", " False"]  # index-aligned with LABEL_NAMES / label ids 0,1,2

# Same env var names / precedence used by training/finetuning/Qwen1.5-MoE-A2.7B.py
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


def build_prompt(premise: str, hypothesis: str) -> str:
    return f"{premise}\nQuestion: {hypothesis} True, False, or Neither?\nAnswer:"


def load_test_data(lang_dir: str):
    path = os.path.join(lang_dir, "test.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Be tolerant of a few common JSON shapes.
    if isinstance(data, dict):
        for key in ("data", "examples", "rows", "test"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Unrecognized test.json structure at {path}")
    return data


def load_base_model_with_lora(args, device):
    """Load the base Qwen1.5-MoE-A2.7B model + tokenizer, then attach the LoRA adapter
    checkpoint published on the Hugging Face Hub by the finetuning script."""
    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required by the log-likelihood slicing logic below

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    print(f"Loading base model: {args.model_name_or_path} (device={device}, dtype={args.dtype})")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
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


@torch.no_grad()
def score_candidates_batch(model, tokenizer, prompts, device):
    """
    Compute length-normalized log-likelihood of each candidate continuation
    for a batch of prompts.

    Returns: numpy array of shape (len(prompts), len(CANDIDATES))
    """
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
    log_probs = F.log_softmax(outputs.logits, dim=-1)  # (N, T, V)

    seq_lens = attention_mask.sum(dim=1).tolist()  # real (unpadded) length per row, right-padding assumed
    n = input_ids.shape[0]

    scores = torch.empty(n, dtype=torch.float32)
    for i in range(n):
        ctx_len = context_lens[i]
        real_len = int(seq_lens[i])
        if real_len <= ctx_len:
            scores[i] = float("-inf")
            continue
        token_ids = input_ids[i, ctx_len:real_len]                     # continuation tokens
        pred_log_probs = log_probs[i, ctx_len - 1 : real_len - 1, :]   # logits that predict them
        gathered = pred_log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        scores[i] = gathered.mean().item()  # length-normalized log-likelihood

    return scores.view(b, num_cand).numpy()


def evaluate_language(model, tokenizer, lang, data_root, device, batch_size, max_examples=None):
    lang_dir = os.path.join(data_root, lang)
    data = load_test_data(lang_dir)
    if max_examples is not None:
        data = data[:max_examples]

    correct, total = 0, 0
    records = []

    for i in tqdm(range(0, len(data), batch_size), desc=f"XNLI[{lang}]"):
        batch = data[i : i + batch_size]
        prompts = [build_prompt(ex["premise"], ex["hypothesis"]) for ex in batch]
        scores = score_candidates_batch(model, tokenizer, prompts, device)
        preds = scores.argmax(axis=1)

        for ex, pred in zip(batch, preds):
            gold = int(ex["label"])
            is_correct = int(pred) == gold
            correct += is_correct
            total += 1
            records.append(
                {
                    "premise": ex["premise"],
                    "hypothesis": ex["hypothesis"],
                    "gold_label": LABEL_NAMES[gold],
                    "pred_label": LABEL_NAMES[int(pred)],
                    "correct": is_correct,
                }
            )

    acc = correct / total if total > 0 else 0.0
    return acc, total, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen1.5-MoE-A2.7B",
                         help="Base model to load before attaching the LoRA adapter.")
    parser.add_argument("--lora_model_id", default="ducanhdinh/Qwen1.5-MoE-A2.7B-Finetuning",
                         help="Hugging Face Hub repo id of the LoRA adapter checkpoint "
                              "(same default as --hub_model_id in the finetuning script).")
    parser.add_argument("--lora_revision", default=None,
                         help="Specific Hub revision/commit/branch of the adapter, default = latest (main).")
    parser.add_argument("--merge_lora", action="store_true",
                         help="Merge the LoRA weights into the base model after loading.")
    parser.add_argument("--hf_token", type=str, default=None,
                         help="HF Hub token, needed if --lora_model_id is a private repo. "
                              "Overrides env var / .env if set.")
    parser.add_argument("--env_file", type=str, default=".env",
                         help="Path to a .env file containing HF_TOKEN, auto-loaded via python-dotenv.")
    parser.add_argument("--data_root", default="data/downstream/xnli")
    parser.add_argument("--languages", nargs="+", default=None, help="subset of languages, default = all found")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=None, help="debug: limit examples per language")
    parser.add_argument("--output_dir", default="eval/finetuning/Qwen1.5-MoE-A2.7B/results")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_predictions", action="store_true", help="dump per-example predictions to CSV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer, model = load_base_model_with_lora(args, device)

    languages = args.languages or sorted(
        d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))
    )
    print(f"Languages to evaluate ({len(languages)}): {languages}")

    results = {}
    for lang in languages:
        acc, total, records = evaluate_language(
            model, tokenizer, lang, args.data_root, device, args.batch_size, args.max_examples
        )
        results[lang] = {"accuracy": acc, "n_examples": total}
        print(f"[{lang}] accuracy = {acc:.4f}  ({total} examples)")

        if args.save_predictions:
            pd.DataFrame(records).to_csv(
                os.path.join(args.output_dir, f"xnli_predictions_{lang}_lora.csv"), index=False
            )

    overall_correct = sum(r["accuracy"] * r["n_examples"] for r in results.values())
    overall_total = sum(r["n_examples"] for r in results.values())
    overall_micro_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    macro_acc = sum(r["accuracy"] for r in results.values()) / len(results) if results else 0.0

    print("=" * 60)
    print(f"Overall (micro, weighted by #examples) accuracy: {overall_micro_acc:.4f}")
    print(f"Macro-average (mean over languages) accuracy:    {macro_acc:.4f}")

    df = pd.DataFrame(
        [{"language": lang, "accuracy": r["accuracy"], "n_examples": r["n_examples"]} for lang, r in results.items()]
    ).sort_values("language")
    csv_path = os.path.join(args.output_dir, "xnli_results_lora.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "base_model": args.model_name_or_path,
        "lora_model_id": args.lora_model_id,
        "lora_revision": args.lora_revision or "main",
        "merged": args.merge_lora,
        "per_language": results,
        "overall_micro_accuracy": overall_micro_acc,
        "macro_average_accuracy": macro_acc,
    }
    json_path = os.path.join(args.output_dir, "xnli_results_lora.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()