"""
Zero-shot MMMLU (Multilingual MMLU) evaluation for Nano-Macro-Instruct via
log-likelihood scoring.

Method
------
Nano-Macro-Instruct is a chat/instruct-tuned model, so the question is
formatted with the tokenizer's chat template (system/user turn +
`add_generation_prompt=True`) rather than the raw completion-style prompt
used for base models:

    {Question}
    A. {A}
    B. {B}
    C. {C}
    D. {D}
    Answer with the letter of the correct option only.

We score the continuations for "A", "B", "C", "D" with teacher-forced
log-likelihood (no free generation) and take the argmax as the prediction,
compared against the gold `Answer` field. Because whether a leading space
belongs to the first generated token can differ between base and
chat-template tokenizers (and right after chat special tokens), we score
both a space-prefixed and a bare variant for each letter and keep the max
per letter -- this keeps the scoring robust without needing to know the
exact tokenizer/template quirks of Nano-Macro-Instruct ahead of time.

If the tokenizer has no chat_template (or --no_chat_template is passed),
the script falls back automatically to the original raw "Answer:"-style
prompt.

Only `test.json` is used for every language folder under
`data/downstream/mmmlu/<LANG>/` (e.g. AR_XY, DE_DE, ZH_CN, ...).

Usage
-----
    python MMMLU_eval.py \
        --model_name_or_path Nano-Macro-Instruct \
        --data_root data/downstream/mmmlu \
        --batch_size 8 \
        --output_dir eval/Nano-Macro-Instruct/results

Besides per-language + overall accuracy (same convention as XNLI_eval.py),
this script also dumps a per-subject breakdown (aggregated across all
languages) to `mmmlu_subject_results.csv`, since MMLU-style benchmarks are
commonly reported both by language and by subject.
"""

import argparse
import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]

# For each letter, the candidate continuation variants we'll score. We take
# the leading-space and no-space forms and keep whichever the tokenizer /
# model scores higher, so this works whether or not the tokenizer treats a
# leading space as part of the first generated token.
CANDIDATE_VARIANTS = {
    "A": [" A", "A"],
    "B": [" B", "B"],
    "C": [" C", "C"],
    "D": [" D", "D"],
}
_FLAT_CANDIDATES = [(letter, variant) for letter in LETTERS for variant in CANDIDATE_VARIANTS[letter]]


def build_prompt(ex: dict, tokenizer, use_chat_template: bool = True) -> str:
    question_block = (
        f"{ex['Question']}\n"
        f"A. {ex['A']}\n"
        f"B. {ex['B']}\n"
        f"C. {ex['C']}\n"
        f"D. {ex['D']}"
    )
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [
            {
                "role": "user",
                "content": question_block + "\n\nAnswer with the letter of the correct option only.",
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback: raw completion-style prompt (base-model convention)
    return question_block + "\nAnswer:"


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
    """Length-normalized log-likelihood of each letter for a batch of prompts,
    reduced (max) over the space/no-space tokenization variants of that letter.
    Returns numpy array of shape (len(prompts), len(LETTERS))."""
    b = len(prompts)

    all_texts = []
    context_lens = []
    for p in prompts:
        ctx_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
        for _, variant in _FLAT_CANDIDATES:
            all_texts.append(p + variant)
            context_lens.append(len(ctx_ids))

    enc = tokenizer(all_texts, add_special_tokens=False, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = F.log_softmax(outputs.logits, dim=-1)

    seq_lens = attention_mask.sum(dim=1).tolist()
    n = input_ids.shape[0]
    flat_scores = torch.empty(n, dtype=torch.float32)

    for i in range(n):
        ctx_len = context_lens[i]
        real_len = int(seq_lens[i])
        if real_len <= ctx_len:
            flat_scores[i] = float("-inf")
            continue
        token_ids = input_ids[i, ctx_len:real_len]
        pred_log_probs = log_probs[i, ctx_len - 1 : real_len - 1, :]
        gathered = pred_log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        flat_scores[i] = gathered.mean().item()

    num_flat = len(_FLAT_CANDIDATES)
    flat_scores = flat_scores.view(b, num_flat)

    scores = torch.full((b, len(LETTERS)), float("-inf"))
    col = 0
    for li, letter in enumerate(LETTERS):
        n_variants = len(CANDIDATE_VARIANTS[letter])
        group = flat_scores[:, col : col + n_variants]
        scores[:, li] = group.max(dim=1).values
        col += n_variants

    return scores.numpy()


def evaluate_language(model, tokenizer, lang, data_root, device, batch_size, use_chat_template, max_examples=None):
    lang_dir = os.path.join(data_root, lang)
    data = load_test_data(lang_dir)
    if max_examples is not None:
        data = data[:max_examples]

    correct, total = 0, 0
    records = []

    for i in tqdm(range(0, len(data), batch_size), desc=f"MMMLU[{lang}]"):
        batch = data[i : i + batch_size]
        prompts = [build_prompt(ex, tokenizer, use_chat_template) for ex in batch]
        scores = score_candidates_batch(model, tokenizer, prompts, device)
        preds = scores.argmax(axis=1)

        for ex, pred in zip(batch, preds):
            gold_letter = str(ex["Answer"]).strip()
            pred_letter = LETTERS[int(pred)]
            is_correct = pred_letter == gold_letter
            correct += is_correct
            total += 1
            records.append(
                {
                    "subject": ex.get("Subject", ""),
                    "question": ex["Question"],
                    "gold": gold_letter,
                    "pred": pred_letter,
                    "correct": is_correct,
                }
            )

    acc = correct / total if total > 0 else 0.0
    return acc, total, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Nano-Macro-Instruct")
    parser.add_argument("--data_root", default="data/downstream/mmmlu")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", default="eval/Nano-Macro-Instruct/results")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_predictions", action="store_true")
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

    print(f"Loading model: {args.model_name_or_path} (device={device}, dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if use_chat_template and not getattr(tokenizer, "chat_template", None):
        print("[WARN] Tokenizer has no chat_template; falling back to raw completion-style prompts.")
        use_chat_template = False
    elif use_chat_template:
        print("[INFO] Using tokenizer chat_template to format prompts (Instruct-model mode).")

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
    for lang in languages:
        acc, total, records = evaluate_language(
            model, tokenizer, lang, args.data_root, device, args.batch_size, use_chat_template, args.max_examples
        )
        results[lang] = {"accuracy": acc, "n_examples": total}
        print(f"[{lang}] accuracy = {acc:.4f}  ({total} examples)")
        for r in records:
            r["language"] = lang
        all_records.extend(records)

        if args.save_predictions:
            pd.DataFrame(records).to_csv(
                os.path.join(args.output_dir, f"mmmlu_predictions_{lang}.csv"), index=False
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
    }
    json_path = os.path.join(args.output_dir, "mmmlu_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()