"""
Zero-shot XNLI evaluation cho ban LoRA-finetuned cua Macro-Nano-Instruct.

Khac voi eval/Macro-Nano-Instruct/XNLI_eval.py (danh gia thang base model),
file nay:
    1. Load base model tu --base_model_name_or_path (mac dinh
       "ATH-MaaS/Marco-Nano-Instruct", giong --model_name_or_path mac dinh
       trong Macro-Nano-Instruct.py).
    2. Gan (attach) LoRA adapter tu --lora_adapter_id, pull thang tu
       Hugging Face Hub (mac dinh "ducanhdinh/Macro-Nano-Instruct-Finetuning",
       giong --hub_model_id mac dinh trong Macro-Nano-Instruct.py -- day
       cung la noi Macro-Nano-Instruct.py push checkpoint LoRA moi nhat len,
       ghi de tai repo root moi lan push_to_hub() nen luon la ban moi nhat).
    3. Ket qua duoc luu vao --output_dir rieng (mac dinh
       "eval/finetuning/Macro-Nano-Instruct/results"), tach biet voi ket qua
       cua base model, vi script nay nam sau hon 1 cap trong eval/finetuning/.

Phuong phap cham diem (log-likelihood scoring) giu nguyen 100% logic cua
eval/Macro-Nano-Instruct/XNLI_eval.py: build 1 prompt / cap (premise,
hypothesis) bang chat template cua tokenizer, cham log-likelihood da chuan
hoa do dai cho 3 tu tiep dien "True" / "Neither" / "False" (moi tu deu thu
ca bien co-space va khong-space, lay max), khong sampling / khong free
generation.

Usage
-----
    python eval/finetuning/Macro-Nano-Instruct/XNLI_eval.py \
        --base_model_name_or_path ATH-MaaS/Marco-Nano-Instruct \
        --lora_adapter_id ducanhdinh/Macro-Nano-Instruct-Finetuning \
        --data_root data/downstream/xnli \
        --batch_size 8 \
        --output_dir eval/finetuning/Macro-Nano-Instruct/results

Neu repo LoRA adapter (hoac base model) la private, truyen token qua
--hf_token, bien moi truong HF_TOKEN / HUGGINGFACE_HUB_TOKEN /
HUGGING_FACE_HUB_TOKEN, hoac file --env_file (mac dinh ".env", doc bang
python-dotenv) -- dung chung quy uoc voi Macro-Nano-Instruct.py.

Notes
-----
- `--languages` de gioi han mot vai ngon ngu, vd `--languages en vi zh`.
- `--max_examples` de smoke-test nhanh truoc khi chay full.
- `--no_chat_template` tat chat formatting, dung raw-prompt kieu base model.
- `--merge_adapter` gop trong so LoRA vao base model (merge_and_unload)
  truoc khi suy luan -- suy luan nhanh hon mot chut, ket qua numeric giong
  het truong hop khong merge.
- `--lora_revision` de ghim mot commit/branch/tag cu the tren Hub thay vi
  luon lay ban moi nhat.
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


# XNLI label id -> label name
LABEL_NAMES = ["entailment", "neutral", "contradiction"]

# Cho moi label, cac bien the tu tiep dien se cham (co-space / khong-space);
# giu bien nao model cho diem cao hon.
CANDIDATE_VARIANTS = {
    "entailment": [" True", "True"],
    "neutral": [" Neither", "Neither"],
    "contradiction": [" False", "False"],
}
_FLAT_CANDIDATES = [(label, variant) for label in LABEL_NAMES for variant in CANDIDATE_VARIANTS[label]]

# Cac ten bien moi truong pho bien cho HF token, thu theo thu tu nay
# (dung chung quy uoc voi training/finetuning/Macro-Nano-Instruct.py).
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_hf_token(env_file, cli_token):
    """Uu tien: --hf_token (CLI) > bien moi truong da set san > file .env."""
    if cli_token:
        print("[INFO] Dung HF token truyen qua --hf_token.")
        return cli_token

    for var in _HF_TOKEN_ENV_VARS:
        if os.environ.get(var):
            print(f"[INFO] Dung HF token co san trong bien moi truong {var}.")
            return os.environ[var]

    if env_file and os.path.exists(env_file):
        if not DOTENV_AVAILABLE:
            print(
                f"[WARN] Tim thay {env_file} nhung chua cai python-dotenv "
                f"(pip install python-dotenv --break-system-packages) -> khong the tu dong doc HF_TOKEN."
            )
            return None
        load_dotenv(env_file, override=False)
        for var in _HF_TOKEN_ENV_VARS:
            if os.environ.get(var):
                print(f"[INFO] Da nap HF token tu {env_file} (bien {var}).")
                return os.environ[var]
        print(f"[WARN] Da nap {env_file} nhung khong tim thay bien {_HF_TOKEN_ENV_VARS} ben trong.")
        return None

    print(
        f"[INFO] Khong tim thay HF token (khong co --hf_token, bien moi truong, hay file {env_file}). "
        f"Tiep tuc khong xac thuc -- chi hoat dong voi model/repo public."
    )
    return None


def build_prompt(premise: str, hypothesis: str, tokenizer, use_chat_template: bool = True) -> str:
    question = f"{premise}\nQuestion: {hypothesis} True, False, or Neither?"
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [
            {
                "role": "user",
                "content": question + "\nAnswer with exactly one word: True, False, or Neither.",
            }
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback: raw completion-style prompt (base-model convention)
    return question + "\nAnswer:"


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


@torch.no_grad()
def score_candidates_batch(model, tokenizer, prompts, device):
    """
    Compute length-normalized log-likelihood of each label continuation for
    a batch of prompts, reduced (max) over each label's tokenization variants.

    Returns: numpy array of shape (len(prompts), len(LABEL_NAMES))
    """
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
    log_probs = F.log_softmax(outputs.logits, dim=-1)  # (N, T, V)

    seq_lens = attention_mask.sum(dim=1).tolist()  # real (unpadded) length per row, right-padding assumed
    n = input_ids.shape[0]

    flat_scores = torch.empty(n, dtype=torch.float32)
    for i in range(n):
        ctx_len = context_lens[i]
        real_len = int(seq_lens[i])
        if real_len <= ctx_len:
            flat_scores[i] = float("-inf")
            continue
        token_ids = input_ids[i, ctx_len:real_len]                     # continuation tokens
        pred_log_probs = log_probs[i, ctx_len - 1 : real_len - 1, :]   # logits that predict them
        gathered = pred_log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        flat_scores[i] = gathered.mean().item()  # length-normalized log-likelihood

    num_flat = len(_FLAT_CANDIDATES)
    flat_scores = flat_scores.view(b, num_flat)

    scores = torch.full((b, len(LABEL_NAMES)), float("-inf"))
    col = 0
    for li, label in enumerate(LABEL_NAMES):
        n_variants = len(CANDIDATE_VARIANTS[label])
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

    for i in tqdm(range(0, len(data), batch_size), desc=f"XNLI[{lang}]"):
        batch = data[i : i + batch_size]
        prompts = [build_prompt(ex["premise"], ex["hypothesis"], tokenizer, use_chat_template) for ex in batch]
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


def load_model_and_tokenizer(args, device, dtype_map, hf_token):
    print(f"Loading base model: {args.base_model_name_or_path} (device={device}, dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required by the log-likelihood slicing logic above

    use_chat_template = not args.no_chat_template
    if use_chat_template and not getattr(tokenizer, "chat_template", None):
        print("[WARN] Tokenizer has no chat_template; falling back to raw completion-style prompts.")
        use_chat_template = False
    elif use_chat_template:
        print("[INFO] Using tokenizer chat_template to format prompts (Instruct-model mode).")

    model_kwargs = dict(torch_dtype=dtype_map[args.dtype], trust_remote_code=args.trust_remote_code, token=hf_token)
    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_name_or_path, **model_kwargs)

    revision_note = f" (revision={args.lora_revision})" if args.lora_revision else " (latest)"
    print(f"Attaching LoRA adapter: {args.lora_adapter_id}{revision_note}")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_adapter_id,
        revision=args.lora_revision,
        token=hf_token,
    )
    if args.merge_adapter:
        print("[INFO] Merging LoRA weights into base model (merge_and_unload) ...")
        model = model.merge_and_unload()

    model.eval()
    if device == "cpu":
        model.to(device)

    return tokenizer, model, use_chat_template


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name_or_path", default="ATH-MaaS/Marco-Nano-Instruct")
    parser.add_argument("--lora_adapter_id", default="ducanhdinh/Macro-Nano-Instruct-Finetuning")
    parser.add_argument("--lora_revision", default=None, help="commit/branch/tag cu the tren Hub, None = moi nhat")
    parser.add_argument("--merge_adapter", action="store_true", help="gop LoRA vao base model truoc khi suy luan")
    parser.add_argument("--data_root", default="data/downstream/xnli")
    parser.add_argument("--languages", nargs="+", default=None, help="subset of languages, default = all found")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=None, help="debug: limit examples per language")
    parser.add_argument("--output_dir", default="eval/finetuning/Macro-Nano-Instruct/results")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_predictions", action="store_true", help="dump per-example predictions to CSV")
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat-template formatting and use raw completion-style prompts instead (base-model style).",
    )
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--hf_token", default=None, help="Override HF token thu cong")
    parser.add_argument("--env_file", default=".env", help="Duong dan file .env chua HF_TOKEN")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    hf_token = load_hf_token(args.env_file, args.hf_token)

    tokenizer, model, use_chat_template = load_model_and_tokenizer(args, device, dtype_map, hf_token)

    languages = args.languages or sorted(
        d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))
    )
    print(f"Languages to evaluate ({len(languages)}): {languages}")

    results = {}
    for lang in languages:
        acc, total, records = evaluate_language(
            model, tokenizer, lang, args.data_root, device, args.batch_size, use_chat_template, args.max_examples
        )
        results[lang] = {"accuracy": acc, "n_examples": total}
        print(f"[{lang}] accuracy = {acc:.4f}  ({total} examples)")

        if args.save_predictions:
            pd.DataFrame(records).to_csv(
                os.path.join(args.output_dir, f"xnli_predictions_{lang}.csv"), index=False
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
    csv_path = os.path.join(args.output_dir, "xnli_results.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "base_model": args.base_model_name_or_path,
        "lora_adapter": args.lora_adapter_id,
        "lora_revision": args.lora_revision,
        "merged": args.merge_adapter,
        "per_language": results,
        "overall_micro_accuracy": overall_micro_acc,
        "macro_average_accuracy": macro_acc,
    }
    json_path = os.path.join(args.output_dir, "xnli_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()