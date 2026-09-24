"""
eval/english-task-only/Qwen1.5-MoE-A2.7B/XNLI.py

Zero-shot cross-lingual XNLI evaluation cho model Qwen1.5-MoE-A2.7B-SNLI-Task-Only.

Boi canh
--------
Model KHONG duoc finetune tren XNLI. Model chi duoc LoRA-finetune tren SNLI (tieng Anh),
theo dung format prompt/label trong `training/english-task-only/Qwen1.5-MoE-A2.7B-SNLI.py`
(xem lai file do de doi chieu):

    Premise: <premise>
    Hypothesis: <hypothesis>
    Question: What is the relationship between the premise and the hypothesis? \
Choose one: entailment, neutral, or contradiction.
    Answer: <entailment|neutral|contradiction><eos>

Checkpoint duoc push len HF Hub CHI la 1 LoRA adapter (model.save_pretrained() cua mot
PeftModel), KHONG phai full model da merge — dung y het `save_checkpoint()` /
`push_to_hub()` trong file training. Vi vay de eval, script nay:

    1) Load base model goc (Qwen/Qwen1.5-MoE-A2.7B) tu HF.
    2) Load LoRA adapter tu repo checkpoint (--adapter_repo_id) bang PeftModel.from_pretrained.
    3) (Mac dinh) merge_and_unload() de suy luan nhanh hon, giong nhu 1 model thuong.

Eval la ZERO-SHOT CROSS-LINGUAL: giu NGUYEN instruction tieng Anh (giong luc train SNLI),
CHI thay premise/hypothesis bang van ban cua tung ngon ngu XNLI (ar, bg, de, el, en, es,
fr, hi, ru, sw, th, tr, ur, vi, zh, ...) — model chua tung thay cac ngon ngu nay luc train.

Phuong phap scoring: giong `XNLI_eval.py` mau (vanilla model) — KHONG generate tu do, ma tinh
log-likelihood (length-normalized) cua 3 candidate continuation tuong ung 3 nhan, roi chon
candidate co log-likelihood trung binh cao nhat. Khac biet DUY NHAT so voi file mau: 3
candidate o day la " entailment" / " neutral" / " contradiction" (thay vi " True"/"Neither"/
" False") de khop CHINH XAC voi label_word ma model da duoc day cung train (build_full_text
trong file training noi prompt va label_word bang 1 khoang trang: f"{prompt} {label_word}").

Du lieu: chi doc `test.json` trong tung thu muc ngon ngu duoi `data/downstream/xnli/<lang>/`
(cung cau truc thu muc nhu trong XNLI_eval.py mau).

Cach chay
---------
    python eval/english-task-only/Qwen1.5-MoE-A2.7B/XNLI.py \
        --base_model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --adapter_repo_id ducanhdinh/Qwen1.5-MoE-A2.7B-SNLI-Task-Only \
        --data_root data/downstream/xnli \
        --batch_size 8 \
        --output_dir eval/english-task-only/Qwen1.5-MoE-A2.7B/results/xnli

    # Smoke-test nhanh tren vai example, chi 2 ngon ngu:
    python eval/english-task-only/Qwen1.5-MoE-A2.7B/XNLI.py --languages en vi --max_examples 20

Ghi chu
-------
- Neu repo adapter la private, truyen --hf_token hoac set bien moi truong HF_TOKEN /
  HUGGINGFACE_HUB_TOKEN / HUGGING_FACE_HUB_TOKEN.
- Can GPU du VRAM de hold base model Qwen1.5-MoE-A2.7B (~14.3B tong tham so, ~2.7B active/
  token) o bf16 (~28GB). --dtype float16/float32 se ton nhieu VRAM hon.
- --no_merge_adapter de giu adapter tach rieng (PeftModel) thay vi merge vao base weights
  (huu ich neu muon debug hoac neu merge bi loi voi mot so kien truc MoE dac thu).
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

# ============================================================================================
# Nhan XNLI/SNLI chuan: 0 = entailment, 1 = neutral, 2 = contradiction.
# CANDIDATES phai khop CHINH XAC voi label_word dung luc train SNLI (co 1 khoang trang dau,
# vi build_full_text() trong file training noi prompt + " " + label_word).
# ============================================================================================
LABEL_NAMES = ["entailment", "neutral", "contradiction"]
CANDIDATES = [" entailment", " neutral", " contradiction"]  # index-aligned voi LABEL_NAMES / label ids 0,1,2

_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_hf_token(cli_token):
    if cli_token:
        return cli_token
    for var in _HF_TOKEN_ENV_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return None


def build_prompt(premise: str, hypothesis: str) -> str:
    """GIONG HET build_prompt() trong training/english-task-only/Qwen1.5-MoE-A2.7B-SNLI.py.
    Chi instruction tieng Anh giu nguyen; premise/hypothesis la van ban goc cua ngon ngu XNLI
    dang eval (zero-shot cross-lingual transfer, khong dich sang tieng Anh)."""
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Question: What is the relationship between the premise and the hypothesis? "
        f"Choose one: entailment, neutral, or contradiction.\n"
        f"Answer:"
    )


def load_test_data(lang_dir: str):
    path = os.path.join(lang_dir, "test.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Chiu duoc mot vai dang JSON pho bien (giong XNLI_eval.py mau).
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
    """
    Tinh log-likelihood trung binh (length-normalized) cua tung candidate continuation
    (co the nhieu token, khac voi " True"/" False"/" Neither" cua vanilla eval) cho 1 batch
    prompt.

    Tra ve: numpy array shape (len(prompts), len(CANDIDATES)).
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
    log_probs = F.log_softmax(outputs.logits.float(), dim=-1)  # (N, T, V)

    seq_lens = attention_mask.sum(dim=1).tolist()  # do dai thuc (khong tinh padding) tung dong, gia dinh right-padding
    n = input_ids.shape[0]

    scores = torch.empty(n, dtype=torch.float32)
    for i in range(n):
        ctx_len = context_lens[i]
        real_len = int(seq_lens[i])
        if real_len <= ctx_len:
            scores[i] = float("-inf")
            continue
        token_ids = input_ids[i, ctx_len:real_len]                     # token cua candidate
        pred_log_probs = log_probs[i, ctx_len - 1 : real_len - 1, :]   # logits du doan cac token do
        gathered = pred_log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        scores[i] = gathered.mean().item()  # log-likelihood trung binh (length-normalized)

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
    parser = argparse.ArgumentParser(description="Zero-shot XNLI eval cho Qwen1.5-MoE-A2.7B-SNLI-Task-Only (LoRA)")

    # Model: base model goc + LoRA adapter checkpoint da push len HF Hub luc train SNLI.
    parser.add_argument("--base_model_name_or_path", default="Qwen/Qwen1.5-MoE-A2.7B",
                         help="Model goc (PHAI khop voi model dung luc train LoRA).")
    parser.add_argument("--adapter_repo_id", default="ducanhdinh/Qwen1.5-MoE-A2.7B-SNLI-Task-Only",
                         help="HF Hub repo (hoac duong dan local) chua LoRA adapter da push luc train SNLI.")
    parser.add_argument("--adapter_revision", default=None,
                         help="Branch/tag/commit cu the cua adapter repo, None = mac dinh (main).")
    parser.add_argument("--merge_adapter", action="store_true", default=True,
                         help="Merge LoRA vao base weights truoc khi eval (nhanh hon, mac dinh True).")
    parser.add_argument("--no_merge_adapter", dest="merge_adapter", action="store_false",
                         help="Giu adapter tach rieng (PeftModel) thay vi merge_and_unload().")
    parser.add_argument("--hf_token", default=None, help="HF token (repo private); mac dinh doc tu bien moi truong.")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    # Data / eval
    parser.add_argument("--data_root", default="data/downstream/xnli")
    parser.add_argument("--languages", nargs="+", default=None,
                         help="subset ngon ngu, mac dinh = tat ca thu muc tim thay trong data_root")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=None, help="debug: gioi han so example moi ngon ngu")
    parser.add_argument("--output_dir", default="eval/english-task-only/Qwen1.5-MoE-A2.7B/results/xnli")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_predictions", action="store_true", help="dump du doan tung example ra CSV")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hf_token = load_hf_token(args.hf_token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    # ---------------------------------------------------------------------------------- model
    print(f"Loading tokenizer + base model: {args.base_model_name_or_path} (device={device}, dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # bat buoc cho logic slicing log-likelihood o tren

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        trust_remote_code=args.trust_remote_code,
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

    # ------------------------------------------------------------------------------- languages
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
        "adapter_repo_id": args.adapter_repo_id,
        "adapter_revision": args.adapter_revision,
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