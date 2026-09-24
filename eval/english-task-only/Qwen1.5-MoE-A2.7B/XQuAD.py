#!/usr/bin/env python3
"""
eval/english-task-only/Qwen1.5-MoE-A2.7B/XQuAD.py

Zero-shot cross-lingual XQuAD evaluation cho model Qwen1.5-MoE-A2.7B-SQuAD-Task-Only.

Boi canh
--------
Model KHONG duoc finetune tren XQuAD. Model chi duoc LoRA-finetune tren SQuAD (tieng Anh),
theo dung format prompt/answer trong `training/english-task-only/Qwen1.5-MoE-A2.7B-SQuAD.py`
(xem lai file do de doi chieu):

    Context: <context>
    Question: <question>
    Answer: <answer_text><eos>

Checkpoint duoc push len HF Hub CHI la 1 LoRA adapter (`model.save_pretrained()` cua mot
PeftModel), KHONG phai full model da merge — dung y het `save_checkpoint()` / `push_to_hub()`
trong file training (docstring cua file training o dong 79-85 con noi ro cach load lai: base
model + `PeftModel.from_pretrained(base, "ducanhdinh/Qwen1.5-MoE-A2.7B-SQuAD-Task-Only")`).
Vi vay de eval, script nay:

    1) Load base model goc (Qwen/Qwen1.5-MoE-A2.7B) tu HF.
    2) Load LoRA adapter tu repo checkpoint (--adapter_repo_id) bang PeftModel.from_pretrained.
    3) (Mac dinh) merge_and_unload() de suy luan nhanh hon, giong nhu 1 model thuong.

Eval la ZERO-SHOT CROSS-LINGUAL: giu NGUYEN prompt tieng Anh dung luc train SQuAD, CHI thay
context/question bang van ban cua tung ngon ngu XQuAD (ar, de, el, en, es, hi, ro, ru, th, tr,
vi, zh) — model chua tung thay cac ngon ngu nay luc train.

KHAC BIET DUY NHAT so voi `XQuAD_eval.py` mau (vanilla, chua finetune):
  - PROMPT_TEMPLATE o day la "Context: ...\nQuestion: ...\nAnswer:" — GIONG HET
    `build_prompt()` cua file training, KHONG dung instruction dai dong "Answer the question
    using only the information..." nhu ban vanilla. Model finetune roi thi da hoc dung format
    ngan gon nay de sinh "<answer_text><eos>", nen giu prompt luc eval GIONG luc train se cho
    ket qua dung ban chat zero-shot-transfer (chuyen tu SQuAD sang XQuAD) hon la doi prompt.
  - Con lai: cach doc du lieu XQuAD (`load_xquad_split`, ho tro ca dang list "processed" va
    dang SQuAD goc {"data": [...]}), cach chuan hoa + tinh EM/F1 da ngon ngu (character-level
    cho zh/th, whitespace-level cho cac ngon ngu con lai), generate greedy (do_sample=False),
    va cach tong hop macro-average deu GIU NGUYEN y het ban mau (day la protocol XTREME/XQuAD
    chuan, khong phu thuoc vao model).

Du lieu: chi doc `validation.json` trong tung thu muc ngon ngu duoi
`data/downstream/xquad/<lang>/` (cung cau truc thu muc nhu XQuAD_eval.py mau).

Cach chay
---------
    python eval/english-task-only/Qwen1.5-MoE-A2.7B/XQuAD.py \
        --base_model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --adapter_repo_id ducanhdinh/Qwen1.5-MoE-A2.7B-SQuAD-Task-Only \
        --data_root data/downstream/xquad \
        --output_dir eval/english-task-only/Qwen1.5-MoE-A2.7B/results/xquad \
        --batch_size 8 \
        --max_new_tokens 32

    # Chi vai ngon ngu:
    python eval/english-task-only/Qwen1.5-MoE-A2.7B/XQuAD.py --languages en vi zh ar

    # Debug nhanh tren vai example:
    python eval/english-task-only/Qwen1.5-MoE-A2.7B/XQuAD.py --limit 20

Ghi chu
-------
- Neu repo adapter la private, truyen --hf_token hoac set bien moi truong HF_TOKEN /
  HUGGINGFACE_HUB_TOKEN / HUGGING_FACE_HUB_TOKEN.
- Can GPU du VRAM de hold base model Qwen1.5-MoE-A2.7B (~14.3B tong tham so, ~2.7B active/
  token) o bf16 (~28GB).
- --no_merge_adapter de giu adapter tach rieng (PeftModel) thay vi merge_and_unload().
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


# ----------------------------------------------------------------------------- #
# Ngon ngu co san trong data/downstream/xquad/<lang>/validation.json
# ----------------------------------------------------------------------------- #
XQUAD_LANGS = ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"]

# Ngon ngu khong phan tach tu bang khoang trang: F1 tinh theo TU KY TU thay vi tu tach bang
# whitespace (dung theo cach official XQuAD/MLQA eval script).
MIXED_SEGMENTATION_LANGS = {"zh", "th"}

# Prompt GIONG HET build_prompt() trong training/english-task-only/Qwen1.5-MoE-A2.7B-SQuAD.py
# — day chinh la ly do dung prompt nay: model da duoc SFT de sinh "<answer_text><eos>" ngay
# sau "Answer:" theo dung format nay, khong phai theo instruction dai cua ban vanilla.
PROMPT_TEMPLATE = "Context: {context}\nQuestion: {question}\nAnswer:"

_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_hf_token(cli_token):
    if cli_token:
        return cli_token
    for var in _HF_TOKEN_ENV_VARS:
        if os.environ.get(var):
            return os.environ[var]
    return None


# ----------------------------------------------------------------------------- #
# Data loading (giong het XQuAD_eval.py mau)
# ----------------------------------------------------------------------------- #
def load_xquad_split(path):
    """
    Doc 1 file XQuAD validation cua 1 ngon ngu, tra ve list flat cac record
    {id, context, question, answers}.

    Ho tro ca:
      - dang list flat (dang da xu ly dung trong repo nay), vi du:
            {
              "id": "56beb4343aeaaa14008c925c",
              "context": "...",
              "question": "...",
              "answers": {"text": ["136"], "answer_start": [557]}
            }
      - dang SQuAD nested goc {"data": [{"paragraphs": [...]}]}.
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
        raise ValueError(f"Khong nhan dang duoc cau truc file XQuAD: {path}")

    return records


# ----------------------------------------------------------------------------- #
# Metric kieu SQuAD/XQuAD (chuan hoa da ngon ngu) — giong het ban mau
# ----------------------------------------------------------------------------- #
def normalize_answer(text, lang):
    """Lowercase, bo dau cau, bo mao tu tieng Anh, gon khoang trang."""

    def remove_articles(s):
        # Bo mao tu chi co y nghia (va chi duoc official eval script ap dung) voi tieng Anh.
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
    """Token theo tung ky tu voi script khong dung whitespace, con lai tach theo whitespace."""
    if lang in MIXED_SEGMENTATION_LANGS:
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def compute_em(prediction, gold, lang):
    return int(normalize_answer(prediction, lang) == normalize_answer(gold, lang))


def compute_f1(prediction, gold, lang):
    pred_tokens = tokenize_for_f1(normalize_answer(prediction, lang), lang)
    gold_tokens = tokenize_for_f1(normalize_answer(gold, lang), lang)

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # F1 = 1 chi khi CA HAI prediction va gold deu rong, con lai la 0.
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_example(prediction, gold_answers, lang):
    """Max EM / F1 tren tat ca cac dap an gold cua 1 example (quy uoc SQuAD)."""
    em = max(compute_em(prediction, g, lang) for g in gold_answers)
    f1 = max(compute_f1(prediction, g, lang) for g in gold_answers)
    return em, f1


# ----------------------------------------------------------------------------- #
# Generation
# ----------------------------------------------------------------------------- #
def clean_generated_answer(text):
    """Model doi khi sinh lan man sau cau tra loi; chi giu lai phan span dap an."""
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
# Eval theo tung ngon ngu
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
    parser = argparse.ArgumentParser(description="Zero-shot generative XQuAD eval cho Qwen1.5-MoE-A2.7B-SQuAD-Task-Only (LoRA)")

    # Model: base model goc + LoRA adapter checkpoint da push len HF Hub luc train SQuAD.
    parser.add_argument("--base_model_name_or_path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B",
                         help="Model goc (PHAI khop voi model dung luc train LoRA).")
    parser.add_argument("--adapter_repo_id", type=str, default="ducanhdinh/Qwen1.5-MoE-A2.7B-SQuAD-Task-Only",
                         help="HF Hub repo (hoac duong dan local) chua LoRA adapter da push luc train SQuAD.")
    parser.add_argument("--adapter_revision", type=str, default=None,
                         help="Branch/tag/commit cu the cua adapter repo, None = mac dinh (main).")
    parser.add_argument("--merge_adapter", action="store_true", default=True,
                         help="Merge LoRA vao base weights truoc khi eval (nhanh hon, mac dinh True).")
    parser.add_argument("--no_merge_adapter", dest="merge_adapter", action="store_false",
                         help="Giu adapter tach rieng (PeftModel) thay vi merge_and_unload().")
    parser.add_argument("--hf_token", type=str, default=None,
                         help="HF token (repo private); mac dinh doc tu bien moi truong.")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    # Data / eval
    parser.add_argument("--data_root", type=str, default="data/downstream/xquad",
                         help="Path toi data/downstream/xquad")
    parser.add_argument("--output_dir", type=str, default="eval/english-task-only/Qwen1.5-MoE-A2.7B/results/xquad")
    parser.add_argument("--languages", type=str, nargs="+", default=XQUAD_LANGS,
                         help="Subset ngon ngu XQuAD can eval")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                         help="Gioi han so example moi ngon ngu, de debug nhanh")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
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
    tokenizer.padding_side = "left"  # bat buoc cho batched causal-LM generation

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=args.trust_remote_code,
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

    # ---------------------------------------------------------------------------- evaluation
    results = OrderedDict()
    t0 = time.time()

    for lang in tqdm(args.languages, desc="languages"):
        data_path = os.path.join(args.data_root, lang, "validation.json")
        if not os.path.exists(data_path):
            print(f"[WARN] Bo qua '{lang}': khong tim thay {data_path}")
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

        # Luu prediction tung ngon ngu ngay (an toan neu crash o ngon ngu sau)
        with open(os.path.join(args.output_dir, f"{lang}_predictions.json"), "w", encoding="utf-8") as f:
            json.dump(lang_result["predictions"], f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------- #
    # Bao cao tong hop (overall = macro-average giua cac ngon ngu, dung quy
    # uoc chuan XTREME/XQuAD)
    # ------------------------------------------------------------------- #
    if not results:
        print("Khong co ngon ngu nao duoc eval (khong tim thay du lieu). Thoat.")
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

    summary_meta = {
        "base_model": args.base_model_name_or_path,
        "adapter_repo_id": args.adapter_repo_id,
        "adapter_revision": args.adapter_revision,
        "merged": args.merge_adapter,
        "results": summary,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_meta, f, ensure_ascii=False, indent=2)
    print(f"\nSaved per-language predictions and summary.json to: {args.output_dir}")


if __name__ == "__main__":
    main()