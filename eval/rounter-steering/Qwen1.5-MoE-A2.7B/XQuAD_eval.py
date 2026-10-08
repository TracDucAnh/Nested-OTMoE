#!/usr/bin/env python3
"""
Zero-shot generative evaluation of Qwen1.5-MoE-A2.7B on XQuAD.

For every language available under `data/downstream/xquad/<lang>/validation.json`,
the model is prompted zero-shot (no in-context examples) to *generate* an answer
given the (context, question) pair. Generated answers are scored against the gold
answers with standard SQuAD-style Exact Match (EM) and token-level F1, using the
multilingual normalization scheme from the official XQuAD/MLQA evaluation scripts
(character-level matching for languages without whitespace word boundaries:
Chinese and Thai; whitespace-token matching otherwise).

Router steering (inference-time, no training)
---------------------------------------------
Implements the routing intervention of Bandarkar et al., "Multilingual Routing in
Mixture-of-Experts" (ICLR 2026), Sec. 5.2-5.3, on top of the original evaluation below:

  1. Expert identification (Eq. 3). For every MoE layer, the relative activation
     frequency a_i/L_i (fraction of tokens of a sequence whose top-K contains the expert)
     is averaged over English in-task prompts and over English FLORES (generic baseline);
     Delta = freq(task) - freq(FLORES). Expert k of layer l is a task expert iff
     Delta[l,k] > tau. Delta is cached (--delta_path) so tau / lambda sweeps are cheap.
  2. Intervention (Eq. 4/5). Only in the target middle layers (--target_layers, 1-indexed
     inclusive) and BEFORE the router softmax, on every token and every language:
         soft: z'_k = z_k + lambda * std(z)        (--steering soft --steer_lambda 0.5)
         hard: z'_k = max(z) + eps, eps~N(0,1e-6)  (--steering hard)
     for every selected expert k. `--steering none` reproduces the original model.

Usage (run from the repo root)
------------------------------
    # (a) baseline = original model
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/XQuAD_eval.py --steering none \
        --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/xquad_baseline

    # (b) look at Delta and how many experts each tau selects (no evaluation)
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/XQuAD_eval.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --data_root data/downstream/xquad \
        --flores_path data/processed_alignment/flores.json \
        --delta_path eval/rounter-steering/Qwen1.5-MoE-A2.7B/delta/xquad.pt \
        --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/xquad_soft --identify_only

    # (c) steered evaluation (soft, lambda=0.5)
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/XQuAD_eval.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --data_root data/downstream/xquad \
        --flores_path data/processed_alignment/flores.json \
        --delta_path eval/rounter-steering/Qwen1.5-MoE-A2.7B/delta/xquad.pt \
        --steering soft --steer_lambda 0.5 --tau 0.2 --target_layers 4 19 \
        --batch_size 8 --max_new_tokens 32 --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/xquad_soft_tau0.2

Only a subset of languages:   add  --languages en vi zh ar
Quick debug run:              add  --limit 20
"""

import argparse
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter, OrderedDict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# =============================================================================
# >>> ROUTER STEERING <<<  (identical block in XNLI_eval.py / MMMLU_eval.py / XQuAD_eval.py)
#
# Implementation of the inference-time routing intervention from
#   Bandarkar et al., "Multilingual Routing in Mixture-of-Experts", ICLR 2026
#   (Sec. 5.2 "Expert identification" + Sec. 5.3 "Routing interventions").
#
# 1. Expert identification (Eq. 3)
#    For every MoE layer we compute the *relative activation frequency*
#    a_i / L_i in [0,1]^E  (a_i[e] = #tokens of sequence i whose top-K contains
#    expert e, L_i = #tokens of sequence i), average it over the sequences of
#      (1) English in-task data   and   (2) the generic English baseline (FLORES),
#    and take  Delta = mean(1) - mean(2)  in [-1,1]^E.
#    Expert k of layer l is a *task expert* (member of A+) iff Delta[l,k] > tau.
#
# 2. Intervention (Eq. 4 / 5), applied only in the target (middle) layers,
#    to the router logits *before* the softmax, on every token / every language:
#      soft :  z'_k = z_k + lambda * std(z)          (std over all E logits)
#      hard :  z'_k = max(z) + eps,  eps ~ N(0, 1e-6)
#    for every k in A+ of that layer.
# =============================================================================
class MoERouterController:
    """Forward hooks on every MoE router (`mlp.gate`, an nn.Linear) of a Qwen2-MoE model.

    mode == "off"    : hooks are no-ops (original model behaviour)
    mode == "record" : accumulate per-sequence expert activation frequencies (identification)
    mode == "steer"  : modify router logits of the selected experts (intervention)
    """

    def __init__(self, model):
        backbone = model.model
        self.gates = {}
        for li, layer in enumerate(backbone.layers):
            gate = getattr(getattr(layer, "mlp", None), "gate", None)
            if gate is None:
                continue  # dense MLP layer
            if not isinstance(gate, torch.nn.Linear):
                raise RuntimeError(
                    f"Layer {li}: router `mlp.gate` is {type(gate).__name__}, expected nn.Linear. "
                    "This steering code hooks the router logits of transformers 4.x Qwen2MoE "
                    "(e.g. `pip install 'transformers>=4.40,<5'`)."
                )
            self.gates[li] = gate
        if not self.gates:
            raise RuntimeError("No MoE router (`layer.mlp.gate`) found in this model.")

        self.num_layers = len(backbone.layers)
        first = min(self.gates)
        self.num_experts = self.gates[first].out_features
        self.top_k = int(getattr(backbone.layers[first].mlp, "top_k", None) or model.config.num_experts_per_tok)

        self.mode = "off"
        self.method = "soft"
        self.lam = 0.5
        self.selected = {}  # layer idx (0-based) -> LongTensor of expert ids (A+)
        self.steer_calls = 0  # sanity counter: number of steered router calls
        self._mask = None
        self._freq_sum = None
        self._handles = [g.register_forward_hook(self._make_hook(li)) for li, g in self.gates.items()]

    # ---- hooks ---------------------------------------------------------
    def _make_hook(self, li):
        def hook(module, inputs, output):
            if self.mode == "record":
                self._record(li, output)
            elif self.mode == "steer" and li in self.selected:
                return self._steer(li, output)  # returned tensor replaces the router logits
            return None

        return hook

    @torch.no_grad()
    def _record(self, li, logits):
        B, T = self._mask.shape
        k = self.top_k
        if logits.shape[0] != B * T:
            raise RuntimeError(f"Unexpected router logits shape {tuple(logits.shape)} for batch ({B}, {T}).")
        mask = self._mask.to(logits.device)
        idx = logits.float().topk(k, dim=-1).indices.reshape(B, T * k)  # top-K experts per token
        w = mask.unsqueeze(-1).expand(B, T, k).reshape(B, T * k).float()  # ignore padding tokens
        counts = torch.zeros(B, self.num_experts, device=logits.device).scatter_add_(1, idx, w)  # a_i
        freq = counts / mask.sum(dim=1, keepdim=True).clamp(min=1)  # a_i / L_i
        self._freq_sum[li] += freq.sum(dim=0).double().cpu()

    @torch.no_grad()
    def _steer(self, li, logits):
        idx = self.selected[li]
        if idx.device != logits.device:
            idx = idx.to(logits.device)
            self.selected[li] = idx
        z = logits.to(torch.float32).clone()  # float32 so that the 1e-6 hard-mode epsilon survives
        if self.method == "soft":
            s = z.std(dim=-1, keepdim=True, unbiased=False)  # s(z): std over all E logits of each token
            z[:, idx] = z[:, idx] + self.lam * s
        elif self.method == "hard":
            mx = z.max(dim=-1, keepdim=True).values
            eps = torch.randn(z.shape[0], idx.numel(), device=z.device) * 1e-6
            z[:, idx] = mx + eps
        else:
            raise ValueError(self.method)
        self.steer_calls += 1
        return z

    # ---- identification -----------------------------------------------
    @torch.no_grad()
    def activation_frequency(self, model, tokenizer, texts, batch_size, max_length, device, desc):
        """Mean over sequences of a_i / L_i  ->  float tensor (num_layers, num_experts)."""
        prev_mode, prev_side = self.mode, tokenizer.padding_side
        self.mode = "record"
        self._freq_sum = torch.zeros(self.num_layers, self.num_experts, dtype=torch.float64)
        tokenizer.padding_side = "right"
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-sorted -> little padding
        try:
            for s in tqdm(range(0, len(order), batch_size), desc=desc):
                batch = [texts[i] for i in order[s : s + batch_size]]
                enc = tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=max_length, add_special_tokens=False,
                )
                self._mask = enc["attention_mask"].bool()
                # backbone only (skips the 150k-vocab LM head); all MoE routers still run
                model.model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                    use_cache=False,
                )
        finally:
            self.mode, tokenizer.padding_side, self._mask = prev_mode, prev_side, None
        return (self._freq_sum / max(1, len(texts))).float()

    def configure_steering(self, selected, method, lam):
        self.selected = {li: torch.tensor(ids, dtype=torch.long) for li, ids in selected.items()}
        self.method, self.lam = method, float(lam)


def add_steering_args(parser):
    g = parser.add_argument_group("router steering (Bandarkar et al., ICLR 2026)")
    g.add_argument("--steering", default="soft", choices=["none", "soft", "hard"],
                   help="none = original model (baseline); soft: z_k += lambda*std(z); hard: z_k = max(z)+eps")
    g.add_argument("--steer_lambda", type=float, default=0.5, help="lambda of the soft intervention (paper: |lambda|=0.5)")
    g.add_argument("--tau", type=float, default=0.2, help="expert-selection threshold on Delta (tune per model!)")
    g.add_argument("--target_layers", type=int, nargs=2, default=[4, 19], metavar=("START", "END"),
                   help="1-indexed INCLUSIVE range of layers to intervene in (middle layers). Qwen1.5-MoE has 24 layers.")
    g.add_argument("--flores_path", default="data/processed_alignment/flores.json",
                   help="FLORES json; its English side is the generic baseline for Delta")
    g.add_argument("--flores_lang", default="eng_Latn")
    g.add_argument("--task_data_path", default=None,
                   help="optional json (file or dir) with English in-task data used to identify task experts. "
                        "Default: the English split of the downstream eval data (inputs only, no labels).")
    g.add_argument("--id_lang", default=None, help="folder name of the English split under --data_root (default per task)")
    g.add_argument("--id_num_samples", type=int, default=1000, help="#sequences per dataset for identification")
    g.add_argument("--id_batch_size", type=int, default=8)
    g.add_argument("--id_max_length", type=int, default=1024)
    g.add_argument("--delta_path", default=None, help="cache file for Delta (default: <output_dir>/delta_<task>.pt)")
    g.add_argument("--recompute_delta", action="store_true", help="ignore cached Delta")
    g.add_argument("--identify_only", action="store_true",
                   help="only compute/cache Delta and print how many experts each tau selects, then exit")
    g.add_argument("--seed", type=int, default=42)


def steering_signature(args):
    """Config that must match when resuming from an existing results file."""
    if args.steering == "none":
        return {"steering": "none"}
    sig = {"steering": args.steering, "tau": args.tau, "target_layers": list(args.target_layers)}
    if args.steering == "soft":
        sig["lambda"] = args.steer_lambda
    return sig


def resolve_json_path(path):
    if os.path.isdir(path):
        for name in ("train.json", "test.json", "validation.json", "dev.json"):
            cand = os.path.join(path, name)
            if os.path.exists(cand):
                return cand
        raise FileNotFoundError(f"No train/test/validation/dev.json found in {path}")
    return path


def read_json_records(path):
    with open(resolve_json_path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "examples", "rows", "train", "test", "validation"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"Unrecognized json structure at {path}")
    return data


def subsample(items, n, seed):
    items = list(items)
    if n is not None and len(items) > n:
        items = random.Random(seed).sample(items, n)
    return items


def load_flores_texts(path, lang, n, seed):
    records = read_json_records(path)
    if records and lang not in records[0]:
        cands = [k for k in records[0] if k.startswith("eng")]
        if not cands:
            raise KeyError(f"'{lang}' not in {path}; available keys: {list(records[0])[:8]}...")
        lang = cands[0]
        print(f"[steering] FLORES key '{lang}' used for the English baseline")
    texts = [r[lang] for r in records if r.get(lang)]
    return subsample(texts, n, seed)


def setup_router_steering(args, model, tokenizer, device, task_texts_fn, tag):
    """Attach router hooks, identify task experts (cached), and switch steering on.

    Returns (controller or None, config dict to store in the results json)."""
    if args.steering == "none":
        print("[steering] OFF -> original (baseline) model")
        return None, {"steering": "none"}

    ctrl = MoERouterController(model)
    num_layers, E = ctrl.num_layers, ctrl.num_experts
    start, end = args.target_layers
    if not (1 <= start <= end <= num_layers):
        raise ValueError(f"--target_layers {start} {end} invalid for a {num_layers}-layer model")
    target = [l for l in range(start - 1, end) if l in ctrl.gates]
    print(f"[steering] {args.steering} | layers {start}-{end} (1-indexed, {len(target)} MoE layers) | "
          f"E={E}, top_k={ctrl.top_k} | tau={args.tau}"
          + (f" | lambda={args.steer_lambda}" if args.steering == "soft" else ""))

    # ---- Delta = freq(English in-task data) - freq(English FLORES) -----------
    delta_path = args.delta_path or os.path.join(args.output_dir, f"delta_{tag}.pt")
    meta = {"model": args.model_name_or_path, "n": args.id_num_samples, "seed": args.seed,
            "flores": os.path.basename(args.flores_path), "task_data": args.task_data_path or f"default-{args.id_lang or 'en'}"}
    delta = None
    if os.path.exists(delta_path) and not args.recompute_delta:
        blob = torch.load(delta_path, map_location="cpu")
        if blob.get("meta") == meta and tuple(blob["delta"].shape) == (num_layers, E):
            delta = blob["delta"]
            print(f"[steering] loaded cached Delta from {delta_path}")
        else:
            print("[steering] cached Delta does not match current settings -> recomputing")
    if delta is None:
        task_texts = task_texts_fn()
        base_texts = load_flores_texts(args.flores_path, args.flores_lang, args.id_num_samples, args.seed)
        print(f"[steering] identification: {len(task_texts)} English task prompts vs {len(base_texts)} FLORES-EN sentences")
        f_task = ctrl.activation_frequency(model, tokenizer, task_texts, args.id_batch_size, args.id_max_length, device, "identify[task]")
        f_base = ctrl.activation_frequency(model, tokenizer, base_texts, args.id_batch_size, args.id_max_length, device, "identify[flores-en]")
        delta = f_task - f_base  # Eq. (3)
        os.makedirs(os.path.dirname(os.path.abspath(delta_path)), exist_ok=True)
        torch.save({"delta": delta, "freq_task": f_task, "freq_base": f_base, "meta": meta}, delta_path)
        print(f"[steering] saved Delta to {delta_path}")

    # ---- report: how many experts each tau would select -----------------------
    tgt_delta = delta[target]
    print("[steering] Delta stats in target layers: max={:.3f}, 99th pct={:.3f}".format(
        tgt_delta.max().item(), torch.quantile(tgt_delta.flatten(), 0.99).item()))
    for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        print(f"[steering]   tau={t:<4}: {int((tgt_delta > t).sum())} experts selected in target layers")
    if args.identify_only:
        print("[steering] --identify_only: exiting.")
        sys.exit(0)

    # ---- A+ : experts with Delta > tau inside the target layers ---------------
    selected = {l: (delta[l] > args.tau).nonzero(as_tuple=True)[0].tolist() for l in target}
    selected = {l: ids for l, ids in selected.items() if ids}
    total = sum(len(v) for v in selected.values())
    if total == 0:
        raise RuntimeError(f"No expert has Delta > tau={args.tau} in layers {start}-{end} "
                           f"(max Delta = {tgt_delta.max().item():.3f}). Lower --tau.")
    print(f"[steering] selected {total} experts: " + ", ".join(f"L{l + 1}:{len(v)}" for l, v in sorted(selected.items())))
    if args.steering == "hard" and any(len(v) > ctrl.top_k for v in selected.values()):
        print(f"[steering][WARN] hard mode forces > top_k={ctrl.top_k} experts in some layer; the paper reports this "
              "derails the model -> use a larger --tau.")

    ctrl.configure_steering(selected, args.steering, args.steer_lambda)
    ctrl.mode = "steer"
    cfg = dict(steering_signature(args))
    cfg["num_selected_experts"] = total
    cfg["selected_experts_per_layer_1idx"] = {str(l + 1): v for l, v in sorted(selected.items())}
    return ctrl, cfg


def report_steering_usage(ctrl):
    if ctrl is not None:
        print(f"[steering] steered router calls during evaluation: {ctrl.steer_calls} "
              "(0 would mean the hooks never fired)")
# >>> END ROUTER STEERING <<<



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
def get_task_texts(args):
    """English in-task prompts (same prompt as the evaluation) used to identify task experts."""
    if args.task_data_path:
        path = resolve_json_path(args.task_data_path)
    else:
        path = os.path.join(args.data_root, args.id_lang or "en", "validation.json")
    records = load_xquad_split(path)
    texts = [PROMPT_TEMPLATE.format(context=r["context"], question=r["question"]) for r in records]
    return subsample(texts, args.id_num_samples, args.seed)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot generative XQuAD eval for Qwen1.5-MoE-A2.7B")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--data_root", type=str, default="data/downstream/xquad",
                         help="Path to data/downstream/xquad (relative to the repo root)")
    parser.add_argument("--output_dir", type=str, default="eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/xquad")
    parser.add_argument("--languages", type=str, nargs="+", default=XQUAD_LANGS,
                         help="Subset of XQuAD languages to evaluate")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of examples per language, for quick debugging")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    add_steering_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    print(f"Loading tokenizer & model: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched causal-LM generation

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=args.trust_remote_code,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    controller, steering_cfg = setup_router_steering(
        args, model, tokenizer, device, lambda: get_task_texts(args), tag="xquad"
    )

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
    report_steering_usage(controller)
    summary["steering_config"] = steering_cfg

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved per-language predictions and summary.json to: {args.output_dir}")


if __name__ == "__main__":
    main()