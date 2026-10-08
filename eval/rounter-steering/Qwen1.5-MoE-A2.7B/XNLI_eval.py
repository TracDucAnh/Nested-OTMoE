"""
Zero-shot XNLI evaluation for Qwen1.5-MoE-A2.7B via log-likelihood scoring.

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
prediction. This is the same style of prompt used in the GPT-3 paper and in
lm-evaluation-harness for ANLI/RTE/XNLI-like tasks, and avoids the format /
parsing errors that come with free-form generation, especially in
low-resource languages.

Only `test.json` is used for every language folder under
`data/downstream/xnli/<lang>/`.

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
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/XNLI_eval.py --steering none \
        --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/xnli_baseline

    # (b) look at Delta and how many experts each tau selects (no evaluation)
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/XNLI_eval.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --data_root data/downstream/xnli \
        --flores_path data/processed_alignment/flores.json \
        --delta_path eval/rounter-steering/Qwen1.5-MoE-A2.7B/delta/xnli.pt \
        --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/xnli_soft --identify_only

    # (c) steered evaluation (soft, lambda=0.5)
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/XNLI_eval.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --data_root data/downstream/xnli \
        --flores_path data/processed_alignment/flores.json \
        --delta_path eval/rounter-steering/Qwen1.5-MoE-A2.7B/delta/xnli.pt \
        --steering soft --steer_lambda 0.5 --tau 0.2 --target_layers 4 19 \
        --batch_size 8 --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/xnli_soft_tau0.2

Notes
-----
- Requires a GPU with enough VRAM to hold the model (Qwen1.5-MoE-A2.7B has
  ~14.3B total params / 2.7B activated, so budget ~28GB in bf16). Falls back
  to CPU automatically but will be very slow.
- `--languages` lets you restrict to a subset, e.g. `--languages en vi zh`.
- `--max_examples` is handy to smoke-test the pipeline on a few examples
  before launching the full run.
"""

import argparse
import json
import os
import random
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


# XNLI label id -> candidate continuation text (leading space matters for BPE tokenizers)
LABEL_NAMES = ["entailment", "neutral", "contradiction"]
CANDIDATES = [" True", " Neither", " False"]  # index-aligned with LABEL_NAMES / label ids 0,1,2


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


def get_task_texts(args):
    """English in-task prompts (same prompt format as the evaluation) used to identify task experts."""
    if args.task_data_path:
        data = read_json_records(args.task_data_path)
    else:
        data = load_test_data(os.path.join(args.data_root, args.id_lang or "en"))
    texts = []
    for ex in data:
        p = ex.get("premise", ex.get("sentence1"))
        h = ex.get("hypothesis", ex.get("sentence2"))
        if p is None or h is None:
            raise KeyError(f"Expected premise/hypothesis (or sentence1/sentence2) fields, got {list(ex)}")
        texts.append(build_prompt(p, h))
    return subsample(texts, args.id_num_samples, args.seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--data_root", default="data/downstream/xnli")
    parser.add_argument("--languages", nargs="+", default=None, help="subset of languages, default = all found")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=None, help="debug: limit examples per language")
    parser.add_argument("--output_dir", default="eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/xnli")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_predictions", action="store_true", help="dump per-example predictions to CSV")
    add_steering_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    print(f"Loading model: {args.model_name_or_path} (device={device}, dtype={args.dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # required by the log-likelihood slicing logic above

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    if device == "cpu":
        model.to(device)

    controller, steering_cfg = setup_router_steering(
        args, model, tokenizer, device, lambda: get_task_texts(args), tag="xnli"
    )

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
        "model": args.model_name_or_path,
        "per_language": results,
        "overall_micro_accuracy": overall_micro_acc,
        "macro_average_accuracy": macro_acc,
        "steering_config": steering_cfg,
    }
    json_path = os.path.join(args.output_dir, "xnli_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    report_steering_usage(controller)


if __name__ == "__main__":
    main()