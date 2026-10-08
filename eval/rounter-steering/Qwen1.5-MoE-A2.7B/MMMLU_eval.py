"""
Zero-shot MMMLU (Multilingual MMLU) evaluation for Qwen1.5-MoE-A2.7B via
log-likelihood scoring.

Method
------
Standard MMLU zero-shot prompt:

    {Question}
    A. {A}
    B. {B}
    C. {C}
    D. {D}
    Answer:

We score the 4 single-letter continuations " A", " B", " C", " D" with
teacher-forced log-likelihood (no free generation) and take the argmax as
the prediction, compared against the gold `Answer` field.

Only `test.json` is used for every language folder under
`data/downstream/mmmlu/<LANG>/` (e.g. AR_XY, DE_DE, ZH_CN, ...).

OOM-safe dynamic batching
--------------------------
Every chunk of data is first attempted at the full `--batch_size`. If a
chunk raises a CUDA/CPU out-of-memory error, we:
  1. free whatever memory we can (gc.collect + torch.cuda.empty_cache) --
     this only works because we've already exited the `except` block by
     that point, so the exception's traceback (which would otherwise pin
     the failed batch's GPU tensors in memory) has been released,
  2. split the offending chunk in half and retry each half recursively
     (halving again if needed).
This shrinking is purely local to the chunk that OOM'd: it does NOT lower
the starting size for the next chunk of data. Each new chunk always starts
fresh at the full `--batch_size` again, since GPU memory is freed between
chunks. If a single example (batch size 1) still OOMs, that example is
skipped (logged and marked in the output) instead of crashing the whole
run.

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
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/MMMLU_eval.py --steering none \
        --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/mmmlu_baseline

    # (b) look at Delta and how many experts each tau selects (no evaluation).
    #     MMMLU has no English split, so give English MMLU-style data (Question/A/B/C/D) with
    #     --task_data_path (file or dir), e.g. something from data/english_task/.
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/MMMLU_eval.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --data_root data/downstream/mmmlu \
        --flores_path data/processed_alignment/flores.json \
        --delta_path eval/rounter-steering/Qwen1.5-MoE-A2.7B/delta/mmmlu.pt \
        --task_data_path <english_mmlu.json> \
        --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/mmmlu_soft --identify_only

    # (c) steered evaluation (soft, lambda=0.5)
    python eval/rounter-steering/Qwen1.5-MoE-A2.7B/MMMLU_eval.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --data_root data/downstream/mmmlu \
        --flores_path data/processed_alignment/flores.json \
        --delta_path eval/rounter-steering/Qwen1.5-MoE-A2.7B/delta/mmmlu.pt \
        --task_data_path <english_mmlu.json> \
        --steering soft --steer_lambda 0.5 --tau 0.2 --target_layers 4 19 \
        --batch_size 8 --output_dir eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/mmmlu_soft_tau0.2

Besides per-language + overall accuracy (same convention as XNLI_eval.py),
this script also dumps a per-subject breakdown (aggregated across all
languages) to `mmmlu_subject_results.csv`, since MMLU-style benchmarks are
commonly reported both by language and by subject.

Resume support
--------------
`mmmlu_results.json` in `--output_dir` doubles as a checkpoint. As soon as a
language finishes, its result (accuracy + a per-subject breakdown) is
written into that file immediately (atomically, via a temp file + rename,
so a crash mid-write never corrupts it). On the next run, any language
already present in that file is skipped automatically, and evaluation
resumes with the next language that has no result yet. Pass --overwrite to
ignore the checkpoint and re-evaluate every language from scratch.
"""

import argparse
import gc
import json
import os
import random
import sys
import tempfile

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


LETTERS = ["A", "B", "C", "D"]
CANDIDATES = [" A", " B", " C", " D"]  # index-aligned with LETTERS


# --------------------------------------------------------------------------
# OOM-safe dynamic batching helpers
# --------------------------------------------------------------------------
def is_oom_error(err: BaseException) -> bool:
    """True if `err` looks like a CUDA / CPU out-of-memory error."""
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_cls is not None and isinstance(err, oom_cls):
        return True
    if not isinstance(err, RuntimeError):
        return False
    msg = str(err).lower()
    return any(
        s in msg
        for s in (
            "out of memory",
            "cuda error: out of memory",
            "cublas_status_alloc_failed",
            "not enough memory",
        )
    )


def clear_memory():
    """Best-effort release of GPU/CPU memory before the next batch."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class DynamicBatcher:
    """Holds the batch-size bounds for a run.

    Every new outer chunk of data starts fresh at `initial_batch_size` --
    an OOM on one chunk does NOT lower the starting size for the *next*
    chunk. The halving on OOM (down to `min_batch_size`) only happens
    *within* a single chunk's divide-and-conquer retry (see
    `score_chunk_with_oom_retry`) and is discarded once that chunk is
    done; it never leaks into subsequent chunks.
    """

    def __init__(self, initial_batch_size: int, min_batch_size: int = 1):
        self.initial_batch_size = max(1, initial_batch_size)
        self.min_batch_size = max(1, min_batch_size)


def score_chunk_with_oom_retry(model, tokenizer, chunk, build_prompt_fn, device, min_batch_size=1):
    """Score `chunk` (a list of examples), recursively halving on OOM.

    Returns a list of (example, pred_index_or_None) pairs aligned with
    `chunk`. `pred_index` is None only when even a single example could not
    be scored (persistent OOM at batch size 1) -- that example is skipped
    rather than crashing the run.
    """
    if not chunk:
        return []

    prompts = [build_prompt_fn(ex) for ex in chunk]

    # NOTE on the fix below: we deliberately do NOT call clear_memory() or
    # recurse from *inside* the `except` block. In Python 3, an `except X as e`
    # clause keeps `e` (and therefore its traceback) alive for the entire
    # duration of that block. The traceback holds a reference to every stack
    # frame between where the exception was raised and where it was caught --
    # including score_candidates_batch's frame, with its still-allocated GPU
    # tensors (input_ids, attention_mask, logits, log_probs, ...). While `e`
    # is alive, gc.collect()/torch.cuda.empty_cache() cannot reclaim that
    # memory, so clear_memory() was effectively a no-op. Worse, because the
    # old code recursed *inside* the except block, every OOM'd ancestor call
    # in the recursion tree kept its own `e`/traceback (and its own failed
    # batch's tensors) alive simultaneously, all the way down -- so by the
    # time batch_size reached 1, GPU memory was still clogged with every
    # larger failed batch above it, and even a single tiny example could OOM.
    #
    # The fix: catch the exception, record that it happened, then let the
    # `except` block end (Python auto-clears `e`/the traceback at that
    # point). Only after we're back to a clean scope do we call
    # clear_memory() and recurse -- so the memory is actually freed before
    # each retry.
    oom = False
    try:
        scores = score_candidates_batch(model, tokenizer, prompts, device)
    except RuntimeError as e:
        if not is_oom_error(e):
            raise
        oom = True
    # `e` and its traceback are now out of scope and cleared.

    if not oom:
        preds = scores.argmax(axis=1)
        return list(zip(chunk, preds))

    clear_memory()

    if len(chunk) <= min_batch_size:
        q_preview = str(chunk[0].get("Question", ""))[:80]
        print(f"[OOM][WARN] batch_size=1 still OOM, skipping example: {q_preview!r}")
        return [(ex, None) for ex in chunk]

    new_size = max(min_batch_size, len(chunk) // 2)
    print(
        f"[OOM] batch_size={len(chunk)} failed -> halving to {new_size} and retrying "
        f"(next outer chunk still starts fresh at the full --batch_size)"
    )
    mid = len(chunk) // 2
    left = score_chunk_with_oom_retry(model, tokenizer, chunk[:mid], build_prompt_fn, device, min_batch_size)
    clear_memory()
    right = score_chunk_with_oom_retry(model, tokenizer, chunk[mid:], build_prompt_fn, device, min_batch_size)
    return left + right


def build_prompt(ex: dict) -> str:
    return (
        f"{ex['Question']}\n"
        f"A. {ex['A']}\n"
        f"B. {ex['B']}\n"
        f"C. {ex['C']}\n"
        f"D. {ex['D']}\n"
        f"Answer:"
    )


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
    """Length-normalized log-likelihood of each candidate for a batch of prompts.
    Returns numpy array of shape (len(prompts), len(CANDIDATES))."""
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
    log_probs = F.log_softmax(outputs.logits, dim=-1)

    seq_lens = attention_mask.sum(dim=1).tolist()
    n = input_ids.shape[0]
    scores = torch.empty(n, dtype=torch.float32)

    for i in range(n):
        ctx_len = context_lens[i]
        real_len = int(seq_lens[i])
        if real_len <= ctx_len:
            scores[i] = float("-inf")
            continue
        token_ids = input_ids[i, ctx_len:real_len]
        pred_log_probs = log_probs[i, ctx_len - 1 : real_len - 1, :]
        gathered = pred_log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)
        scores[i] = gathered.mean().item()

    return scores.view(b, num_cand).numpy()


# --------------------------------------------------------------------------
# Resume / checkpoint helpers
# --------------------------------------------------------------------------
def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Write `data` to `path` atomically: write to a temp file in the same
    directory, then os.replace() it into place. This means a crash or kill
    mid-write can never leave a truncated/corrupted result file behind --
    important here since this file also serves as the resume checkpoint."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_mmmlu_", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_write_json(path: str, obj) -> None:
    _atomic_write_bytes(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def atomic_write_csv(path: str, df: pd.DataFrame) -> None:
    _atomic_write_bytes(path, df.to_csv(index=False).encode("utf-8"))


def compute_subject_breakdown(records):
    """Aggregate a language's records into {subject: {"correct", "total"}}.

    `total` includes skipped (persistent-OOM) examples, counted as
    incorrect -- this matches the original semantics of the subject CSV
    (mean of the `correct` boolean across every record, skipped or not).
    """
    breakdown = {}
    for r in records:
        subj = r.get("subject") or "unknown"
        entry = breakdown.setdefault(subj, {"correct": 0, "total": 0})
        entry["total"] += 1
        if r.get("correct"):
            entry["correct"] += 1
    return breakdown


def merge_subject_breakdown(subject_totals: dict, breakdown: dict) -> None:
    """In-place merge of one language's subject breakdown into the running,
    all-languages totals used to build mmmlu_subject_results.csv."""
    for subj, stats in breakdown.items():
        entry = subject_totals.setdefault(subj, {"correct": 0, "total": 0})
        entry["correct"] += stats["correct"]
        entry["total"] += stats["total"]


def load_existing_results(json_path: str):
    """Load a previous run's mmmlu_results.json, if any, so we can resume.

    Returns (results, subject_totals, langs_missing_breakdown):
      - results: {lang: {...}} for every language already fully evaluated
        in a prior run -- these will be SKIPPED this run.
      - subject_totals: {subject: {"correct", "total"}} merged across all
        of those already-done languages.
      - langs_missing_breakdown: languages found in the file that predate
        the "subject_breakdown" field (saved by an older version of this
        script). They still count as done and are skipped, but can't
        contribute to mmmlu_subject_results.csv since their raw per-example
        records are gone.
    """
    if not os.path.exists(json_path):
        return {}, {}, []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Could not read existing results file {json_path} ({e}); starting fresh.")
        return {}, {}, []

    results = {}
    subject_totals = {}
    langs_missing_breakdown = []
    for lang, stats in prev.get("per_language", {}).items():
        results[lang] = dict(stats)
        breakdown = stats.get("subject_breakdown")
        if breakdown:
            merge_subject_breakdown(subject_totals, breakdown)
        else:
            langs_missing_breakdown.append(lang)
    return results, subject_totals, langs_missing_breakdown


def write_all_outputs(output_dir, model_name, results, subject_totals, total_skipped, steering_cfg=None):
    """(Re)write mmmlu_results.csv, mmmlu_subject_results.csv, and
    mmmlu_results.json from current in-memory state. Called after every
    single language finishes (not just once at the end) so a crash never
    loses more than the language currently in progress, and so the next run
    can resume by reading mmmlu_results.json."""
    overall_correct = sum(r["accuracy"] * r["n_examples"] for r in results.values())
    overall_total = sum(r["n_examples"] for r in results.values())
    overall_micro_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    macro_acc = sum(r["accuracy"] for r in results.values()) / len(results) if results else 0.0

    lang_rows = [
        {"language": lang, "accuracy": r["accuracy"], "n_examples": r["n_examples"], "n_skipped": r["n_skipped"]}
        for lang, r in results.items()
    ]
    lang_df = pd.DataFrame(lang_rows, columns=["language", "accuracy", "n_examples", "n_skipped"])
    if not lang_df.empty:
        lang_df = lang_df.sort_values("language")
    csv_path = os.path.join(output_dir, "mmmlu_results.csv")
    atomic_write_csv(csv_path, lang_df)

    subj_csv_path = os.path.join(output_dir, "mmmlu_subject_results.csv")
    if subject_totals:
        subj_rows = [
            {
                "subject": subj,
                "accuracy": (stats["correct"] / stats["total"]) if stats["total"] else 0.0,
                "n_examples": stats["total"],
            }
            for subj, stats in subject_totals.items()
        ]
        subj_df = pd.DataFrame(subj_rows).sort_values("subject")
        atomic_write_csv(subj_csv_path, subj_df)

    summary = {
        "model": model_name,
        "per_language": results,
        "overall_micro_accuracy": overall_micro_acc,
        "macro_average_accuracy": macro_acc,
        "total_skipped": total_skipped,
        "steering_config": steering_cfg if steering_cfg is not None else {"steering": "none"},
    }
    json_path = os.path.join(output_dir, "mmmlu_results.json")
    atomic_write_json(json_path, summary)

    return overall_micro_acc, macro_acc, csv_path, subj_csv_path, json_path


def evaluate_language(model, tokenizer, lang, data_root, device, batch_size, max_examples=None, min_batch_size=1):
    lang_dir = os.path.join(data_root, lang)
    data = load_test_data(lang_dir)
    if max_examples is not None:
        data = data[:max_examples]

    correct, total, skipped = 0, 0, 0
    records = []
    batcher = DynamicBatcher(initial_batch_size=batch_size, min_batch_size=min_batch_size)

    pbar = tqdm(total=len(data), desc=f"MMMLU[{lang}]")
    idx = 0
    while idx < len(data):
        # Always start each new chunk at the full requested batch size.
        # An OOM on a previous chunk only shrinks *that* chunk's own
        # divide-and-conquer retry internally (see score_chunk_with_oom_retry);
        # it does not carry over and shrink subsequent chunks.
        cur_bs = batcher.initial_batch_size
        chunk = data[idx : idx + cur_bs]

        pair_results = score_chunk_with_oom_retry(
            model, tokenizer, chunk, build_prompt, device, min_batch_size
        )

        for ex, pred in pair_results:
            gold_letter = str(ex["Answer"]).strip()
            total += 1
            if pred is None:
                skipped += 1
                records.append(
                    {
                        "subject": ex.get("Subject", ""),
                        "question": ex["Question"],
                        "gold": gold_letter,
                        "pred": None,
                        "correct": False,
                        "skipped": True,
                    }
                )
                continue
            pred_letter = LETTERS[int(pred)]
            is_correct = pred_letter == gold_letter
            correct += is_correct
            records.append(
                {
                    "subject": ex.get("Subject", ""),
                    "question": ex["Question"],
                    "gold": gold_letter,
                    "pred": pred_letter,
                    "correct": is_correct,
                    "skipped": False,
                }
            )

        idx += len(chunk)
        pbar.update(len(chunk))
        clear_memory()
    pbar.close()

    if skipped:
        print(f"[{lang}] WARNING: {skipped} example(s) skipped due to persistent OOM at batch_size=1.")

    acc = correct / total if total > 0 else 0.0
    subject_breakdown = compute_subject_breakdown(records)
    return acc, total, skipped, records, subject_breakdown


def _mmlu_fields(ex):
    """Normalize an MMLU-style record to {Question, A, B, C, D}."""
    if all(k in ex for k in ("Question", "A", "B", "C", "D")):
        return ex
    low = {k.lower(): v for k, v in ex.items()}
    if "question" in low and all(k in low for k in "abcd"):
        return {"Question": low["question"], "A": low["a"], "B": low["b"], "C": low["c"], "D": low["d"]}
    ch = low.get("choices")
    if "question" in low and isinstance(ch, list) and len(ch) >= 4:
        return {"Question": low["question"], "A": ch[0], "B": ch[1], "C": ch[2], "D": ch[3]}
    raise KeyError(f"Expected Question/A/B/C/D (or question + choices) fields, got {list(ex)}")


def get_task_texts(args):
    """English in-task prompts (same MMLU prompt as the evaluation) used to identify task experts."""
    if args.task_data_path:
        data = read_json_records(args.task_data_path)
    else:
        cands = [args.id_lang] if args.id_lang else ["EN_US", "EN", "en", "EN_XX", "EN_GB", "english"]
        folder = next((c for c in cands if os.path.isdir(os.path.join(args.data_root, c))), None)
        if folder is None:
            raise FileNotFoundError(
                f"No English split found under {args.data_root} (tried {cands}). MMMLU has no English folder: "
                "pass --task_data_path <English MMLU-style json with Question/A/B/C/D> (or --id_lang <folder>)."
            )
        data = load_test_data(os.path.join(args.data_root, folder))
    return subsample([build_prompt(_mmlu_fields(ex)) for ex in data], args.id_num_samples, args.seed)


def saved_steering_signature(json_path):
    """Steering signature stored in a previous results file (missing -> baseline)."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f).get("steering_config", {"steering": "none"})
    except (OSError, json.JSONDecodeError):
        return None
    return {k: v for k, v in cfg.items() if k in ("steering", "tau", "target_layers", "lambda")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--data_root", default="data/downstream/mmmlu")
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size each chunk starts at; halves locally (per-chunk only) on OOM.")
    parser.add_argument("--min_batch_size", type=int, default=1, help="Never split batches smaller than this before skipping an example.")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--output_dir", default="eval/rounter-steering/Qwen1.5-MoE-A2.7B/results/mmmlu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing mmmlu_results.json and re-evaluate every language from scratch "
             "(by default, languages already present in that file are skipped and treated as done).",
    )
    add_steering_args(parser)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    json_path = os.path.join(args.output_dir, "mmmlu_results.json")

    # ------------------------------------------------------------------
    # Resume support: mmmlu_results.json doubles as the checkpoint. Any
    # language already recorded there from a previous run is skipped below,
    # unless --overwrite is passed.
    # ------------------------------------------------------------------
    if args.overwrite:
        results, subject_totals, langs_missing_breakdown = {}, {}, []
    else:
        results, subject_totals, langs_missing_breakdown = load_existing_results(json_path)

    if results and not args.overwrite:
        saved_sig = saved_steering_signature(json_path)
        if saved_sig is not None and saved_sig != steering_signature(args):
            raise SystemExit(
                f"{json_path} was produced with steering config {saved_sig}, but this run uses "
                f"{steering_signature(args)}. Use a different --output_dir (or pass --overwrite) so "
                "results from different configurations are never mixed."
            )

    if results:
        print(f"Found existing results for {len(results)} language(s) in {json_path}.")
    if langs_missing_breakdown:
        print(
            f"[WARN] {len(langs_missing_breakdown)} of those were saved by an older version of "
            f"this script with no stored subject breakdown ({langs_missing_breakdown}); they'll "
            f"still be skipped, but won't contribute to mmmlu_subject_results.csv. Pass "
            f"--overwrite if you need to regenerate subject-level stats for them."
        )

    total_skipped = sum(r.get("n_skipped", 0) for r in results.values())

    languages = args.languages or sorted(
        d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))
    )

    already_done = [lang for lang in languages if lang in results]
    to_run = [lang for lang in languages if lang not in results]

    if already_done:
        print(f"Skipping {len(already_done)} already-completed language(s): {already_done}")
    if to_run:
        print(f"Languages to evaluate ({len(to_run)}): {to_run}")
    else:
        print("Nothing left to evaluate -- every requested language already has a result.")

    controller, steering_cfg = None, steering_signature(args)

    # Only load the (potentially huge) model if there's actually work to do.
    if to_run:
        print(f"Loading model: {args.model_name_or_path} (device={device}, dtype={args.dtype})")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

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
            args, model, tokenizer, device, lambda: get_task_texts(args), tag="mmmlu"
        )

    for lang in to_run:
        acc, total, skipped, records, subject_breakdown = evaluate_language(
            model, tokenizer, lang, args.data_root, device, args.batch_size, args.max_examples, args.min_batch_size
        )
        results[lang] = {
            "accuracy": acc,
            "n_examples": total,
            "n_skipped": skipped,
            "subject_breakdown": subject_breakdown,
        }
        total_skipped += skipped
        print(f"[{lang}] accuracy = {acc:.4f}  ({total} examples, {skipped} skipped)")

        merge_subject_breakdown(subject_totals, subject_breakdown)

        if args.save_predictions:
            for r in records:
                r["language"] = lang
            atomic_write_csv(
                os.path.join(args.output_dir, f"mmmlu_predictions_{lang}.csv"), pd.DataFrame(records)
            )

        # Save/checkpoint immediately -- a crash on the *next* language will
        # never lose this one, and a fresh run can resume right after it.
        _, _, _, _, checkpoint_path = write_all_outputs(
            args.output_dir, args.model_name_or_path, results, subject_totals, total_skipped, steering_cfg
        )
        print(f"  -> checkpointed to {checkpoint_path}")

        # free memory before moving on to the next language
        clear_memory()

    overall_micro_acc, macro_acc, csv_path, subj_csv_path, json_path = write_all_outputs(
        args.output_dir, args.model_name_or_path, results, subject_totals, total_skipped, steering_cfg
    )

    print("=" * 60)
    print(f"Overall (micro, weighted by #examples) accuracy: {overall_micro_acc:.4f}")
    print(f"Macro-average (mean over languages) accuracy:    {macro_acc:.4f}")
    if total_skipped:
        print(f"Total skipped examples (persistent OOM): {total_skipped}")

    print(f"\nSaved: {csv_path}")
    if subject_totals:
        print(f"Saved: {subj_csv_path}")
    print(f"Saved: {json_path}")
    report_steering_usage(controller)


if __name__ == "__main__":
    main()