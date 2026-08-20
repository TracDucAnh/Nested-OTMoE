"""
MidAlign baseline cho mo hinh Mixture-of-Experts ATH-MaaS/Marco-Nano-Instruct.

Adapt tu "Middle-Layer Representation Alignment for Cross-Lingual Transfer in
Fine-Tuned LLMs" (Liu & Niehues, 2025) sang backbone MoE, dung Alternate
Training (Figure 2 trong paper): moi step CHI toi uu MOT trong hai objective,
xen ke theo global_step (chan = task step, le = align step).

  - Task step:  L_task = L_LM (+ lb_loss_coef * L_LB neu model la MoE)
        L_LM: causal LM loss (cross-entropy chuan, shift-by-1) tinh tren cau
              TARGET LANGUAGE (phia "other", khong phai tieng Anh).
        L_LB: load-balancing loss chuan cua MoE (Switch/Mixtral style), tinh
              tren (cac) router nam trong layer duoc gan LoRA. Day la phan
              phu tro danh rieng cho backbone MoE, khong thuoc dinh nghia
              goc cua MidAlign nhung duoc giu lai vi model la MoE; co the tat
              bang --lb_loss_coef 0 neu chi muon L_task = L_LM thuan tuy.
  - Align step: L_align = contrastive loss (in-batch negatives, symmetric
        InfoNCE, tuong duong Eq.1 trong paper) giua mean-pooled hidden state
        cua cau tieng Anh va cau target, trich xuat tai DUNG layer 16
        (--align_layer, mac dinh 16). Quy uoc chi so: hidden_states[16] tuc
        la output SAU decoder block co index 0-based = 15 (giong truc "Layer
        ID" trong Figure 1/4 cua paper, trong do Layer ID 0 = embedding).

Cac dieu kien giu nguyen theo yeu cau:
  1. Alternate training giua task loss va contrastive loss, batch_size mac
     dinh = 128 (per-process, xem phan Distributed ben duoi).
  2. Task loss la causal LM tren TARGET LANGUAGE.
  3. Cap ngon ngu la english - other (khong phai cap other-other).
  4. Alignment (va LoRA) CHI ap dung tren dung 1 layer (mac dinh layer 16).

Cac dieu kien thay doi theo yeu cau:
  1. Dataset/DataLoader: doc bitext english-other duoc SAMPLE tu du lieu goc
     dang JSON multiway-parallel (moi record co nhieu key = ma ngon ngu,
     vd "eng_Latn", "ace_Arab", ...). Voi moi record, cau eng_Latn duoc ghep
     voi TUNG ngon ngu khac trong record de tao thanh 1 cap bitext rieng.
  2. LoRA chi ap dung tren dung middle layer 16 (khong con la range
     [L/3, 2L/3) nhu code goc).

Distributed training (thay the hoan toan co che OOM dynamic-split cua code
goc):
  - Khong con retry/chia doi batch khi OOM, khong con skip sample.
  - Chay multi-GPU bang torch.distributed (DistributedDataParallel), khoi
    chay qua torchrun. Sau moi optimizer step, TAT CA process dong bo qua
    dist.barrier() (dam bao moi GPU da chay xong step do) roi CHI rank 0
    thuc hien ghi checkpoint + push len Hugging Face Hub; sau khi rank 0
    xong, mot barrier thu hai dam bao cac rank khac cho truoc khi sang step
    tiep theo.
  - Checkpoint chi giu ban moi nhat: sau khi luu checkpoint-N thanh cong,
    checkpoint truoc do (vd checkpoint-(N - save_steps)) se bi xoa
    (shutil.rmtree) ngay lap tuc.

Vi kien truc chi tiet cua Marco-Nano-Instruct khong duoc cung cap truoc,
script nay TU DONG DO TIM cac module attention / router / experts bang ten
(regex) thay vi hard-code, va cho phep override qua CLI neu can.

Vi du chay (4 GPU tren 1 node):
    torchrun --standalone --nproc_per_node=4 Macro-Nano-Instruct.py \
        --model_name_or_path ATH-MaaS/Marco-Nano-Instruct \
        --data_dir data/processed_alignment \
        --align_layer 16 \
        --push_to_hub

Chay 1 GPU / CPU (khong can torchrun):
    python Macro-Nano-Instruct.py --model_name_or_path ATH-MaaS/Marco-Nano-Instruct

Resume:
    torchrun --standalone --nproc_per_node=4 Macro-Nano-Instruct.py --resume_from_checkpoint auto
"""

import argparse
import gc
import glob
import json
import logging
import math
import os
import random
import re
import shutil
import time
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

try:
    from huggingface_hub import HfApi
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("midalign_marco_nano")


# Cac ten bien moi truong pho bien cho HF token, thu theo thu tu nay
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_hf_token(env_file: Optional[str], cli_token: Optional[str]) -> Optional[str]:
    """Uu tien: --hf_token (CLI) > bien moi truong da set san > file .env (qua dotenv)."""
    if cli_token:
        logger.info("Dung HF token truyen qua --hf_token.")
        return cli_token

    for var in _HF_TOKEN_ENV_VARS:
        if os.environ.get(var):
            logger.info(f"Dung HF token co san trong bien moi truong {var}.")
            return os.environ[var]

    if env_file and os.path.exists(env_file):
        if not DOTENV_AVAILABLE:
            logger.warning(
                f"Tim thay {env_file} nhung chua cai python-dotenv "
                f"(pip install python-dotenv --break-system-packages) -> khong the tu dong doc HF_TOKEN."
            )
            return None
        load_dotenv(env_file, override=False)
        for var in _HF_TOKEN_ENV_VARS:
            if os.environ.get(var):
                logger.info(f"Da nap HF token tu {env_file} (bien {var}).")
                return os.environ[var]
        logger.warning(f"Da nap {env_file} nhung khong tim thay bien {_HF_TOKEN_ENV_VARS} ben trong.")
        return None

    logger.info(
        f"Khong tim thay HF token (khong co --hf_token, bien moi truong, hay file {env_file}). "
        f"Tiep tuc khong xac thuc — chi hoat dong voi model/repo public."
    )
    return None


# ============================================================================================
# Argparse
# ============================================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MidAlign baseline (alternate CLM + contrastive align) cho MoE Marco-Nano-Instruct")

    # Model / data / output
    p.add_argument("--model_name_or_path", type=str, default="ATH-MaaS/Marco-Nano-Instruct")
    p.add_argument("--data_dir", type=str, default="data/processed_alignment")
    p.add_argument("--data_files", type=str, nargs="+",
                    default=["flores.json", "bible.json", "ntrex.json"])
    p.add_argument("--output_dir", type=str,
                    default="training/MidAlign/checkpoints/Macro-Nano-Instruct")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Gioi han so cap bitext (debug/smoke test), None = dung het du lieu")

    # Bitext english-other (dieu kien thay doi #1)
    p.add_argument("--eng_key", type=str, default="eng_Latn",
                    help="Ten key tieng Anh trong moi record JSON (vd 'eng_Latn')")
    p.add_argument("--max_lang_pairs_per_record", type=int, default=None,
                    help="Gioi han so ngon ngu khac duoc ghep voi eng_key trong 1 record "
                         "(None = dung tat ca ngon ngu co trong record, vd toan bo FLORES-200)")

    # Hugging Face Hub
    p.add_argument("--push_to_hub", action="store_true", default=True)
    p.add_argument("--no_push_to_hub", dest="push_to_hub", action="store_false")
    p.add_argument("--hub_model_id", type=str, default="ducanhdinh/Macro-Nano-Instruct-MidAlign")
    p.add_argument("--hub_private", action="store_true")
    p.add_argument("--env_file", type=str, default=".env",
                    help="Duong dan file .env chua HF_TOKEN, tu dong nap bang python-dotenv")
    p.add_argument("--hf_token", type=str, default=None,
                    help="Override HF token thu cong, uu tien cao hon .env/bien moi truong")

    # Training schedule
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128,
                    help="Batch size per-process (moi GPU xu ly ngan nay cap bitext / step)")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2, help="So worker cho DataLoader")

    # MidAlign: alignment objective (dieu kien giu nguyen #4)
    p.add_argument("--align_layer", type=int, default=16,
                    help="Layer dung de trich xuat hidden state cho contrastive loss VA gan "
                         "LoRA. Quy uoc: hidden_states[align_layer], tuc output SAU decoder "
                         "block co index 0-based = align_layer - 1 (giong truc Layer ID trong "
                         "paper MidAlign, Layer ID 0 = embedding).")
    p.add_argument("--align_temperature", type=float, default=0.1,
                    help="Nhiet do tau cho contrastive loss (tune tren dev loss, xem App. D.1 "
                         "paper MidAlign: Llama dung 0.1, Qwen dung 1.5)")

    # MoE loss (phu tro cho task step, xem docstring dau file)
    p.add_argument("--lb_loss_coef", type=float, default=None,
                    help="He so cho load-balancing loss. None = lay tu config.router_aux_loss_coef, "
                         "fallback 0.01. Dat = 0 neu chi muon task loss la CLM thuan tuy.")
    p.add_argument("--num_local_experts", type=int, default=None,
                    help="Override so luong experts, None = tu doc trong config model")
    p.add_argument("--num_experts_per_tok", type=int, default=None,
                    help="Override top-k router, None = tu doc trong config model")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Checkpoint / resume
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="'auto' de tu tim checkpoint moi nhat trong output_dir, hoac duong dan cu the")

    # Distributed
    p.add_argument("--local_rank", type=int, default=-1,
                    help="Duoc torchrun/torch.distributed.launch tu dong truyen qua bien moi "
                         "truong LOCAL_RANK; CLI arg nay chi la fallback.")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Chi dung khi CHAY DON PROCESS (khong qua torchrun)")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--diagnostics_dir", type=str, default=None,
                    help="None = <output_dir>/diagnostics")
    p.add_argument("--log_every", type=int, default=10, help="Cap nhat plot loss moi N step")

    return p


# ============================================================================================
# Utils chung
# ============================================================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================================================
# Distributed setup: thay the hoan toan co che OOM dynamic-split cua code goc bang DDP.
# ============================================================================================
def setup_distributed(args) -> Tuple[bool, int, int, int, torch.device]:
    """Tra ve (is_distributed, local_rank, global_rank, world_size, device).

    Neu duoc khoi chay qua torchrun (WORLD_SIZE > 1 trong bien moi truong), khoi tao
    process group va tra ve thong tin distributed. Nguoc lai, chay don process nhu binh
    thuong (tuong thich nguoc, khong bat buoc phai co torchrun)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        global_rank = dist.get_rank()
        logger.info(f"[DDP] Da khoi tao process group: backend={backend}, "
                    f"global_rank={global_rank}/{world_size}, local_rank={local_rank}")
        return True, local_rank, global_rank, world_size, device
    return False, 0, 0, 1, torch.device(args.device)


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def get_underlying_model(model):
    """Tra ve PeftModel thuc su ben duoi, bo qua lop boc DDP neu co."""
    return model.module if hasattr(model, "module") else model


# ============================================================================================
# Du lieu: doc bitext english-other tu du lieu multiway-parallel dang JSON (dieu kien thay
# doi #1). Moi record trong file JSON co dang {"id": ..., "eng_Latn": "...", "<lang>": "...", ...}
# -> voi moi ngon ngu khac ngoai eng_key, tao 1 cap (eng_text, other_text, lang_code).
# ============================================================================================
def load_bitext_pairs(data_dir: str, data_files: Sequence[str], eng_key: str,
                       max_lang_pairs_per_record: Optional[int] = None,
                       seed: int = 42) -> List[Tuple[str, str, str]]:
    pairs: List[Tuple[str, str, str]] = []
    rng = random.Random(seed)
    for fname in data_files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"Khong tim thay file {path}, bo qua.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = list(data.values()) if isinstance(data, dict) else data

        n_before = len(pairs)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            eng_text = rec.get(eng_key)
            if not isinstance(eng_text, str) or not eng_text.strip():
                continue
            eng_text = eng_text.strip()

            other_keys = [k for k in rec.keys() if k not in ("id", eng_key)]
            if max_lang_pairs_per_record is not None and len(other_keys) > max_lang_pairs_per_record:
                other_keys = rng.sample(other_keys, max_lang_pairs_per_record)

            for k in other_keys:
                v = rec.get(k)
                if isinstance(v, str) and v.strip():
                    pairs.append((eng_text, v.strip(), k))

        logger.info(f"{fname}: +{len(pairs) - n_before} cap bitext ({eng_key}-other), "
                    f"tong so record = {len(records)}")
    return pairs


class BitextPairDataset(Dataset):
    """Moi sample la 1 tuple (eng_text, other_text, lang_code). Tokenize duoc thuc hien
    theo tung batch trong vong lap training (khong tokenize truoc toan bo)."""

    def __init__(self, pairs: List[Tuple[str, str, str]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_bitext(batch: List[Tuple[str, str, str]]):
    eng_texts, other_texts, lang_codes = zip(*batch)
    return list(eng_texts), list(other_texts), list(lang_codes)


# ============================================================================================
# Tu dong tim target module cho LoRA: attention / router / experts trong DUNG 1 layer
# (dieu kien thay doi #2 — khong con la range [L/3, 2L/3) nhu code goc).
# ============================================================================================
LAYER_IDX_PATTERN = re.compile(r"\.(?:layers|h|blocks|block)\.(\d+)\.")


def get_num_layers(config) -> int:
    for attr in ("num_hidden_layers", "num_layers", "n_layer", "n_layers"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError("Khong tim thay so luong layer trong model.config. Hay kiem tra ten attribute.")


def is_router_leaf_name(name: str) -> bool:
    leaf = name.split(".")[-1]
    return leaf in ("gate", "router", "gating") and ".experts." not in name


def build_lora_target_modules(model, layer_indices: set) -> List[str]:
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        m = LAYER_IDX_PATTERN.search("." + name)
        if not m:
            continue
        idx = int(m.group(1))
        if idx not in layer_indices:
            continue
        is_attn = bool(re.search(r"(self_attn|attention|attn)\.", name))
        is_expert = ".experts." in name or ".expert." in name
        is_router = is_router_leaf_name(name)
        if is_attn or is_expert or is_router:
            targets.append(name)
    return targets


def register_router_hooks(peft_model, router_names: List[str], cache_list: list):
    """Hook forward tren cac router Linear (da duoc PEFT wrap LoRA) de lay logits phuc vu
    tinh load-balancing loss. Khong detach de gradient van chay ve LoRA cua router."""
    hooks = []
    router_name_set = set(router_names)
    if not router_name_set:
        return hooks
    matched = set()
    for name, module in peft_model.named_modules():
        if any(name.endswith(rn) for rn in router_name_set):
            h = module.register_forward_hook(lambda mod, inp, out, cache=cache_list: cache.append(out))
            hooks.append(h)
            matched.add(name)
    if len(hooks) != len(router_name_set):
        logger.warning(
            f"Da dang ky {len(hooks)} hook nhung co {len(router_name_set)} router target "
            f"-> kiem tra lai neu so luong khong khop (co the do trung ten suffix)."
        )
    return hooks


def infer_moe_dims(config, args):
    num_experts = args.num_local_experts
    top_k = args.num_experts_per_tok
    if num_experts is None:
        for attr in ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"):
            if hasattr(config, attr):
                num_experts = int(getattr(config, attr))
                break
    if top_k is None:
        for attr in ("num_experts_per_tok", "moe_top_k", "top_k", "num_selected_experts"):
            if hasattr(config, attr):
                top_k = int(getattr(config, attr))
                break
    if num_experts is None or top_k is None:
        logger.warning(
            "Khong tu suy ra duoc num_experts/top_k tu model.config. "
            "L_LB se = 0 tru khi ban truyen --num_local_experts va --num_experts_per_tok thu cong."
        )
    return num_experts, top_k


# ============================================================================================
# MoE load-balancing loss chuan (Switch/Mixtral style) — phu tro cho task step (xem docstring)
# ============================================================================================
def compute_load_balancing_loss(router_logits_list: List[torch.Tensor], attention_mask: torch.Tensor,
                                 num_experts: int, top_k: int):
    """attention_mask: [batch, seq_len] (1 = token that, 0 = padding). Phai loai bo vi tri
    padding truoc khi tinh bat ky thong ke nao, tuong tu cach L_LM loai padding qua
    ignore_index=-100."""
    mask_flat = attention_mask.reshape(-1).bool()

    losses = []
    for logits in router_logits_list:
        logits = logits.reshape(-1, logits.shape[-1])
        if logits.shape[0] == mask_flat.shape[0]:
            logits = logits[mask_flat]
        else:
            logger.warning(
                "compute_load_balancing_loss: kich thuoc router logits "
                f"({logits.shape[0]}) khong khop attention_mask ({mask_flat.shape[0]}) -> "
                "bo qua loc padding cho lan tinh nay."
            )
        if logits.shape[0] == 0:
            continue
        routing_weights = F.softmax(logits, dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        expert_mask = F.one_hot(selected_experts, num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)
        avg_prob_per_expert = routing_weights.mean(dim=0)
        loss = num_experts * torch.sum(tokens_per_expert * avg_prob_per_expert)
        losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=attention_mask.device)
    return torch.stack(losses).mean()


# ============================================================================================
# Task step: causal LM tren TARGET LANGUAGE (dieu kien giu nguyen #2) (+ L_LB phu tro MoE)
# ============================================================================================
def compute_task_step(other_texts: List[str], tokenizer, model, max_length: int, device,
                       router_logits_cache: list, num_experts, top_k, lb_loss_coef: float):
    enc = tokenizer(other_texts, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt")
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attention_mask = enc["attention_mask"].to(device, non_blocking=True)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    router_logits_cache.clear()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    lm_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    if router_logits_cache and num_experts and top_k:
        lb_loss = compute_load_balancing_loss(router_logits_cache, attention_mask, num_experts, top_k)
        lb_loss = lb_loss.to(lm_loss.device)
    else:
        lb_loss = torch.zeros((), device=lm_loss.device)

    total_loss = lm_loss + lb_loss_coef * lb_loss
    return lm_loss, lb_loss, total_loss


# ============================================================================================
# Align step: contrastive loss tai DUNG layer 16, cap english-other (dieu kien giu nguyen #3, #4)
# ============================================================================================
def mean_pool_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def encode_layer_representation(texts: List[str], tokenizer, model, align_layer: int,
                                 max_length: int, device, router_logits_cache: list) -> torch.Tensor:
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attention_mask = enc["attention_mask"].to(device, non_blocking=True)

    router_logits_cache.clear()  # khong dung cho align step, chi de tranh cache tich luy
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    # Quy uoc: hidden_states[0] = embedding output, hidden_states[i] = output SAU decoder
    # block index 0-based (i-1). align_layer=16 -> output sau block thu 16 (1-indexed),
    # dung "Layer ID" nhu truc x trong Figure 1/4 cua paper MidAlign.
    hidden = outputs.hidden_states[align_layer]
    pooled = mean_pool_hidden(hidden, attention_mask)
    return pooled


def compute_alignment_step(eng_texts: List[str], other_texts: List[str], tokenizer, model,
                            align_layer: int, max_length: int, device, temperature: float,
                            router_logits_cache: list):
    pooled_eng = encode_layer_representation(eng_texts, tokenizer, model, align_layer,
                                              max_length, device, router_logits_cache)
    pooled_other = encode_layer_representation(other_texts, tokenizer, model, align_layer,
                                                max_length, device, router_logits_cache)

    a = F.normalize(pooled_eng, dim=-1)
    b = F.normalize(pooled_other, dim=-1)
    sim = torch.matmul(a, b.t()) / temperature  # [n, n]

    target = torch.arange(sim.size(0), device=sim.device)
    # Cong thuc nay tuong duong Eq.1 trong paper (-log softmax cua cap dung), tinh doi
    # xung ca 2 chieu eng->other va other->eng (chuan InfoNCE), roi lay trung binh.
    loss_e2o = F.cross_entropy(sim, target)
    loss_o2e = F.cross_entropy(sim.t(), target)
    align_loss = (loss_e2o + loss_o2e) / 2.0
    return align_loss


# ============================================================================================
# Checkpoint / resume — chi giu 1 checkpoint moi nhat, tu dong xoa checkpoint cu
# ============================================================================================
def save_checkpoint_and_rotate(output_dir, model, optimizer, scheduler, epoch, step_in_epoch,
                                global_step, prev_checkpoint_dir: Optional[str]) -> str:
    underlying = get_underlying_model(model)
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    underlying.save_pretrained(ckpt_dir)  # PeftModel: chi luu adapter LoRA
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        },
        os.path.join(ckpt_dir, "trainer_state.pt"),
    )
    with open(os.path.join(output_dir, "latest_checkpoint.txt"), "w") as f:
        f.write(ckpt_dir)

    # Chi giu checkpoint moi nhat: xoa checkpoint truoc do ngay sau khi luu xong checkpoint moi.
    if prev_checkpoint_dir and os.path.isdir(prev_checkpoint_dir) and prev_checkpoint_dir != ckpt_dir:
        shutil.rmtree(prev_checkpoint_dir, ignore_errors=True)
        logger.info(f"Da xoa checkpoint cu: {prev_checkpoint_dir}")

    return ckpt_dir


def find_resume_checkpoint(output_dir, resume_arg: Optional[str]) -> Optional[str]:
    if resume_arg is None:
        return None
    if resume_arg == "auto":
        pointer = os.path.join(output_dir, "latest_checkpoint.txt")
        if os.path.exists(pointer):
            with open(pointer) as f:
                path = f.read().strip()
            if os.path.isdir(path):
                return path
        candidates = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if candidates:
            candidates.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
            return candidates[-1]
        return None
    return resume_arg if os.path.isdir(resume_arg) else None


# ============================================================================================
# Diagnostics: jsonl + plot (tach rieng duong task loss va align loss vi 2 loai step khac nhau)
# ============================================================================================
def log_step_to_jsonl(jsonl_path, global_step, epoch, step_type, lm_loss=None, lb_loss=None,
                       task_total_loss=None, align_loss=None):
    rec = {
        "step": global_step,
        "epoch": epoch,
        "step_type": step_type,
        "lm_loss": lm_loss,
        "lb_loss": lb_loss,
        "task_total_loss": task_total_loss,
        "align_loss": align_loss,
        "timestamp": time.time(),
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def plot_losses(jsonl_path, out_png, align_layer):
    if not os.path.exists(jsonl_path):
        return
    task_steps, lm, lb, task_total = [], [], [], []
    align_steps, align = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["step_type"] == "task":
                task_steps.append(rec["step"])
                lm.append(rec["lm_loss"])
                lb.append(rec["lb_loss"])
                task_total.append(rec["task_total_loss"])
            else:
                align_steps.append(rec["step"])
                align.append(rec["align_loss"])
    if not task_steps and not align_steps:
        return

    plt.figure(figsize=(10, 6))
    if task_steps:
        plt.plot(task_steps, lm, label="L_LM (task step, target language)")
        plt.plot(task_steps, lb, label="L_LB (task step, MoE)")
        plt.plot(task_steps, task_total, label="L_task_total (task step)")
    if align_steps:
        plt.plot(align_steps, align, label=f"L_align (contrastive @ layer {align_layer})")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("MidAlign baseline (Macro-Nano-Instruct) — alternate task/align training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ============================================================================================
# Hugging Face Hub push
# ============================================================================================
def build_model_card(args, num_experts, top_k, align_layer_0based, num_layers) -> str:
    return f"""---
license: apache-2.0
base_model: {args.model_name_or_path}
tags:
- lora
- peft
- moe
- mixture-of-experts
- cross-lingual-alignment
- midalign
- machine-translation
---

# Macro-Nano-Instruct-MidAlign

LoRA adapter finetune tu [`{args.model_name_or_path}`]\
(https://huggingface.co/{args.model_name_or_path}), mot mo hinh Mixture-of-Experts, theo
baseline **MidAlign** (Middle-Layer Representation Alignment, Liu & Niehues 2025) — Alternate
Training giua task objective (causal LM tren target language) va alignment objective
(contrastive loss tai 1 middle layer) — adapt sang backbone MoE.

## Cau hinh LoRA / Alignment
- LoRA + trich xuat hidden state cho contrastive loss CHI ap dung tai layer thu
  `{args.align_layer}` (0-indexed block = {align_layer_0based}) trong tong so `{num_layers}` layer.
- Module duoc gan LoRA: **attention**, **router**, **experts** tai layer tren.
- r = {args.lora_r}, alpha = {args.lora_alpha}, dropout = {args.lora_dropout}
- Nhiet do contrastive tau = {args.align_temperature}

## Loss (Alternate Training — moi step chi 1 trong 2)
- **Task step**: `L_task = L_LM + lb_loss_coef * L_LB`
  - `L_LM`: causal LM loss tren cau TARGET LANGUAGE (phia "other" trong cap english-other).
  - `L_LB`: load balancing loss chuan cua MoE tai router trong layer duoc finetune.
  - `lb_loss_coef` = {args.lb_loss_coef}, `num_experts` = {num_experts}, `top_k` = {top_k}
- **Align step**: `L_align` = symmetric InfoNCE / contrastive loss (in-batch negatives) giua
  mean-pooled hidden state cua cau tieng Anh va cau target tai layer {args.align_layer}.

## Du lieu
Cap bitext english-other duoc sample tu cac bo du lieu multiway-parallel: `{", ".join(args.data_files)}`.
Voi moi record, cau `{args.eng_key}` duoc ghep voi tung ngon ngu khac trong cung record de tao
1 cap bitext rieng.

## Training
- {args.num_train_epochs} epoch, batch_size = {args.batch_size} (per-process).
- Multi-GPU: DistributedDataParallel (torchrun), checkpoint chi giu ban moi nhat.

## Diagnostics
Xem `diagnostics/loss_log.jsonl` (log theo tung step, phan biet step_type=task/align) va
`diagnostics/loss_curve.png`.
"""


def push_to_hub(local_ckpt_dir, diagnostics_dir, hub_model_id, private, readme_text):
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub chua duoc cai, bo qua buoc push_to_hub.")
        return
    api = HfApi()
    api.create_repo(repo_id=hub_model_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=local_ckpt_dir, repo_id=hub_model_id, path_in_repo=".",
                       commit_message=f"Update checkpoint: {os.path.basename(local_ckpt_dir)}")
    if os.path.isdir(diagnostics_dir):
        api.upload_folder(folder_path=diagnostics_dir, repo_id=hub_model_id,
                           path_in_repo="diagnostics", commit_message="Update diagnostics")
    readme_path = os.path.join(local_ckpt_dir, "_README_tmp.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_text)
    api.upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=hub_model_id,
                     commit_message="Update model card")
    os.remove(readme_path)


# ============================================================================================
# Main
# ============================================================================================
def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    is_distributed, local_rank, global_rank, world_size, device = setup_distributed(args)
    is_main_process = (global_rank == 0)

    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    diagnostics_dir = args.diagnostics_dir or os.path.join(args.output_dir, "diagnostics")
    if is_main_process:
        os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = os.path.join(diagnostics_dir, "loss_log.jsonl")
    plot_path = os.path.join(diagnostics_dir, "loss_curve.png")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # ---------------------------------------------------------------------------------- model
    if is_main_process:
        logger.info(f"Dang load tokenizer va model tu {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                               trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
    )
    base_model.to(device)

    num_layers = get_num_layers(base_model.config)
    if not (1 <= args.align_layer <= num_layers):
        raise ValueError(f"--align_layer={args.align_layer} phai nam trong [1, {num_layers}] "
                          f"(model co {num_layers} layer).")
    align_layer_0based = args.align_layer - 1  # dung de match ten module trong named_modules()
    layer_indices = {align_layer_0based}
    if is_main_process:
        logger.info(f"Tong so layer = {num_layers}. LoRA + alignment CHI ap dung tai layer "
                    f"{args.align_layer} (block 0-indexed = {align_layer_0based}, "
                    f"hidden_states[{args.align_layer}]).")

    target_modules = build_lora_target_modules(base_model, layer_indices)
    if not target_modules:
        raise RuntimeError(
            "Khong tim thay module (attention/router/experts) nao tai layer da chon. "
            "Kien truc model co the dat ten khac quy uoc — kiem tra lai regex trong build_lora_target_modules()."
        )
    router_target_names = [n for n in target_modules if is_router_leaf_name(n)]
    if is_main_process:
        logger.info(f"Tim thay {len(target_modules)} target module cho LoRA "
                    f"({len(router_target_names)} router). Vi du: {target_modules[:8]}")

    num_experts, top_k = infer_moe_dims(base_model.config, args)

    lb_loss_coef = args.lb_loss_coef
    if lb_loss_coef is None:
        lb_loss_coef = float(getattr(base_model.config, "router_aux_loss_coef", 0.01))
    if is_main_process:
        logger.info(f"lb_loss_coef (phu tro task step) = {lb_loss_coef}")

    # ---------------------------------------------------------------------------- resume / LoRA
    resume_dir = find_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_dir:
        if is_main_process:
            logger.info(f"Resume LoRA adapter tu checkpoint: {resume_dir}")
        model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(base_model, lora_config)
    if is_main_process:
        model.print_trainable_parameters()
    model.to(device)

    if is_distributed:
        ddp_kwargs = dict(device_ids=[local_rank], output_device=local_rank) if torch.cuda.is_available() else {}
        model = DDP(model, find_unused_parameters=False, **ddp_kwargs)

    router_logits_cache: list = []
    hooks = register_router_hooks(get_underlying_model(model), router_target_names, router_logits_cache)

    # ------------------------------------------------------------------------------------ data
    if is_main_process:
        logger.info(f"Dang doc bitext {args.eng_key}-other tu {args.data_dir} ({args.data_files}) ...")
    pairs = load_bitext_pairs(args.data_dir, args.data_files, args.eng_key,
                               args.max_lang_pairs_per_record, args.seed)
    if args.max_samples:
        random.Random(args.seed).shuffle(pairs)
        pairs = pairs[: args.max_samples]
    if is_main_process:
        logger.info(f"Tong so cap bitext sau khi gom du lieu: {len(pairs)}")
    if len(pairs) == 0:
        raise RuntimeError("Khong doc duoc cap bitext nao — kiem tra lai --data_dir / --data_files / --eng_key.")

    dataset = BitextPairDataset(pairs)
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=global_rank,
                                      shuffle=True, seed=args.seed, drop_last=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                 collate_fn=collate_bitext, num_workers=args.num_workers,
                                 drop_last=True, pin_memory=torch.cuda.is_available())
    else:
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                 collate_fn=collate_bitext, num_workers=args.num_workers,
                                 drop_last=True, pin_memory=torch.cuda.is_available())

    # ------------------------------------------------------------------------------- optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                   weight_decay=args.weight_decay, foreach=True)

    steps_per_epoch = len(dataloader)
    total_steps = steps_per_epoch * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    start_epoch, start_step_in_epoch, global_step = 0, 0, 0
    prev_checkpoint_dir = resume_dir  # checkpoint hien co tren dia (se bi xoa khi luu ban moi)
    if resume_dir:
        state_path = os.path.join(resume_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            for group in optimizer.param_groups:
                group["foreach"] = True
            if state.get("scheduler"):
                scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"]
            start_step_in_epoch = state["step_in_epoch"] + 1
            global_step = state["global_step"]
            torch.set_rng_state(state["torch_rng_state"])
            random.setstate(state["python_rng_state"])
            if is_main_process:
                logger.info(f"Da resume: epoch={start_epoch}, step_in_epoch={start_step_in_epoch}, "
                            f"global_step={global_step}")
            if start_step_in_epoch >= steps_per_epoch:
                start_epoch += 1
                start_step_in_epoch = 0

    readme_text = build_model_card(args, num_experts, top_k, align_layer_0based, num_layers)

    # ------------------------------------------------------------------------------ training loop
    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            if is_distributed:
                sampler.set_epoch(epoch)
            step_offset = start_step_in_epoch if epoch == start_epoch else 0

            pbar = tqdm(
                enumerate(dataloader),
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
                disable=not is_main_process,
            )
            for step_in_epoch, batch in pbar:
                if step_in_epoch < step_offset:
                    continue  # dang resume: bo qua nhanh cac batch da xu ly o lan chay truoc

                eng_texts, other_texts, lang_codes = batch

                model.train()
                optimizer.zero_grad(set_to_none=True)

                # Alternate Training (dieu kien giu nguyen #1): chan = task step, le = align step
                step_type = "task" if (global_step % 2 == 0) else "align"

                if step_type == "task":
                    lm_loss, lb_loss, task_total = compute_task_step(
                        other_texts, tokenizer, model, args.max_length, device,
                        router_logits_cache, num_experts, top_k, lb_loss_coef,
                    )
                    task_total.backward()
                    log_kwargs = dict(lm_loss=lm_loss.item(), lb_loss=lb_loss.item(),
                                       task_total_loss=task_total.item(), align_loss=None)
                    postfix = {"type": "task", "L_LM": f"{lm_loss.item():.4f}",
                               "L_LB": f"{lb_loss.item():.4f}"}
                else:
                    align_loss = compute_alignment_step(
                        eng_texts, other_texts, tokenizer, model, args.align_layer,
                        args.max_length, device, args.align_temperature, router_logits_cache,
                    )
                    align_loss.backward()
                    log_kwargs = dict(lm_loss=None, lb_loss=None, task_total_loss=None,
                                       align_loss=align_loss.item())
                    postfix = {"type": "align", "L_align": f"{align_loss.item():.4f}"}

                torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1

                if is_distributed:
                    # Dam bao MOI GPU da chay xong step nay (forward+backward+optimizer.step)
                    # truoc khi sang phan checkpoint/push chi danh cho rank 0.
                    dist.barrier()

                if is_main_process:
                    pbar.set_postfix(postfix)
                    log_step_to_jsonl(jsonl_path, global_step, epoch, step_type, **log_kwargs)

                    if global_step % args.log_every == 0:
                        plot_losses(jsonl_path, plot_path, args.align_layer)

                    if global_step % args.save_steps == 0:
                        ckpt_dir = save_checkpoint_and_rotate(
                            args.output_dir, model, optimizer, scheduler,
                            epoch, step_in_epoch, global_step, prev_checkpoint_dir,
                        )
                        prev_checkpoint_dir = ckpt_dir
                        plot_losses(jsonl_path, plot_path, args.align_layer)
                        logger.info(f"Da luu checkpoint local: {ckpt_dir}")
                        if args.push_to_hub:
                            push_to_hub(ckpt_dir, diagnostics_dir, args.hub_model_id,
                                        args.hub_private, readme_text)
                            logger.info(f"Da push checkpoint len hub: {args.hub_model_id}")

                if is_distributed:
                    # Cac rank khac cho rank 0 ghi checkpoint/push xong roi moi sang step tiep theo.
                    dist.barrier()

            start_step_in_epoch = 0

        # checkpoint cuoi cung sau khi hoan thanh training
        if is_main_process:
            final_ckpt = save_checkpoint_and_rotate(
                args.output_dir, model, optimizer, scheduler,
                args.num_train_epochs - 1, steps_per_epoch - 1, global_step, prev_checkpoint_dir,
            )
            plot_losses(jsonl_path, plot_path, args.align_layer)
            if args.push_to_hub:
                push_to_hub(final_ckpt, diagnostics_dir, args.hub_model_id, args.hub_private, readme_text)
            logger.info("Training hoan tat.")
        if is_distributed:
            dist.barrier()

    except KeyboardInterrupt:
        if is_main_process:
            logger.warning("Nhan KeyboardInterrupt — luu checkpoint khan cap truoc khi thoat ...")
            save_checkpoint_and_rotate(args.output_dir, model, optimizer, scheduler,
                                        epoch, step_in_epoch, global_step, prev_checkpoint_dir)
            plot_losses(jsonl_path, plot_path, args.align_layer)
        if is_distributed:
            dist.barrier()
        raise
    finally:
        for h in hooks:
            h.remove()
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()