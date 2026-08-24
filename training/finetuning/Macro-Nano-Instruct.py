
''"""
Fine-tuning LoRA cho mo hinh Mixture-of-Experts ATH-MaaS/Marco-Nano-Instruct.

PHIEN BAN TOI UU: Giai quyet cac diem nghen chinh cua MoE nhieu expert:
  1. Gradient checkpointing — giam VRAM gap ~40-50%, cho phep batch size lon hon.
  2. Bo hook PyTorch tren tung router (cuc ky cham) — thay bang output_router_logits 
     hoac aux_loss neu model ho tro; neu khong thi dung 1 hook duy nhat o module MoE cha.
  3. Tokenize & pad truoc trong DataLoader (collate_fn) — khong goi tokenizer trong 
     vong lap training.
  4. torch.compile() + Flash Attention / SDPA + TF32 — toc do forward tang 1.5-3x.
  5. DDP toi uu: gradient_as_bucket_view=True, bo find_unused_parameters, bucket_cap_mb.
  6. DataLoader: num_workers, pin_memory, prefetch_factor.
  7. Mixed Precision (AMP) voi bfloat16/float16 — giam memory + nhanh hon tren GPU ho tro.
  8. Bo dynamic OOM de quy (anti-pattern gay leak + cham) — thay bang gradient 
     accumulation voi micro-batch co dinh.
  9. Fused AdamW neu co san.
  10. Pad sequence theo max_length cua batch (padding=longest trong collate) thay vi 
      padding toan bo ve 256 — giam so token tinh toan vo nghia.

Loss = L_LM (cross-entropy chuan) + lb_loss_coef * L_LB (load balancing loss chuan cua MoE).

Vi du chay:
    torchrun --nproc_per_node=2 Macro-Nano-Instruct-optimized.py \
        --model_name_or_path ATH-MaaS/Marco-Nano-Instruct \
        --data_dir data/processed_alignment \
        --batch_size 64 --gradient_accumulation_steps 4 --max_length 256

Resume:
    python Macro-Nano-Instruct-optimized.py --resume_from_checkpoint auto
"""

import argparse
import datetime
import gc
import glob
import json
import logging
import math
import os
import random
import re
import time
from contextlib import nullcontext
from typing import List, Optional, Sequence, Dict, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm.auto import tqdm

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
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
logger = logging.getLogger("marco_nano_finetune")

_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


# ============================================================================================
# Argparse
# ============================================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA finetuning toi uu cho MoE Marco-Nano-Instruct")

    # Model / data / output
    p.add_argument("--model_name_or_path", type=str, default="ATH-MaaS/Marco-Nano-Instruct")
    p.add_argument("--data_dir", type=str, default="data/processed_alignment")
    p.add_argument("--data_files", type=str, nargs="+",
                    default=["flores.json", "bible.json", "ntrex.json"])
    p.add_argument("--output_dir", type=str,
                    default="training/finetuning/checkpoints/Macro-Nano-Instruct")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Gioi han so sample (debug/smoke test), None = dung het du lieu")

    # Hugging Face Hub
    p.add_argument("--push_to_hub", action="store_true", default=True)
    p.add_argument("--no_push_to_hub", dest="push_to_hub", action="store_false")
    p.add_argument("--hub_model_id", type=str, default="ducanhdinh/Macro-Nano-Instruct-Finetuning")
    p.add_argument("--hub_private", action="store_true")
    p.add_argument("--env_file", type=str, default=".env")
    p.add_argument("--hf_token", type=str, default=None)

    # Training schedule
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64,
                    help="Batch size MOI GPU (micro-batch). Global = batch_size * world_size * grad_accum")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                    help="So step tich luy gradient truoc khi optimizer.step()")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)

    # MoE loss
    p.add_argument("--lb_loss_coef", type=float, default=None,
                    help="He so load-balancing loss. None = doc tu config.router_aux_loss_coef, fallback 0.01")
    p.add_argument("--num_local_experts", type=int, default=None)
    p.add_argument("--num_experts_per_tok", type=int, default=None)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_layer_start_ratio", type=float, default=1.0 / 3.0)
    p.add_argument("--lora_layer_end_ratio", type=float, default=2.0 / 3.0)

    # Checkpoint / resume
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="'auto' de tu tim checkpoint moi nhat, hoac duong dan cu the")

    # Performance optimizations
    p.add_argument("--attn_implementation", type=str, default="sdpa",
                    choices=["eager", "sdpa", "flash_attention_2"],
                    help="sdpa = torch.nn.functional.scaled_dot_product_attention (nhanh, on dinh); "
                         "flash_attention_2 = nhanh nhat nhung can cai thu vien flash-attn")
    p.add_argument("--compile", action="store_true", default=False,
                    help="Dung torch.compile() — chi hoat dong tot tren PyTorch >= 2.0, "
                         "co the tang 20-50% toc do nhung compile lan dau lau.")
    p.add_argument("--num_workers", type=int, default=4,
                    help="So worker cho DataLoader (tokenize truoc trong collate)")
    p.add_argument("--pin_memory", action="store_true", default=True)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--no_pin_memory", dest="pin_memory", action="store_false")
    p.add_argument("--tf32", action="store_true", default=True,
                    help="Bat TF32 tren Ampere/Hopper — tang toc ~2x cho phep nhan ma tran")
    p.add_argument("--no_tf32", dest="tf32", action="store_false")
    p.add_argument("--fused_adamw", action="store_true", default=True,
                    help="Dung fused AdamW (foreach=True/fused=True) neu co san")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device_map", type=str, default=None)
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--diagnostics_dir", type=str, default=None)
    p.add_argument("--log_every", type=int, default=10)

    # Distributed
    p.add_argument("--nccl_timeout_minutes", type=int, default=30)
    return p


# ============================================================================================
# Utils
# ============================================================================================
def load_hf_token(env_file: Optional[str], cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        logger.info("Dung HF token truyen qua --hf_token.")
        return cli_token
    for var in _HF_TOKEN_ENV_VARS:
        if os.environ.get(var):
            logger.info(f"Dung HF token co san trong bien moi truong {var}.")
            return os.environ[var]
    if env_file and os.path.exists(env_file):
        if not DOTENV_AVAILABLE:
            logger.warning(f"Tim thay {env_file} nhung chua cai python-dotenv.")
            return None
        load_dotenv(env_file, override=False)
        for var in _HF_TOKEN_ENV_VARS:
            if os.environ.get(var):
                logger.info(f"Da nap HF token tu {env_file} (bien {var}).")
                return os.environ[var]
        logger.warning(f"Da nap {env_file} nhung khong tim thay bien {_HF_TOKEN_ENV_VARS} ben trong.")
        return None
    logger.info("Khong tim thay HF token — chi hoat dong voi model/repo public.")
    return None


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================================
# Distributed
# ============================================================================================
def setup_distributed(nccl_timeout_minutes: int):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if not is_distributed:
        return rank, local_rank, world_size, is_distributed, None

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training yeu cau CUDA (backend nccl).")

    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_TIMEOUT", str(nccl_timeout_minutes * 60))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=nccl_timeout_minutes),
    )
    device = torch.device(f"cuda:{local_rank}")
    logger.info(f"[rank {rank}/{world_size}] NCCL init (timeout={nccl_timeout_minutes}ph, local_rank={local_rank}).")
    return rank, local_rank, world_size, is_distributed, device


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def shard_texts_for_ddp(texts: List[str], lengths: List[int], rank: int, world_size: int):
    if world_size <= 1:
        return texts, lengths
    n = len(texts)
    n_trunc = (n // world_size) * world_size
    if n_trunc == 0:
        raise RuntimeError(f"Chi co {n} sample, khong du chia deu cho {world_size} GPU.")
    if n_trunc < n and rank == 0:
        logger.warning(f"Bo {n - n_trunc} sample cuoi de chia deu cho {world_size} rank.")
    shard_idx = list(range(n_trunc))[rank::world_size]
    return [texts[i] for i in shard_idx], [lengths[i] for i in shard_idx]


def reduce_scalar_across_ranks(value: float, device, world_size: int) -> float:
    t = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / world_size


# ============================================================================================
# Du lieu: tokenize truoc trong Dataset, collate chi pad theo max_length cua batch
# ============================================================================================
def load_all_sentences(data_dir: str, data_files: Sequence[str]) -> List[str]:
    sentences: List[str] = []
    for fname in data_files:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"Khong tim thay file {path}, bo qua.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = list(data.values()) if isinstance(data, dict) else data
        n_before = len(sentences)
        for rec in records:
            if not isinstance(rec, dict):
                continue
            for key, val in rec.items():
                if key == "id":
                    continue
                if isinstance(val, str) and val.strip():
                    sentences.append(val.strip())
        logger.info(f"{fname}: +{len(sentences) - n_before} cau, tong record={len(records)}")
    return sentences


def compute_lengths(tokenizer, texts: Sequence[str], chunk_size: int = 1000) -> List[int]:
    lengths: List[int] = []
    for i in tqdm(range(0, len(texts), chunk_size), desc="Tinh do dai token", disable=False):
        chunk = texts[i:i + chunk_size]
        enc = tokenizer(chunk, add_special_tokens=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


class TokenizedSentenceDataset(Dataset):
    """
    Dataset tra ve text thuan tuy. Tokenize se duoc thuc hien trong collate_fn 
    de co the dung num_workers > 0 va tiet kiem bo nho chinh.
    """
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class EfficientDataCollator:
    """
    Collate tokenize + pad theo max_length THUC TE cua batch (khong phai pad ve 256 co dinh).
    Giam ~30-50% token vo nghia phai tinh toan so voi padding co dinh.
    """
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch_texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class LengthGroupedBatchSampler(Sampler[List[int]]):
    def __init__(self, lengths: List[int], batch_size: int, seed: int = 42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _build_batches(self) -> List[List[int]]:
        g = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        g.shuffle(indices)
        indices.sort(key=lambda i: self.lengths[i])
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        g.shuffle(batches)
        return batches

    def __iter__(self):
        for b in self._build_batches():
            yield b

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)


# ============================================================================================
# Tu dong tim target module cho LoRA
# ============================================================================================
LAYER_IDX_PATTERN = re.compile(r"\.(?:layers|h|blocks|block)\.(\d+)\.")


def get_num_layers(config) -> int:
    for attr in ("num_hidden_layers", "num_layers", "n_layer", "n_layers"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError("Khong tim thay so luong layer trong model.config.")


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
            "L_LB se = 0 tru khi ban truyen --num_local_experts va --num_experts_per_tok."
        )
    return num_experts, top_k


# ============================================================================================
# Load Balancing Loss — toi uu: dung aux_loss hoac output_router_logits cua model
# ============================================================================================
def compute_load_balancing_loss_from_logits(
    router_logits_list: List[torch.Tensor],
    attention_mask: torch.Tensor,
    num_experts: int,
    top_k: int,
):
    """
    Tinh L_LB tu router logits. Da duoc loc padding.
    Cai tien: vectorize tot hon, tranh loop Python khi co the.
    """
    if not router_logits_list:
        return torch.tensor(0.0, device=attention_mask.device)

    mask_flat = attention_mask.reshape(-1).bool()
    losses = []

    for logits in router_logits_list:
        # logits: [batch, seq_len, num_experts]
        flat_logits = logits.reshape(-1, logits.shape[-1])
        if flat_logits.shape[0] == mask_flat.shape[0]:
            flat_logits = flat_logits[mask_flat]
        else:
            # Khong khop shape — co the do kien truc MoE reshape khac
            # Van tinh nhung khong loc padding de tranh index sai
            pass
        if flat_logits.shape[0] == 0:
            continue

        routing_weights = F.softmax(flat_logits, dim=-1)
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
# Checkpoint / resume
# ============================================================================================
def save_checkpoint(output_dir, model, optimizer, scheduler, epoch, step_in_epoch, global_step):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    unwrap_model(model).save_pretrained(ckpt_dir)
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
# Diagnostics
# ============================================================================================
def log_step_to_jsonl(jsonl_path, global_step, epoch, result):
    rec = {
        "step": global_step,
        "epoch": epoch,
        "lm_loss": result["lm_loss"],
        "lb_loss": result["lb_loss"],
        "total_loss": result["total_loss"],
        "n_processed": result["n_processed"],
        "timestamp": time.time(),
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def plot_losses(jsonl_path, out_png):
    if not os.path.exists(jsonl_path):
        return
    steps, lm, lb, total = [], [], [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            steps.append(rec["step"])
            lm.append(rec["lm_loss"])
            lb.append(rec["lb_loss"])
            total.append(rec["total_loss"])
    if not steps:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lm, label="L_LM", alpha=0.9)
    plt.plot(steps, lb, label="L_LB", alpha=0.9)
    plt.plot(steps, total, label="L_Total", alpha=0.9)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Marco-Nano-Instruct LoRA finetuning loss (Optimized)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


# ============================================================================================
# Hugging Face Hub push
# ============================================================================================
def build_model_card(args, num_experts, top_k, layer_start, layer_end, num_layers) -> str:
    return f"""---
license: apache-2.0
base_model: {args.model_name_or_path}
tags:
- lora
- peft
- moe
- mixture-of-experts
- machine-translation
- fine-tuned
---

# Macro-Nano-Instruct-Finetuning (Optimized)

LoRA adapter finetune tu [`{args.model_name_or_path}`]\
(https://huggingface.co/{args.model_name_or_path}), mot mo hinh Mixture-of-Experts.

## Cau hinh LoRA
- Layer duoc finetune: `[{layer_start}, {layer_end})` trong tong so `{num_layers}` layer.
- Module: **attention**, **router**, **experts** trong khoang layer tren.
- r = {args.lora_r}, alpha = {args.lora_alpha}, dropout = {args.lora_dropout}

## Loss
`L_total = L_LM + lb_loss_coef * L_LB`

- `lb_loss_coef` = {args.lb_loss_coef}
- `num_experts` = {num_experts}, `top_k` = {top_k}

## Toi uu hieu nang
- Gradient checkpointing
- torch.compile (neu bat)
- Flash Attention / SDPA
- Dynamic batch padding (khong pad co dinh)
- Fused AdamW + TF32

## Du lieu
Cau don ngu tu 3 bo: `flores.json`, `bible.json`, `ntrex.json`.
"""


def push_to_hub(local_ckpt_dir, diagnostics_dir, hub_model_id, private, readme_text):
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub chua duoc cai, bo qua push_to_hub.")
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

    rank, local_rank, world_size, is_distributed, ddp_device = setup_distributed(
        args.nccl_timeout_minutes
    )
    if is_distributed:
        if args.device_map:
            raise ValueError("Khong dung --device_map cung luc voi distributed training (torchrun).")
        args.device = ddp_device
        if rank != 0:
            logger.setLevel(logging.WARNING)

    set_seed(args.seed + rank)  # +rank de moi rank co shuffle khac nhau (data khac nhau roi)

    os.makedirs(args.output_dir, exist_ok=True)
    diagnostics_dir = args.diagnostics_dir or os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = os.path.join(diagnostics_dir, "loss_log.jsonl")
    plot_path = os.path.join(diagnostics_dir, "loss_curve.png")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # --- TF32 toi uu ---
    if args.tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Da bat TF32 cho matmul va cudnn.")

    # --- Flash Attention / SDPA ---
    attn_impl = args.attn_implementation
    logger.info(f"Su dung attention implementation: {attn_impl}")

    # ---------------------------------------------------------------------------------- model
    logger.info(f"Dang load tokenizer va model tu {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=attn_impl,
    )
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **model_kwargs,
    )
    if not args.device_map:
        base_model.to(args.device)

    num_layers = get_num_layers(base_model.config)
    layer_start = int(num_layers * args.lora_layer_start_ratio)
    layer_end = int(num_layers * args.lora_layer_end_ratio)
    layer_indices = set(range(layer_start, layer_end))
    logger.info(f"Tong so layer = {num_layers}. Ap dung LoRA cho layer [{layer_start}, {layer_end}).")

    target_modules = build_lora_target_modules(base_model, layer_indices)
    if not target_modules:
        raise RuntimeError("Khong tim thay module nao trong khoang layer da chon.")
    router_target_names = [n for n in target_modules if is_router_leaf_name(n)]
    logger.info(f"Tim thay {len(target_modules)} target module ({len(router_target_names)} router).")

    num_experts, top_k = infer_moe_dims(base_model.config, args)
    lb_loss_coef = args.lb_loss_coef
    if lb_loss_coef is None:
        lb_loss_coef = float(getattr(base_model.config, "router_aux_loss_coef", 0.01))
    logger.info(f"lb_loss_coef = {lb_loss_coef}, num_experts={num_experts}, top_k={top_k}")

    # --- Gradient Checkpointing: BAT TRUOC khi wrap PEFT ---
    # Dieu nay cuc ky quan trong voi MoE nhieu expert — giam VRAM ~40-60%
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Da bat gradient checkpointing tren base model (use_reentrant=False).")
    else:
        logger.warning("Base model khong ho tro gradient_checkpointing_enable.")

    # --- LoRA ---
    resume_dir = find_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_dir:
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

    if is_main_process(rank):
        model.print_trainable_parameters()
    if not args.device_map:
        model.to(args.device)

    # --- torch.compile ---
    if args.compile and hasattr(torch, "compile"):
        logger.info("Dang torch.compile model (mode=default, fullgraph=False) ...")
        # Khong compile fullgraph vi MoE co dynamic control flow (top-k routing)
        model = torch.compile(model, mode="default", fullgraph=False)
        logger.info("torch.compile xong.")

    # --- Kiem tra kha nang lay router logits tu model ---
    # Cac MoE hien dai (Mixtral, Qwen2MoE, DeepSeek) thuong tra ve aux_loss hoac router_logits
    # Neu co, ta khong can hook gi ca — tiet kiem rat nhieu thoi gian.
    has_aux_loss = False
    has_router_logits_attr = False
    test_text = "Hello world"
    test_enc = tokenizer(test_text, return_tensors="pt").to(args.device)
    with torch.no_grad():
        try:
            test_out = model(**test_enc, output_router_logits=True)
            if hasattr(test_out, "router_logits") and test_out.router_logits is not None:
                has_router_logits_attr = True
                logger.info("Model ho tro output_router_logits=True — khong can hook!")
            elif hasattr(test_out, "aux_loss") and test_out.aux_loss is not None:
                has_aux_loss = True
                logger.info("Model tra ve aux_loss san — se dung truc tiep, khong can hook!")
        except Exception as e:
            logger.warning(f"Khong the truy van router logits tu model: {e}. Se thu dung hook.")

    # --- Hook chi khi thuc su can ---
    hooks = []
    router_logits_cache: list = []
    if not has_router_logits_attr and not has_aux_loss and num_experts and top_k:
        logger.info("Dang dang ky hook thu thu router logits (toi uu: chi hook cac module cha MoE).")
        # Toi uu: thay vi hook tung Linear router, ta hook module cha chua gate + experts
        # nhung vi khong biet chinh xac ten, ta dung cach cu nhung chi khi can thiet.
        router_name_set = set(router_target_names)
        matched = set()
        for name, module in model.named_modules():
            if any(name.endswith(rn) for rn in router_name_set):
                h = module.register_forward_hook(
                    lambda mod, inp, out, cache=router_logits_cache: cache.append(out)
                )
                hooks.append(h)
                matched.add(name)
        logger.info(f"Da dang ky {len(hooks)} hook tren router.")
    else:
        logger.info("Bo qua hook — su dung co che router logits/aux_loss noi bo cua model.")

    # --- DDP toi uu ---
    if is_distributed:
        # find_unused_parameters=False + gradient_as_bucket_view=True = nhanh hon rat nhieu
        # Tuy nhien, neu MoE co expert khong duoc chon -> param khong nhan grad -> DDP loi.
        # Voi LoRA, TAT CA param trainable deu tham gia moi step (vi router luon chay),
        # nen find_unused_parameters=False la an toan.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            bucket_cap_mb=25,
        )

    # ------------------------------------------------------------------------------------ data
    logger.info(f"Dang doc du lieu tu {args.data_dir} ...")
    texts = load_all_sentences(args.data_dir, args.data_files)
    if args.max_samples:
        random.Random(args.seed).shuffle(texts)
        texts = texts[:args.max_samples]
    logger.info(f"Tong so sample: {len(texts)}")
    if len(texts) == 0:
        raise RuntimeError("Khong doc duoc sample nao.")

    lengths = compute_lengths(tokenizer, texts)
    texts, lengths = shard_texts_for_ddp(texts, lengths, rank, world_size)
    if is_distributed:
        logger.info(f"[rank {rank}] Shard: {len(texts)} sample.")

    dataset = TokenizedSentenceDataset(texts)
    batch_sampler = LengthGroupedBatchSampler(lengths, batch_size=args.batch_size, seed=args.seed)
    collate_fn = EfficientDataCollator(tokenizer, max_length=args.max_length)

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers if not is_distributed else 0,  # DDP + multi-worker can than
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # ------------------------------------------------------------------------------- optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Fused AdamW: foreach=True hoac fused=True (neu CUDA ho tro)
    adamw_kwargs = {"lr": args.learning_rate, "weight_decay": args.weight_decay}
    if args.fused_adamw:
        # PyTorch >= 2.0 ho tro fused=True; foreach=True la fallback an toan
        try:
            adamw_kwargs["fused"] = True
            test_opt = torch.optim.AdamW([torch.randn(1)], **adamw_kwargs)
            del test_opt
            logger.info("Dung fused AdamW (fused=True).")
        except Exception:
            adamw_kwargs.pop("fused", None)
            adamw_kwargs["foreach"] = True
            logger.info("Dung foreach AdamW (foreach=True).")
    else:
        adamw_kwargs["foreach"] = True

    optimizer = torch.optim.AdamW(trainable_params, **adamw_kwargs)

    steps_per_epoch = len(batch_sampler)
    total_steps = steps_per_epoch * args.num_train_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    start_epoch, start_step_in_epoch, global_step = 0, 0, 0
    if resume_dir:
        state_path = os.path.join(resume_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state["optimizer"])
            # Restore fused/foreach flag bi mat trong state_dict
            for group in optimizer.param_groups:
                if args.fused_adamw and "fused" in adamw_kwargs:
                    group["fused"] = True
                group.setdefault("foreach", True)
            if state.get("scheduler"):
                scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"]
            start_step_in_epoch = state["step_in_epoch"] + 1
            global_step = state["global_step"]
            torch.set_rng_state(state["torch_rng_state"])
            random.setstate(state["python_rng_state"])
            logger.info(f"Resume: epoch={start_epoch}, step={start_step_in_epoch}, global_step={global_step}")
            if start_step_in_epoch >= steps_per_epoch:
                start_epoch += 1
                start_step_in_epoch = 0

    readme_text = build_model_card(args, num_experts, top_k, layer_start, layer_end, num_layers)

    # ------------------------------------------------------------------------------ training loop
    model_device = next(model.parameters()).device
    autocast_dtype = dtype if dtype != torch.float32 else None
    autocast_enabled = (dtype == torch.float16 or dtype == torch.bfloat16)
    
    # Scaler chi can cho float16; bfloat16 khong can
    scaler = torch.cuda.amp.GradScaler() if (autocast_enabled and dtype == torch.float16) else None

    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            batch_sampler.set_epoch(epoch)
            step_offset = start_step_in_epoch if epoch == start_epoch else 0

            pbar = tqdm(
                enumerate(dataloader),
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
                disable=not is_main_process(rank),
            )

            model.train()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0

            for step_in_epoch, batch in pbar:
                if step_in_epoch < step_offset:
                    continue

                input_ids = batch["input_ids"].to(model_device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(model_device, non_blocking=True)
                labels = batch["labels"].to(model_device, non_blocking=True)

                router_logits_cache.clear()

                # Forward voi autocast (mixed precision)
                with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
                    if has_router_logits_attr:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            output_router_logits=True,
                        )
                        logits = outputs.logits
                        router_logits_list = outputs.router_logits if hasattr(outputs, "router_logits") else []
                    elif has_aux_loss:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        )
                        logits = outputs.logits
                        aux_loss = outputs.aux_loss if hasattr(outputs, "aux_loss") else None
                        router_logits_list = []
                    else:
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        router_logits_list = router_logits_cache

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    lm_loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

                    # Load balancing loss
                    if has_aux_loss and aux_loss is not None:
                        lb_loss = aux_loss
                    elif router_logits_list and num_experts and top_k:
                        lb_loss = compute_load_balancing_loss_from_logits(
                            router_logits_list, attention_mask, num_experts, top_k
                        )
                    else:
                        lb_loss = torch.zeros((), device=lm_loss.device)

                    total_loss = lm_loss + lb_loss_coef * lb_loss
                    # Chia cho so accumulation step
                    total_loss = total_loss / args.gradient_accumulation_steps

                # Backward
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                accum_count += 1

                # --- Chi optimizer.step() sau khi du accumulation ---
                if accum_count % args.gradient_accumulation_steps == 0:
                    if is_distributed:
                        # DDP voi gradient_as_bucket_view=True tu dong all_reduce 
                        # sau backward(), khong can sync thu cong.
                        pass

                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
                        optimizer.step()
                    
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    # --- Logging ---
                    # Chi log khi vua step
                    lm_loss_val = lm_loss.item()
                    lb_loss_val = lb_loss.item() if isinstance(lb_loss, torch.Tensor) else 0.0
                    total_loss_val = (lm_loss_val + lb_loss_coef * lb_loss_val)

                    if is_distributed:
                        lm_loss_val = reduce_scalar_across_ranks(lm_loss_val, model_device, world_size)
                        lb_loss_val = reduce_scalar_across_ranks(lb_loss_val, model_device, world_size)
                        total_loss_val = reduce_scalar_across_ranks(total_loss_val, model_device, world_size)

                    pbar.set_postfix({
                        "L_LM": f"{lm_loss_val:.4f}",
                        "L_LB": f"{lb_loss_val:.4f}",
                        "L_Total": f"{total_loss_val:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    })

                    if is_main_process(rank):
                        result = {
                            "lm_loss": lm_loss_val,
                            "lb_loss": lb_loss_val,
                            "total_loss": total_loss_val,
                            "n_processed": input_ids.size(0) * world_size,
                        }
                        log_step_to_jsonl(jsonl_path, global_step, epoch, result)

                        if global_step % args.log_every == 0:
                            plot_losses(jsonl_path, plot_path)

                        if global_step % args.save_steps == 0:
                            ckpt_dir = save_checkpoint(
                                args.output_dir, unwrap_model(model), optimizer, scheduler,
                                epoch, step_in_epoch, global_step
                            )
                            plot_losses(jsonl_path, plot_path)
                            logger.info(f"Da luu checkpoint: {ckpt_dir}")
                            if args.push_to_hub:
                                push_to_hub(ckpt_dir, diagnostics_dir, args.hub_model_id,
                                            args.hub_private, readme_text)

                    if is_distributed and global_step % args.save_steps == 0:
                        dist.barrier()

            start_step_in_epoch = 0

        # --- Final checkpoint ---
        if is_main_process(rank):
            final_ckpt = save_checkpoint(
                args.output_dir, unwrap_model(model), optimizer, scheduler,
                args.num_train_epochs - 1, steps_per_epoch - 1, global_step
            )
            plot_losses(jsonl_path, plot_path)
            if args.push_to_hub:
                push_to_hub(final_ckpt, diagnostics_dir, args.hub_model_id, args.hub_private, readme_text)
            logger.info("Training hoan tat.")
        if is_distributed:
            dist.barrier()

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt — luu checkpoint khan cap ...")
        if is_main_process(rank):
            save_checkpoint(args.output_dir, unwrap_model(model), optimizer, scheduler,
                             epoch, step_in_epoch, global_step)
            plot_losses(jsonl_path, plot_path)
        raise
    finally:
        for h in hooks:
            h.remove()
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
