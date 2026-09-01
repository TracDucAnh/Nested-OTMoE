"""
Fine-tuning LoRA cho mo hinh Mixture-of-Experts Qwen/Qwen1.5-MoE-A2.7B.

Loss = L_LM (cross-entropy chuan) + lb_loss_coef * L_LB (load balancing loss chuan cua MoE,
tinh tren cac router nam trong khoang layer duoc gan LoRA).

Cac tinh nang chinh:
  1. LoRA chi ap dung tren middle layers [L/3, 2L/3), chi len attention / router / experts,
     moi thanh phan mot rank rieng (mac dinh: router r=4, attention r=16, experts r=16).
  2. Checkpointing + resume tai bat ky epoch/step nao, luu moi 1000 step.
  3. Luu LoRA weight tai training/finetuning/checkpoints/Qwen1.5-MoE-A2.7B/.
  4. Dynamic batching: batch_size mac dinh 512, khi OOM thi chia doi de tri (dequy),
     clear memory sau moi lan chia, skip sample neu OOM ca khi batch_size = 1,
     tra ve batch_size goc ngay cho batch tiep theo.
  5. Dataset/DataLoader gom sample tu cac bo du lieu alignment duoc CHON qua --alignment_data
     (flores / ntrex / bible, co the ket hop nhieu bo, vi du --alignment_data flores ntrex),
     shuffle roi sort theo do dai.
  6. 3 epoch, tqdm day du.
  7. argparse day du de tuy bien.

Qwen1.5-MoE-A2.7B (Qwen2MoeForCausalLM) co kien truc router/experts dat ten theo quy uoc
chuan cua HF transformers (vd: model.layers.{i}.self_attn.*, model.layers.{i}.mlp.gate,
model.layers.{i}.mlp.experts.{e}.*, model.layers.{i}.mlp.shared_expert*). Du kien truc da
biet truoc, script nay VAN GIU nguyen co che TU DONG DO TIM cac module attention / router /
experts bang ten (regex) thay vi hard-code, de dam bao tinh tong quat va cho phep override
qua CLI neu can.

Vi du chay:
    python Qwen1.5-MoE-A2.7B.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --data_dir data/processed_alignment \
        --push_to_hub

Resume:
    python Qwen1.5-MoE-A2.7B.py --resume_from_checkpoint auto
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
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
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
logger = logging.getLogger("qwen15_moe_finetune")


# Cac ten bien moi truong pho bien cho HF token, thu theo thu tu nay
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Anh xa ten alignment dataset (dung trong --alignment_data) -> ten file JSON tuong ung
# trong --data_dir. Them entry moi vao day neu sau nay co them bo du lieu alignment khac.
ALIGNMENT_DATA_FILENAMES = {
    "flores": "flores.json",
    "ntrex": "ntrex.json",
    "bible": "bible.json",
}


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
    p = argparse.ArgumentParser(description="LoRA finetuning cho MoE Qwen1.5-MoE-A2.7B")

    # Model / data / output
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--data_dir", type=str, default="data/processed_alignment")
    p.add_argument("--alignment_data", type=str, nargs="+",
                    default=["flores", "ntrex", "bible"],
                    choices=sorted(ALIGNMENT_DATA_FILENAMES.keys()),
                    help="Chon 1 hoac nhieu bo du lieu alignment de finetune: flores, ntrex, "
                         "bible. Co the ket hop nhieu bo, vi du: --alignment_data flores ntrex "
                         "se chi dung flores + ntrex. Moi ten duoc anh xa toi 1 file JSON "
                         f"trong --data_dir: {ALIGNMENT_DATA_FILENAMES}.")
    p.add_argument("--output_dir", type=str,
                    default="training/finetuning/checkpoints/Qwen1.5-MoE-A2.7B")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Gioi han so sample (debug/smoke test), None = dung het du lieu")

    # Hugging Face Hub
    p.add_argument("--push_to_hub", action="store_true", default=True)
    p.add_argument("--no_push_to_hub", dest="push_to_hub", action="store_false")
    p.add_argument("--hub_model_id", type=str, default="ducanhdinh/Qwen1.5-MoE-A2.7B-Finetuning")
    p.add_argument("--hub_private", action="store_true")
    p.add_argument("--env_file", type=str, default=".env",
                    help="Duong dan file .env chua HF_TOKEN, tu dong nap bang python-dotenv")
    p.add_argument("--hf_token", type=str, default=None,
                    help="Override HF token thu cong, uu tien cao hon .env/bien moi truong")

    # Training schedule
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--min_batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)

    # MoE loss
    p.add_argument("--lb_loss_coef", type=float, default=None,
                    help="He so cho load-balancing loss. None = lay tu config.router_aux_loss_coef, "
                         "fallback 0.001 (dung mac dinh cua Qwen2MoeConfig/Qwen1.5-MoE, KHONG phai "
                         "0.01 nhu ban truoc — 0.01 se lam L_LB lan at L_LM neu model checkpoint "
                         "lai khong co san attribute nay trong config)")
    p.add_argument("--num_local_experts", type=int, default=None,
                    help="Override so luong experts, None = tu doc trong config model")
    p.add_argument("--num_experts_per_tok", type=int, default=None,
                    help="Override top-k router, None = tu doc trong config model")

    # LoRA - rank rieng cho tung thanh phan kien truc (goi bang PEFT rank_pattern)
    p.add_argument("--lora_r_router", type=int, default=4,
                    help="Rank LoRA rieng cho router/gating (mac dinh 4).")
    p.add_argument("--lora_r_attention", type=int, default=16,
                    help="Rank LoRA rieng cho attention Q/K/V/O (mac dinh 16).")
    p.add_argument("--lora_r_experts", type=int, default=16,
                    help="Rank LoRA rieng cho cac expert FFN (mac dinh 16).")
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_layer_start_ratio", type=float, default=1.0 / 3.0)
    p.add_argument("--lora_layer_end_ratio", type=float, default=2.0 / 3.0)

    # Checkpoint / resume
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="'auto' de tu tim checkpoint moi nhat trong output_dir, hoac duong dan cu the")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device_map", type=str, default=None,
                    help="vi du 'auto' cho multi-GPU. Neu set thi bo qua --device")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--diagnostics_dir", type=str, default=None,
                    help="None = <output_dir>/diagnostics")
    p.add_argument("--log_every", type=int, default=10, help="Cap nhat plot loss moi N step")
    p.add_argument("--smooth_window", type=int, default=50,
                    help="So step lien tiep duoc gom lai (tinh trung binh cong) cho moi diem "
                         "tren loss_curve_smoothed.png, giup duong cong de doc hon so voi ve "
                         "tho tung step (rat messy do nhieu step-to-step, dac biet la L_LB).")

    # Distributed (DDP qua torchrun: doc RANK / LOCAL_RANK / WORLD_SIZE tu bien moi truong,
    # khong can truyen tay). Chay 1 GPU binh thuong neu khong launch qua torchrun.
    p.add_argument("--nccl_timeout_minutes", type=int, default=30,
                    help="Timeout cho moi collective op cua NCCL. Mac dinh cua PyTorch la "
                         "10 phut, rat de bi NCCL Watchdog kill oan khi co buoc cham (load "
                         "checkpoint lon luc resume, push_to_hub tren rank 0, tokenize du "
                         "lieu lon, mang cham giua cac node...) -> tang len 30 phut.")

    return p


# ============================================================================================
# Utils chung
# ============================================================================================
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_oom_error(e: RuntimeError) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or "cuda error" in msg and "memory" in msg


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================================
# Distributed (DDP qua torchrun)
# ============================================================================================
def setup_distributed(nccl_timeout_minutes: int):
    """torchrun tu dong set san RANK / LOCAL_RANK / WORLD_SIZE trong bien moi truong. Neu
    chay bang `python script.py` binh thuong (khong qua torchrun) thi WORLD_SIZE khong ton
    tai hoac = 1 -> coi nhu single-process, KHONG init distributed, code chay y het ban goc
    (backward-compatible). device tra ve = None khi khong distributed, de main() giu nguyen
    --device / --device_map nguoi dung tu chon thay vi bi ghi de."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if not is_distributed:
        return rank, local_rank, world_size, is_distributed, None

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training (DDP) yeu cau CUDA (backend nccl).")

    # NCCL Watchdog mac dinh timeout sau 10 phut khong nhan duoc collective op tiep theo tu
    # 1 rank -> crash toan bo job. Cac buoc cham (tokenize du lieu lon, luu/push checkpoint
    # tren rank 0, load model lon luc resume, straggler GPU...) rat de vuot qua 10 phut o
    # cluster/mang cham -> tang len 30 phut de chiu duoc ma khong bi kill oan.
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
    logger.info(f"[rank {rank}/{world_size}] Da init NCCL process group "
                f"(timeout={nccl_timeout_minutes} phut, local_rank={local_rank}).")
    return rank, local_rank, world_size, is_distributed, device


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model):
    """PeftModel.save_pretrained() khong ton tai tren DistributedDataParallel -> phai lay
    .module (peft model that su ben trong) ra truoc khi save/push checkpoint."""
    return model.module if hasattr(model, "module") else model


def shard_texts_for_ddp(texts: List[str], lengths: List[int], rank: int, world_size: int):
    """Chia du lieu deu cho cac rank, moi rank 1 shard rieng khong overlap. Cat bot vai
    sample le cuoi cung de tong so sample chia het cho world_size -> MOI RANK CO CUNG SO
    STEP/EPOCH. Bat buoc phai vay: neu cac rank co so step khac nhau, rank it step hon se
    ra khoi vong lap som va ngung goi collective op (backward/all_reduce) trong khi cac
    rank khac van dang cho -> treo (hang) roi NCCL Watchdog timeout, sap ca job."""
    if world_size <= 1:
        return texts, lengths
    n = len(texts)
    n_trunc = (n // world_size) * world_size
    if n_trunc == 0:
        raise RuntimeError(
            f"Chi co {n} sample, khong du de chia deu cho {world_size} GPU "
            f"(can it nhat {world_size} sample)."
        )
    if n_trunc < n and rank == 0:
        logger.warning(f"Bo {n - n_trunc} sample cuoi (trong tong {n}) de chia deu cho "
                        f"{world_size} rank.")
    shard_idx = list(range(n_trunc))[rank::world_size]
    return [texts[i] for i in shard_idx], [lengths[i] for i in shard_idx]


def sync_grads_across_ranks(trainable_params, world_size: int):
    """All-reduce (trung binh) gradient THU CONG, goi DUNG 1 LAN sau khi toan bo cac lan
    backward() cua 1 step (ke ca cac sub-batch sinh ra do dynamic OOM splitting) da chay
    xong. Xem giai thich chi tiet tai noi goi model.no_sync() trong training loop ve ly do
    khong the de DDP tu dong sync nhu binh thuong."""
    for p in trainable_params:
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def reduce_result_across_ranks(result: dict, device, world_size: int) -> dict:
    """Gop loss / so sample cua tat ca rank lai de log/plot phan anh dung so lieu TOAN CUC,
    thay vi chi so lieu cua rieng shard tren rank 0. Cung dung de dong bo quyet dinh
    do_step giua cac rank (xem training loop)."""
    loss_t = torch.tensor(
        [result["lm_loss"], result["lb_loss"], result["total_loss"]],
        device=device, dtype=torch.float32,
    )
    count_t = torch.tensor(
        [float(result["n_processed"]), float(result["n_skipped"])],
        device=device, dtype=torch.float32,
    )
    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    loss_t /= world_size
    return {
        "lm_loss": loss_t[0].item(),
        "lb_loss": loss_t[1].item(),
        "total_loss": loss_t[2].item(),
        "n_processed": int(count_t[0].item()),
        "n_skipped": int(count_t[1].item()),
    }


# ============================================================================================
# Du lieu: doc flores/bible/ntrex -> flatten thanh list cau (moi field ngon ngu = 1 sample)
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
        logger.info(f"{fname}: +{len(sentences) - n_before} cau, tong so record = {len(records)}")
    return sentences


def compute_lengths(tokenizer, texts: Sequence[str], chunk_size: int = 1000) -> List[int]:
    lengths: List[int] = []
    for i in tqdm(range(0, len(texts), chunk_size), desc="Tinh do dai token cho toan bo sample"):
        chunk = texts[i:i + chunk_size]
        enc = tokenizer(chunk, add_special_tokens=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


class SentenceDataset(Dataset):
    """Moi sample la 1 cau (string), duoc tokenize sau trong vong lap training
    de ho tro chia nho batch khi OOM."""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class LengthGroupedBatchSampler(Sampler[List[int]]):
    """Moi epoch: shuffle toan bo index -> sort theo do dai token -> gom batch -> shuffle
    thu tu cac batch. Buoc shuffle truoc khi sort giup cac cau cung nghia (cung id, khac
    ngon ngu) trong flores/bible/ntrex khong bi dinh lien tuc voi nhau trong 1 batch."""

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
# Tu dong tim target module cho LoRA: attention / router / experts trong middle layers
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


def build_lora_target_modules(model, layer_indices: set):
    """Tra ve (targets, kinds): targets la list ten module Linear duoc chon lam LoRA
    target trong khoang layer_indices, kinds la dict ten module -> "router" / "attention" /
    "expert", dung de gan rank rieng cho tung thanh phan (--lora_r_router / 
    --lora_r_attention / --lora_r_experts) qua rank_pattern cua PEFT."""
    targets = []
    kinds = {}
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
            kinds[name] = "router" if is_router else ("expert" if is_expert else "attention")
    return targets, kinds


def register_router_hooks(peft_model, router_names: List[str], cache_list: list):
    """Hook forward tren cac router Linear (da duoc PEFT wrap LoRA) de lay logits phuc vu
    tinh load-balancing loss. Khong detach de gradient van chay ve LoRA cua router."""
    # Luu y: sau khi get_peft_model() wrap, module tai vi tri router khong con la
    # torch.nn.Linear thuan tuy nua ma la peft.tuners.lora.Linear (chi ke thua nn.Module +
    # LoraLayer, KHONG ke thua nn.Linear) -> khong duoc loc theo isinstance(nn.Linear) o day,
    # chi can match dung ten (da duoc build_lora_target_modules xac dinh tu truoc).
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
# MoE loss chuan: LM loss + Load Balancing loss (Switch/Mixtral style)
# ============================================================================================
def compute_load_balancing_loss(router_logits_list: List[torch.Tensor], attention_mask: torch.Tensor,
                                 num_experts: int, top_k: int):
    """attention_mask: [batch, seq_len] (1 = token that, 0 = padding), CUNG kich thuoc batch/seq
    voi input da dua vao model. Phai loai bo vi tri padding truoc khi tinh bat ky thong ke nao,
    vi khong thi:
      - Token padding (pad_token = eos_token, lap lai giong het nhau) se cho ra router logit
        gan nhu giong nhau moi lan -> thoi phong / lam lech tan suat chon expert mot cach he
        thong, khong phan anh dung phan bo cua token that trong cau.
      - So luong token dung de tinh trung binh (N) cung bi dem du them ca padding, lam sai ca
        f_i (ti le token/expert) lan gia tri loss cuoi cung.
    Day la loi tuong tu nhu cach L_LM da loai padding qua ignore_index=-100, chi khac la L_LB
    truoc do khong nhan attention_mask nen khong loc duoc."""
    mask_flat = attention_mask.reshape(-1).bool()  # [tokens], cung thu tu voi logits.reshape(-1, ...)

    # QUAN TRONG: HF (load_balancing_loss_func trong modeling_mixtral.py / modeling_qwen2_moe.py,
    # dung chung cho ca Qwen1.5-MoE) NOI (torch.cat, dim=0) token cua TAT CA cac router layer lai
    # thanh MOT tap thong ke duy nhat, roi MOI tinh f_i (ti le token/expert) va P_i (xac suat
    # trung binh/expert) tren tap da noi do -> chi 1 loss tong the cho toan bo cac layer.
    # Bان dau code o day tinh rieng tung layer (f_i, P_i, loss theo layer) roi lay .mean() cac
    # loss lai -> VE TOAN HOC KHONG TUONG DUONG voi cach cua HF, vi:
    #   mean_layer( sum_i f_i^(layer) * P_i^(layer) )  !=  sum_i mean_layer(f_i^(layer)) * mean_layer(P_i^(layer))
    # (hai ve chi bang nhau neu khong co hiep phuong sai giua cac layer, noi chung la sai).
    # -> sua lai: concat truoc, tinh thong ke + loss SAU, dung 1 lan cho toan bo cac router
    # trong router_logits_list (van chi gom cac layer da duoc hook, tuc la cac layer nam trong
    # khoang layer duoc gan LoRA — day la lua chon co chu dich cua script, khac voi mac dinh cua
    # Qwen la tinh tren TOAN BO router layer cua model).
    valid_logits = []
    for logits in router_logits_list:
        logits = logits.reshape(-1, logits.shape[-1])  # [tokens, num_experts]
        if logits.shape[0] == mask_flat.shape[0]:
            logits = logits[mask_flat]  # bo cac vi tri padding truoc khi tinh thong ke
        else:
            # Kien truc MoE nay flatten/reshape token theo thu tu khac gia dinh o tren (batch
            # truoc, seq sau) -> khong the index an toan theo mask_flat, bo qua loc padding cho
            # lan nay thay vi index sai vi tri (van con tot hon crash, nhung se kem chinh xac).
            logger.warning(
                "compute_load_balancing_loss: kich thuoc router logits "
                f"({logits.shape[0]}) khong khop attention_mask ({mask_flat.shape[0]}) -> "
                "bo qua loc padding cho lan tinh nay, kiem tra lai thu tu flatten token cua "
                "kien truc MoE nay neu thay canh bao lap lai nhieu lan."
            )
        if logits.shape[0] > 0:
            valid_logits.append(logits)

    if not valid_logits:
        return torch.tensor(0.0, device=attention_mask.device)

    concatenated_logits = torch.cat(valid_logits, dim=0)  # [tong_token_qua_cac_layer, num_experts]
    routing_weights = F.softmax(concatenated_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # [tokens, top_k]
    expert_mask = F.one_hot(selected_experts, num_experts).float()  # [tokens, top_k, num_experts]
    # LUU Y: .mean(dim=0) da chia trung binh theo so token roi (cho ra f_i dung chuan,
    # trong khoang [0,1]) -> KHONG duoc chia them cho logits.shape[0] mot lan nua (bug
    # cu chia 2 lan lam L_LB nho gia tao ~N lan, N = so token trong sub-batch dang tinh).
    tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # [num_experts], = f_i
    avg_prob_per_expert = routing_weights.mean(dim=0)  # [num_experts]
    return num_experts * torch.sum(tokens_per_expert * avg_prob_per_expert)


def forward_backward_one_subbatch(sub_texts, tokenizer, model, max_length, device,
                                   router_logits_cache, num_experts, top_k, lb_loss_coef,
                                   loss_weight):
    """Tokenize + forward + backward cho 1 sub-batch (co the la toan bo batch hoac 1 mieng sau
    khi chia doi vi OOM). Tra ve (lm_loss_val, lb_loss_val, total_loss_val, n_samples)."""
    enc = tokenizer(sub_texts, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
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
    (total_loss * loss_weight).backward()

    return lm_loss.item(), lb_loss.item(), total_loss.item(), len(sub_texts)


def run_batch_with_dynamic_oom_handling(batch_texts, tokenizer, model, max_length, device,
                                         router_logits_cache, num_experts, top_k, lb_loss_coef,
                                         min_batch_size):
    """Chay 1 batch (list text). Neu OOM: clear memory, chia doi, de quy. Neu OOM ca khi
    size = 1 (hoac == min_batch_size) thi skip sample do. Luon quay ve batch_size goc cho
    batch tiep theo (khong giu trang thai giua cac batch).

    QUAN TRONG ve vong doi exception: KHONG duoc goi clear_memory()/de quy retry ngay
    ben trong khoi `except ... as e:`. Trong luc con o trong khoi except do, `e.__traceback__`
    van giu tham chieu toi toan bo frame cua forward_backward_one_subbatch (input_ids, outputs,
    logits, ...) cua LAN VUA OOM -> cac tensor GPU do van "reachable" -> gc.collect()/
    torch.cuda.empty_cache() khong giai phong duoc gi ca, va lan retry (voi batch nho hon)
    lai chay trong khi bo nho cua lan fail truoc van bi ghim, cong don qua tung cap chia doi.
    Vi vay ta tach rieng buoc "thu chay 1 lan" (_attempt) khoi buoc "don dep + de quy retry"
    (_run): _run chi don dep/retry SAU KHI _attempt() da return, tuc la sau khi khoi except
    da thoat va Python da tu dong `del e` (giai phong that su traceback + frame)."""
    original_size = len(batch_texts)
    agg = {"lm_loss": 0.0, "lb_loss": 0.0, "total_loss": 0.0, "n_ok": 0, "n_skipped": 0}

    def _attempt(sub_texts) -> bool:
        """Chi thu forward+backward DUNG 1 LAN. Tra ve True neu thanh cong, False neu OOM.
        Khong lam gi khac trong except (khong clear_memory, khong retry) de dam bao khoi
        except ket thuc ngay, Python tu xoa `e` va giai phong that su frame/tensor bi OOM."""
        try:
            lm, lb, tot, n = forward_backward_one_subbatch(
                sub_texts, tokenizer, model, max_length, device,
                router_logits_cache, num_experts, top_k, lb_loss_coef,
                loss_weight=len(sub_texts) / max(original_size, 1),
            )
        except RuntimeError as e:
            if not is_oom_error(e):
                raise
            return False
        agg["lm_loss"] += lm * n
        agg["lb_loss"] += lb * n
        agg["total_loss"] += tot * n
        agg["n_ok"] += n
        return True

    def _run(sub_texts):
        if _attempt(sub_texts):
            return

        # Toi day khoi except cua _attempt() da thoat hoan toan -> `e`/traceback da bi
        # Python xoa -> frame cua forward_backward_one_subbatch (voi input_ids, outputs,
        # logits, shift_logits...) that su khong con ai tham chieu nua.
        #
        # router_logits_cache: hook forward luu logits KHONG detach (de giu gradient cho
        # LoRA cua router) -> neu lan OOM vua roi da kip chay qua vai router truoc khi fail,
        # cache van con om nguyen do thi autograd cua lan do. Binh thuong cache chi duoc
        # .clear() o DAU lan forward_backward_one_subbatch ke tiep -> qua muon, phai clear
        # ngay tai day truoc khi goi clear_memory().
        router_logits_cache.clear()

        # KHONG goi model.zero_grad() o day: backward() cua cac sub-batch anh em (da chay
        # thanh cong truoc do trong cung batch goc) da tich luy gradient hop le vao .grad
        # theo co che gradient-accumulation (loss_weight = len(sub)/original_size). Goi
        # zero_grad() se xoa sach ca phan gradient hop le do moi khi co 1 sub-batch OOM,
        # lam sai lech gradient cua ca buoc optimizer.step() ke tiep.
        clear_memory()

        if len(sub_texts) <= max(min_batch_size, 1):
            logger.warning(f"OOM ngay ca voi sub-batch size={len(sub_texts)} -> skip sample nay.")
            agg["n_skipped"] += len(sub_texts)
            return
        mid = len(sub_texts) // 2
        logger.warning(f"OOM voi sub-batch size={len(sub_texts)} -> chia doi thanh {mid} + {len(sub_texts) - mid}.")
        _run(sub_texts[:mid])
        _run(sub_texts[mid:])

    _run(batch_texts)
    n = max(agg["n_ok"], 1)
    return {
        "lm_loss": agg["lm_loss"] / n,
        "lb_loss": agg["lb_loss"] / n,
        "total_loss": agg["total_loss"] / n,
        "n_processed": agg["n_ok"],
        "n_skipped": agg["n_skipped"],
    }


# ============================================================================================
# Checkpoint / resume
# ============================================================================================
def save_checkpoint(output_dir, model, optimizer, scheduler, epoch, step_in_epoch, global_step):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)  # PeftModel: chi luu adapter LoRA
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
        # fallback: tim checkpoint-* co global_step lon nhat
        candidates = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if candidates:
            candidates.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
            return candidates[-1]
        return None
    return resume_arg if os.path.isdir(resume_arg) else None


# ============================================================================================
# Diagnostics: jsonl + plot
# ============================================================================================
def log_step_to_jsonl(jsonl_path, global_step, epoch, result):
    rec = {
        "step": global_step,
        "epoch": epoch,
        "lm_loss": result["lm_loss"],
        "lb_loss": result["lb_loss"],
        "total_loss": result["total_loss"],
        "n_processed": result["n_processed"],
        "n_skipped": result["n_skipped"],
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
    plt.plot(steps, lm, label="L_LM")
    plt.plot(steps, lb, label="L_LB")
    plt.plot(steps, total, label="L_Total")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Qwen1.5-MoE-A2.7B LoRA finetuning loss (tho, tung step)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_losses_smoothed(jsonl_path, out_png, window: int):
    """Ve duong loss da lam min: gom `window` step LIEN TIEP (theo thu tu ghi log, tuc la
    theo global_step tang dan) thanh 1 "bin", lay TRUNG BINH CONG lm/lb/total trong bin do
    lam 1 diem tren do thi. Khac voi plot_losses() (ve tho tung step, rat messy vi L_LB/L_LM
    dao qua lai theo tung batch nho), o day so diem giam di ~window lan nen xu huong tang/giam
    that su cua loss de nhin hon nhieu."""
    if not os.path.exists(jsonl_path) or window <= 1:
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

    def bin_mean(values):
        # Diem cuoi cung cua moi bin (step lon nhat trong bin) duoc dung lam nhan truc x, de
        # truc x van la "global_step" chu khong phai "thu tu bin" (de so sanh voi save_steps,
        # log_every de dang hon).
        xs, ys = [], []
        for i in range(0, len(values), window):
            chunk = values[i:i + window]
            xs.append(steps[i + len(chunk) - 1])
            ys.append(sum(chunk) / len(chunk))
        return xs, ys

    x_lm, y_lm = bin_mean(lm)
    x_lb, y_lb = bin_mean(lb)
    x_tot, y_tot = bin_mean(total)

    plt.figure(figsize=(10, 6))
    plt.plot(x_lm, y_lm, label="L_LM (mean)", marker="o", markersize=3)
    plt.plot(x_lb, y_lb, label="L_LB (mean)", marker="o", markersize=3)
    plt.plot(x_tot, y_tot, label="L_Total (mean)", marker="o", markersize=3)
    plt.xlabel(f"Training step (moi diem = trung binh cong cua {window} step lien tiep)")
    plt.ylabel("Loss (trung binh)")
    plt.title(f"Qwen1.5-MoE-A2.7B LoRA finetuning loss - trung binh moi {window} step")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_all(jsonl_path, plot_path, plot_path_smoothed, smooth_window):
    """Goi ca 2 ham ve: loss_curve.png (tho, tung step) va loss_curve_smoothed.png (trung
    binh cong moi smooth_window step) trong 1 lan, dung o moi diem trong training loop can
    cap nhat plot (log_every, save_steps, cuoi training, KeyboardInterrupt) — ca 2 file nam
    chung trong diagnostics_dir nen push_to_hub() (upload_folder toan bo thu muc) se tu dong
    day len hub cung voi checkpoint, khong can sua push_to_hub."""
    plot_losses(jsonl_path, plot_path)
    plot_losses_smoothed(jsonl_path, plot_path_smoothed, smooth_window)


# ============================================================================================
# Hugging Face Hub push
# ============================================================================================
def build_model_card(args, num_experts, top_k, layer_start, layer_end, num_layers, data_files) -> str:
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

# Qwen1.5-MoE-A2.7B-Finetuning

Day la LoRA adapter finetune tu [`{args.model_name_or_path}`]\
(https://huggingface.co/{args.model_name_or_path}), mot mo hinh Mixture-of-Experts.

## Cau hinh LoRA
- Layer duoc finetune: `[{layer_start}, {layer_end})` trong tong so `{num_layers}` layer
  (tuong ung khoang 1L/3 -> 2L/3).
- Module duoc gan LoRA: **attention**, **router**, **experts** trong khoang layer tren, moi
  thanh phan mot rank rieng qua `rank_pattern` cua PEFT:
  - attention: r = {args.lora_r_attention}
  - router: r = {args.lora_r_router}
  - experts: r = {args.lora_r_experts}
- alpha = {args.lora_alpha}, dropout = {args.lora_dropout}

## Loss
Loss MoE tieu chuan:

`L_total = L_LM + lb_loss_coef * L_LB`

- `L_LM`: cross-entropy chuan tren token tiep theo.
- `L_LB`: load balancing loss chuan cua MoE (Switch/Mixtral style), tinh tren cac router
  nam trong khoang layer duoc finetune.
- `lb_loss_coef` = {args.lb_loss_coef}
- `num_experts` = {num_experts}, `top_k` = {top_k}

## Du lieu
Alignment data duoc chon qua `--alignment_data` (`{" ".join(args.alignment_data)}`), gom sample
tu cac file: `{"`, `".join(data_files)}` (moi field ngon ngu trong 1 record duoc coi la 1
sample), shuffle va sort theo do dai token truoc khi gom batch.

## Diagnostics
Xem `diagnostics/loss_log.jsonl` (log theo tung step), `diagnostics/loss_curve.png` (bieu do
tho L_LM / L_LB / L_Total theo tung step) va `diagnostics/loss_curve_smoothed.png` (cung 3
duong loss nhung da lay trung binh cong moi `{args.smooth_window}` step lien tiep — de doc
xu huong hon vi bieu do tho rat messy o cap do tung step).
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

    rank, local_rank, world_size, is_distributed, ddp_device = setup_distributed(
        args.nccl_timeout_minutes
    )
    if is_distributed:
        if args.device_map:
            raise ValueError(
                "Khong dung --device_map cung luc voi distributed training (torchrun). "
                "DDP: moi process giu 1 ban sao model day du tren 1 GPU rieng (qua "
                "--nproc_per_node). device_map (model-parallel, 1 process nhieu GPU) la co "
                "che khac, khong tuong thich voi DDP."
            )
        args.device = ddp_device
        if rank != 0:
            logger.setLevel(logging.WARNING)  # tranh log trung lap tu tat ca rank

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    diagnostics_dir = args.diagnostics_dir or os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = os.path.join(diagnostics_dir, "loss_log.jsonl")
    plot_path = os.path.join(diagnostics_dir, "loss_curve.png")
    plot_path_smoothed = os.path.join(diagnostics_dir, "loss_curve_smoothed.png")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # ---------------------------------------------------------------------------------- model
    logger.info(f"Dang load tokenizer va model tu {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                               trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs = dict(torch_dtype=dtype, trust_remote_code=args.trust_remote_code)
    if args.device_map:
        model_kwargs["device_map"] = args.device_map
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if not args.device_map:
        base_model.to(args.device)

    num_layers = get_num_layers(base_model.config)
    layer_start = int(num_layers * args.lora_layer_start_ratio)
    layer_end = int(num_layers * args.lora_layer_end_ratio)
    layer_indices = set(range(layer_start, layer_end))
    logger.info(f"Tong so layer = {num_layers}. Ap dung LoRA cho layer [{layer_start}, {layer_end}).")

    target_modules, module_kinds = build_lora_target_modules(base_model, layer_indices)
    if not target_modules:
        raise RuntimeError(
            "Khong tim thay module (attention/router/experts) nao trong khoang layer da chon. "
            "Kien truc model co the dat ten khac quy uoc — kiem tra lai regex trong build_lora_target_modules()."
        )
    router_target_names = [n for n in target_modules if module_kinds[n] == "router"]
    attn_target_names = [n for n in target_modules if module_kinds[n] == "attention"]
    expert_target_names = [n for n in target_modules if module_kinds[n] == "expert"]
    logger.info(f"Tim thay {len(target_modules)} target module cho LoRA: "
                f"{len(attn_target_names)} attention (r={args.lora_r_attention}), "
                f"{len(router_target_names)} router (r={args.lora_r_router}), "
                f"{len(expert_target_names)} experts (r={args.lora_r_experts}). "
                f"Vi du: {target_modules[:8]}")

    num_experts, top_k = infer_moe_dims(base_model.config, args)

    lb_loss_coef = args.lb_loss_coef
    if lb_loss_coef is None:
        lb_loss_coef = float(getattr(base_model.config, "router_aux_loss_coef", 0.001))
    logger.info(f"lb_loss_coef (trong so load-balancing, nhu finetune binh thuong) = {lb_loss_coef}")

    # ---------------------------------------------------------------------------- resume / LoRA
    resume_dir = find_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_dir:
        logger.info(f"Resume LoRA adapter tu checkpoint: {resume_dir}")
        model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
    else:
        # rank_pattern cua PEFT: dict {regex ten module -> rank rieng}, khac voi rank
        # mac dinh `r`. Matching duoc peft thuc hien dang "(.*\\.)?(<key>)$" (khop HAU TO
        # cua ten module day du) -> escape ten module bang re.escape() de chi khop CHINH
        # XAC module do (tranh dau "." trong ten bi hieu nham thanh wildcard cua regex).
        # attention KHONG can dua vao rank_pattern vi da dung r mac dinh (lora_r_attention).
        rank_pattern = {}
        rank_pattern.update({re.escape(n): args.lora_r_router for n in router_target_names})
        rank_pattern.update({re.escape(n): args.lora_r_experts for n in expert_target_names})
        lora_config = LoraConfig(
            r=args.lora_r_attention,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            rank_pattern=rank_pattern,
        )
        model = get_peft_model(base_model, lora_config)
    if is_main_process(rank):
        model.print_trainable_parameters()
    if not args.device_map:
        model.to(args.device)

    router_logits_cache: list = []
    hooks = register_router_hooks(model, router_target_names, router_logits_cache)

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # MoE top-k routing: khong phai expert nao cung duoc chon o moi step, va LoRA
            # chi gan tren 1 khoang layer -> luon co tham so KHONG nhan gradient o mot so
            # step -> BAT BUOC True, neu khong DDP se bao loi "Expected to have finished
            # reduction..." ngay khi gap batch dau tien co expert/module khong duoc dung toi.
            find_unused_parameters=True,
        )

    # ------------------------------------------------------------------------------------ data
    # --alignment_data (flores/ntrex/bible, co the ket hop) -> danh sach file JSON thuc te.
    # dict.fromkeys(...) de loai trung neu nguoi dung lo nhap trung ten (van giu thu tu).
    data_files = [ALIGNMENT_DATA_FILENAMES[name] for name in dict.fromkeys(args.alignment_data)]
    logger.info(f"Dang doc du lieu tu {args.data_dir}, alignment_data={args.alignment_data} "
                f"(file tuong ung: {data_files}) ...")
    texts = load_all_sentences(args.data_dir, data_files)
    if args.max_samples:
        random.Random(args.seed).shuffle(texts)
        texts = texts[: args.max_samples]
    logger.info(f"Tong so sample (cau) sau khi gom {len(data_files)} bo du lieu da chon: {len(texts)}")
    if len(texts) == 0:
        raise RuntimeError("Khong doc duoc sample nao — kiem tra lai --data_dir / --alignment_data.")

    lengths = compute_lengths(tokenizer, texts)

    # Distributed: moi rank chi train tren 1 shard rieng, khong overlap. --batch_size la
    # batch size CHO MOI GPU (giong per_device_train_batch_size cua HF Trainer) -> global
    # batch size thuc te = batch_size * world_size.
    texts, lengths = shard_texts_for_ddp(texts, lengths, rank, world_size)
    if is_distributed:
        logger.info(f"[rank {rank}/{world_size}] Shard cua rank nay: {len(texts)} sample.")

    dataset = SentenceDataset(texts)
    batch_sampler = LengthGroupedBatchSampler(lengths, batch_size=args.batch_size, seed=args.seed)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=lambda b: b)

    # ------------------------------------------------------------------------------- optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                weight_decay=args.weight_decay, foreach=True)  # hoặc fused=True nếu CUDA hỗ trợ

    steps_per_epoch = len(batch_sampler)
    total_steps = steps_per_epoch * args.num_train_epochs
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
            for group in optimizer.param_groups:
                group["foreach"] = True
            if state.get("scheduler"):
                scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"]
            start_step_in_epoch = state["step_in_epoch"] + 1
            global_step = state["global_step"]
            torch.set_rng_state(state["torch_rng_state"])
            random.setstate(state["python_rng_state"])
            logger.info(f"Da resume: epoch={start_epoch}, step_in_epoch={start_step_in_epoch}, "
                        f"global_step={global_step}")
            if start_step_in_epoch >= steps_per_epoch:
                start_epoch += 1
                start_step_in_epoch = 0

    readme_text = build_model_card(args, num_experts, top_k, layer_start, layer_end, num_layers, data_files)

    # ------------------------------------------------------------------------------ training loop
    model_device = next(model.parameters()).device
    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            batch_sampler.set_epoch(epoch)
            step_offset = start_step_in_epoch if epoch == start_epoch else 0

            no_sync_ctx = model.no_sync if is_distributed else nullcontext

            pbar = tqdm(
                enumerate(dataloader),
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
                disable=not is_main_process(rank),
            )
            for step_in_epoch, batch_texts in pbar:
                if step_in_epoch < step_offset:
                    # dang resume: bo qua nhanh cac batch da xu ly o lan chay truoc
                    continue

                model.train()
                # set_to_none=False khi distributed: dam bao MOI param luon co san tensor
                # .grad (khong bi None), ke ca khi rank nay bi skip toan bo sample do OOM o
                # step nay -> can thiet de sync_grads_across_ranks() ben duoi goi all_reduce
                # dong bo duoc giua cac rank (all_reduce doi hoi TAT CA rank cung tham gia
                # voi tensor ton tai, khong the "vang mat").
                optimizer.zero_grad(set_to_none=not is_distributed)

                # Toan bo cac lan backward() cua step nay (ke ca cac sub-batch do dynamic
                # OOM splitting sinh ra ben trong ham duoi) duoc boc trong no_sync(): tat co
                # che DDP tu dong all-reduce gradient sau MOI lan backward(). Neu khong tat,
                # 1 step goi backward() nhieu lan se khien DDP all-reduce nhieu lan/sai nhip
                # tren cung 1 tap tham so -> loi kinh dien "Expected to mark a variable ready
                # only once" (cang de gap hon khi find_unused_parameters=True). Thay vao do,
                # ta tu all_reduce THU CONG dung 1 LAN (sync_grads_across_ranks) ngay sau khi
                # toan bo cac lan backward can thiet cua step da chay xong, truoc khi goi
                # optimizer.step().
                with no_sync_ctx():
                    result = run_batch_with_dynamic_oom_handling(
                        batch_texts=batch_texts,
                        tokenizer=tokenizer,
                        model=model,
                        max_length=args.max_length,
                        device=model_device,
                        router_logits_cache=router_logits_cache,
                        num_experts=num_experts,
                        top_k=top_k,
                        lb_loss_coef=lb_loss_coef,
                        min_batch_size=args.min_batch_size,
                    )

                if is_distributed:
                    # Gop n_processed/loss cua TAT CA rank: vua de quyet dinh do_step DONG
                    # BO giua cac rank (tranh truong hop rank nay goi optimizer.step() con
                    # rank kia thi khong — se lam tham so cac rank lech nhau vinh vien vi DDP
                    # gia dinh tham so luon giong het nhau giua cac rank), vua de log dung so
                    # lieu toan cuc thay vi chi shard rieng cua rank 0.
                    result = reduce_result_across_ranks(result, model_device, world_size)
                do_step = result["n_processed"] > 0

                if do_step:
                    if is_distributed:
                        sync_grads_across_ranks(trainable_params, world_size)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
                    optimizer.step()
                scheduler.step()
                global_step += 1

                pbar.set_postfix({
                    "L_LM": f"{result['lm_loss']:.4f}",
                    "L_LB": f"{result['lb_loss']:.4f}",
                    "L_Total": f"{result['total_loss']:.4f}",
                    "skipped": result["n_skipped"],
                })

                if is_main_process(rank):
                    if result["n_processed"] > 0:
                        log_step_to_jsonl(jsonl_path, global_step, epoch, result)

                    if global_step % args.log_every == 0:
                        plot_all(jsonl_path, plot_path, plot_path_smoothed, args.smooth_window)

                    if global_step % args.save_steps == 0:
                        ckpt_dir = save_checkpoint(args.output_dir, unwrap_model(model), optimizer,
                                                    scheduler, epoch, step_in_epoch, global_step)
                        plot_all(jsonl_path, plot_path, plot_path_smoothed, args.smooth_window)
                        logger.info(f"Da luu checkpoint local: {ckpt_dir}")
                        if args.push_to_hub:
                            push_to_hub(ckpt_dir, diagnostics_dir, args.hub_model_id,
                                        args.hub_private, readme_text)
                            logger.info(f"Da push checkpoint len hub: {args.hub_model_id}")

                if is_distributed and global_step % args.save_steps == 0:
                    # Cac rank khac cho rank 0 ghi xong checkpoint/push len hub roi moi vao
                    # step tiep theo, tranh lech nhip qua nhieu giua cac rank (rank 0 lam
                    # I/O/network cham) -> giam nguy co NCCL Watchdog timeout o cac collective
                    # op (all_reduce gradient) cua step ke tiep.
                    dist.barrier()

            start_step_in_epoch = 0  # tu epoch tiep theo tro di, khong can offset resume nua

        # checkpoint cuoi cung sau khi hoan thanh training
        if is_main_process(rank):
            final_ckpt = save_checkpoint(args.output_dir, unwrap_model(model), optimizer, scheduler,
                                          args.num_train_epochs - 1, steps_per_epoch - 1, global_step)
            plot_all(jsonl_path, plot_path, plot_path_smoothed, args.smooth_window)
            if args.push_to_hub:
                push_to_hub(final_ckpt, diagnostics_dir, args.hub_model_id, args.hub_private, readme_text)
            logger.info("Training hoan tat.")
        if is_distributed:
            dist.barrier()

    except KeyboardInterrupt:
        logger.warning("Nhan KeyboardInterrupt — luu checkpoint khan cap truoc khi thoat ...")
        if is_main_process(rank):
            save_checkpoint(args.output_dir, unwrap_model(model), optimizer, scheduler,
                             epoch, step_in_epoch, global_step)
            plot_all(jsonl_path, plot_path, plot_path_smoothed, args.smooth_window)
        raise
    finally:
        for h in hooks:
            h.remove()
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()