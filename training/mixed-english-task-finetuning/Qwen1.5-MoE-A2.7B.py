"""

How to run:

torchrun --standalone --nproc_per_node=8 Qwen1.5-MoE-A2.7B-Finetuning.py \
  --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
  --snli_file data/english_task/snli/train.json \
  --squad_file data/english_task/squad/train.json \
  --mmlu_file data/english_task/mmlu/auxiliary_train.json \
  --max_tokens_per_batch 24576 \
  --save_steps 200 \
  --push_to_hub

Fine-tuning LoRA cho mo hinh Mixture-of-Experts Qwen/Qwen1.5-MoE-A2.7B tren MIX 3 TASK:
SNLI (NLI 3 lop) + SQuAD (extractive QA) + MMLU (trac nghiem), theo huong "english-task-only"
(khong dung du lieu alignment). Day la ban GOP cua 3 file Qwen1.5-MoE-A2.7B-SNLI.py /
Qwen1.5-MoE-A2.7B-SQuAD.py / Qwen1.5-MoE-A2.7B-MMLU.py thanh 1 file duy nhat, 1 lan train, 1 repo hub.

Loss = L_LM (cross-entropy chuan, CHI tinh tren token cua phan dap an "<answer><eos>", prompt bi mask)
       + lb_loss_coef * L_LB (load balancing loss chuan cua MoE, tinh tren cac router nam
       trong khoang layer duoc gan LoRA).

GIU NGUYEN so voi 3 file goc (cung 1 setting cho ca 3 task):
  1. LoRA chi ap dung tren middle layers [L/3, 2L/3), chi len attention / router / experts,
     moi thanh phan mot rank rieng (router r=4, attention r=16, experts r=16), alpha=32, dropout=0.05.
  2. Prompt format, mask loss, windowing context cua SQuAD, loc sample qua dai, lb_loss_coef=0.01,
     lr 2e-4, 3 epoch, warmup 3%, grad clip 1.0 — deu nhu cac file goc.
  3. Checkpointing + resume (--resume_from_checkpoint auto), luu moi --save_steps step, push hub.
  4. Dynamic OOM handling (chia nho batch khi OOM, skip sample neu OOM ca khi size = 1).
  5. Prompt cua tung task GIONG HET file goc nen eval zero-shot tren XNLI/XQuAD/MMMLU van dung format cu:

        SNLI : Premise: ...\nHypothesis: ...\nQuestion: What is the relationship between the premise
               and the hypothesis? Choose one: entailment, neutral, or contradiction.\nAnswer: <label><eos>
        SQuAD: Context: ...\nQuestion: ...\nAnswer: <answer><eos>
        MMLU : The following are multiple choice questions (with answers) about <subject>.\n\n<question>\n
               A. ...\nB. ...\nC. ...\nD. ...\nAnswer: <letter><eos>

TOI UU MOI so voi 3 file goc (cho multi-GPU + batching khi MIX 3 task):
  A. TOKEN-BUDGET BATCHING thay cho batch_size co dinh. 3 task co do dai rat khac nhau (SNLI ~60 token,
     MMLU ~120, SQuAD ~200+), nen "batch_size=128 sample" khong con hop ly: batch SNLI qua nho (GPU doi),
     batch SQuAD qua to (de OOM). Moi batch gio bi gioi han boi --max_tokens_per_batch (so token SAU khi
     pad) va --max_batch_size (tran so sample), nen luong tinh toan moi batch ~ bang nhau.
  B. MIX DEU GIUA CAC RANK: batch duoc gom theo do dai (it padding) roi SHUFFLE, moi step moi rank nhan 1
     batch NGAU NHIEN -> 1 optimizer step tren 8 GPU tron ca 3 task, thay vi step chi toan SNLI hay toan
     SQuAD. Vi moi batch co ~cung so token nen cac rank van can bang tai (khong co straggler).
  C. CHI TINH LM-HEAD TREN VI TRI NHAN: loss chi tren 1-10 token cuoi moi sample, nen thay vi tinh logits
     [batch, seq, 151936] (hang chuc GB), script chay decoder lay hidden state roi chi dua cac vi tri co
     nhan qua lm_head. Ket qua loss/gradient BANG HET, nhung bo nho logits giam ~50 lan.
  D. CHUAN HOA LOSS THEO TONG SO TOKEN NHAN TOAN CUC cua moi step (nhu fairseq): vi batch co so sample khac
     nhau, moi token nhan co trong so bang nhau bat ke nam o rank/batch/chunk nao. Gradient duoc SUM
     (khong chia trung binh) qua 1 lan all-reduce tren buffer phang.
  E. BO DDP WRAPPER: code goc luon chay trong no_sync() va tu all-reduce gradient bang tay, nen DDP chi con
     la overhead (find_unused_parameters=True duyet do thi moi forward) va la nguon cua loi deadlock
     broadcast_buffers. Gio khong con DDP: chi broadcast LoRA tu rank 0 luc khoi tao + 1 all-reduce/step.
     Vi khong co collective nao trong forward/backward nen OOM-split tung rank khong the gay treo NCCL.
  F. OOM HANDLING DUNG HON: code goc khi OOM giua backward co the de lai gradient DO DANG cua lan thu
     that bai roi cong don them lan retry. Ban nay khi OOM thi zero gradient + stats va chay lai CA BATCH
     voi chunk nho hon (chunk = 1 sample ma van OOM thi bo sample do), nen gradient luon chinh xac.
  G. TOKENIZE 1 LAN + CACHE: tokenize full_text/prompt dung 1 lan luc chuan bi (khong tokenize lai o moi
     step), rank 0 lam va luu cache pickle trong <output_dir>/cache, cac rank con lai doc cache (khong
     tokenize trung lap 8 lan, resume cung khong phai tokenize lai). Pad ve boi so cua 8 cho tensor core.
  H. PUSH HUB BAT DONG BO (thread nen) + KHONG upload trainer_state.pt (optimizer state) len hub, nen
     GPU khong phai doi trong luc upload.
  I. DIAGNOSTICS THEO TUNG TASK: loss/accuracy rieng cho snli/squad/mmlu moi step (jsonl + plot), ktok/s.

Dinh nghia "label_acc": ty le token DUNG (teacher forcing) tren toan bo token cua "<answer><eos>"
(SNLI file goc chi do token dau tien cua nhan; o day thong nhat do tren toan bo nhan cho ca 3 task).

Luu y ve ty trong giua cac task: mac dinh la TRON NGUYEN 3 TAP (concat) nen task nao nhieu token hon se
chiem nhieu trong so hon (SNLI full train ~550k sample vs SQuAD ~88k vs MMLU auxiliary ~100k). Neu muon
can bang hon, dung --max_samples_per_task N de gioi han so sample moi task.

Resume:
    torchrun ... Qwen1.5-MoE-A2.7B-Finetuning.py --resume_from_checkpoint auto
(giu nguyen world_size va cac tham so batching/seed/--tasks khi resume, vi ke hoach batch moi epoch duoc
sinh co dinh tu seed + epoch.)

Load lai adapter de eval:
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, "ducanhdinh/Qwen1.5-MoE-A2.7B-Finetuning")
"""

import argparse
import datetime
import gc
import glob
import hashlib
import json
import logging
import os
import pickle
import random
import re
import string
import threading
import time
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
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
logger = logging.getLogger("qwen15_moe_mix_finetune")


# Cac ten bien moi truong pho bien cho HF token, thu theo thu tu nay
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Danh sach task ho tro (thu tu o day = task_id mac dinh).
ALL_TASKS = ("snli", "squad", "mmlu")

# Tang so nay neu doi prompt/cach build du lieu de vo hieu hoa cache cu.
DATA_FORMAT_VERSION = 1

# Nhan SNLI chuan: 0 = entailment, 1 = neutral, 2 = contradiction.
# SNLI goc con co the co label = -1 (khong dong thuan giua annotator) -> bi loc bo khi doc du lieu.
LABEL_TO_WORD = {0: "entailment", 1: "neutral", 2: "contradiction"}

# Chu cai dung de danh so lua chon MMLU: A, B, C, ..., toi da 26 lua chon (MMLU thuong chi co 4).
CHOICE_LETTERS = string.ascii_uppercase

# Tach context SQuAD thanh cac "tu" (chuoi ky tu khac khoang trang) de lam windowing.
WORD_SPAN_PATTERN = re.compile(r"\S+")

# Layout vector thong ke moi step (all-reduce 1 lan): [ce_sum, n_label_tok, n_correct, lb_sum,
# n_samples, n_skipped, n_padded_tok] + [task_ce_sum]*T + [task_tok]*T + [task_correct]*T
_N_BASE_STATS = 7


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
    p = argparse.ArgumentParser(
        description="LoRA finetuning (mix SNLI + SQuAD + MMLU) cho MoE Qwen1.5-MoE-A2.7B"
    )

    # Model / data / output
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--tasks", type=str, default="snli,squad,mmlu",
                    help="Danh sach task de mix, cach nhau boi dau phay. Tap con cua snli,squad,mmlu.")
    p.add_argument("--snli_file", type=str, default="data/english_task/snli/train.json",
                    help="JSON SNLI (list cac object {premise, hypothesis, label}).")
    p.add_argument("--squad_file", type=str, default="data/english_task/squad/train.json",
                    help="JSON SQuAD (list cac object {id, title, context, question, answers}).")
    p.add_argument("--mmlu_file", type=str, default="data/english_task/mmlu/auxiliary_train.json",
                    help="JSON MMLU (list cac object {question, choices:[...], answer:int, subject}).")
    p.add_argument("--output_dir", type=str,
                    default="training/english-task-only/checkpoints/Qwen1.5-MoE-A2.7B-Finetuning")
    p.add_argument("--cache_dir", type=str, default=None,
                    help="Thu muc luu cache du lieu da tokenize. None = <output_dir>/cache")
    p.add_argument("--rebuild_cache", action="store_true",
                    help="Bo qua cache va build + tokenize lai du lieu tu dau.")
    p.add_argument("--max_samples_per_task", type=int, default=None,
                    help="Gioi han so sample MOI task (de can bang ty trong 3 task / debug). "
                         "None = dung het du lieu.")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Gioi han TONG so sample sau khi mix (debug/smoke test), None = dung het.")

    # Hugging Face Hub
    p.add_argument("--push_to_hub", action="store_true", default=True)
    p.add_argument("--no_push_to_hub", dest="push_to_hub", action="store_false")
    p.add_argument("--hub_model_id", type=str, default="ducanhdinh/Qwen1.5-MoE-A2.7B-Finetuning",
                    help="Doi lai namespace/username HF cua ban neu khac 'ducanhdinh'.")
    p.add_argument("--hub_private", action="store_true")
    p.add_argument("--env_file", type=str, default=".env",
                    help="Duong dan file .env chua HF_TOKEN, tu dong nap bang python-dotenv")
    p.add_argument("--hf_token", type=str, default=None,
                    help="Override HF token thu cong, uu tien cao hon .env/bien moi truong")

    # Training schedule (giong cac file goc)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_tokens_per_batch", type=int, default=24576,
                    help="Tran so token SAU KHI PAD cua 1 batch tren MOI rank (= so sample x do dai "
                         "sample dai nhat trong batch). 24576 ~ tuong duong batch goc (SNLI 512x~60, "
                         "MMLU 128x~200, SQuAD 64x~300). Tang len neu con du VRAM.")
    p.add_argument("--max_batch_size", type=int, default=512,
                    help="Tran so sample cua 1 batch (de batch SNLI rat ngan khong phinh qua to).")
    p.add_argument("--min_batch_size", type=int, default=1,
                    help="Khi OOM, chunk <= gia tri nay ma van OOM thi bo qua chunk do.")
    p.add_argument("--pool_size", type=int, default=20000,
                    help="Moi epoch: shuffle toan bo sample, cat thanh pool N sample, sort theo do dai "
                         "TRONG pool de gom batch it padding. Pool lon = it padding hon, nho = ngau "
                         "nhien hon giua cac epoch.")
    p.add_argument("--max_length", type=int, default=512,
                    help="Do dai toi da (token) cua 1 sample. SQuAD qua dai se duoc windowing, sample "
                         "SNLI/MMLU vuot qua bi bo qua (khong truncate vi se cat mat nhan).")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)

    # MoE loss
    p.add_argument("--lb_loss_coef", type=float, default=0.01,
                    help="He so (lambda) cho load-balancing loss. Mac dinh 0.01 nhu cac file goc.")
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
    p.add_argument("--save_steps", type=int, default=200,
                    help="Luu checkpoint local + push len hub moi N step.")
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="'auto' de tu tim checkpoint moi nhat trong output_dir, hoac duong dan cu the")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"],
                    help="Backend attention. sdpa la mac dinh an toan; flash_attention_2 chi dung "
                         "duoc khi da cai flash-attn.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device_map", type=str, default=None,
                    help="vi du 'auto' de chia model qua nhieu GPU (1 process). Khong dung cung torchrun.")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--diagnostics_dir", type=str, default=None,
                    help="None = <output_dir>/diagnostics")
    p.add_argument("--plot_every", type=int, default=100, help="Ve lai cac plot moi N step")
    p.add_argument("--smooth_window", type=int, default=50,
                    help="So diem lien tiep duoc gom lai (trung binh cong) cho moi diem tren cac "
                         "duong *_smoothed.png.")

    # Distributed (qua torchrun: doc RANK / LOCAL_RANK / WORLD_SIZE tu bien moi truong).
    p.add_argument("--nccl_timeout_minutes", type=int, default=30,
                    help="Timeout cho moi collective op cua NCCL (tang len 30 phut de chiu duoc luc "
                         "rank 0 tokenize du lieu / luu checkpoint cham trong khi cac rank khac cho).")

    return p


# ============================================================================================
# Utils chung
# ============================================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_oom_error(e: RuntimeError) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or ("cuda error" in msg and "memory" in msg)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================================
# Distributed (torchrun). KHONG dung DDP wrapper: gradient duoc all-reduce thu cong 1 lan/step.
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
    logger.info(f"[rank {rank}/{world_size}] Da init NCCL process group "
                f"(timeout={nccl_timeout_minutes} phut, local_rank={local_rank}).")
    return rank, local_rank, world_size, is_distributed, device


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def broadcast_trainable_params(params: List[torch.nn.Parameter]):
    """Dong bo LoRA tu rank 0 sang cac rank con lai (thay cho viec DDP tu broadcast luc khoi tao),
    gop thanh 1 buffer lien tuc de chi ton 1 NCCL call moi dtype."""
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    by_dtype: Dict[torch.dtype, List[torch.Tensor]] = {}
    for p in params:
        by_dtype.setdefault(p.data.dtype, []).append(p.data)
    for tensors in by_dtype.values():
        flat = _flatten_dense_tensors(tensors)
        dist.broadcast(flat, src=0)
        for t, synced in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
            t.copy_(synced)


def sync_grads_across_ranks(trainable_params: List[torch.nn.Parameter]):
    """All-reduce (SUM, KHONG chia trung binh) gradient, gop thanh 1 buffer lien tuc. Khong chia vi
    loss moi rank da duoc chuan hoa theo TONG so token nhan TOAN CUC cua step (xem
    forward_backward_one_chunk) nen tong gradient cac rank chinh la gradient cua loss trung binh
    theo token tren toan bo global batch."""
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    grads_by_dtype: Dict[torch.dtype, List[torch.Tensor]] = {}
    for p in trainable_params:
        if p.grad is not None:
            grads_by_dtype.setdefault(p.grad.dtype, []).append(p.grad)

    for grads in grads_by_dtype.values():
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)


# ============================================================================================
# Du lieu: doc 3 task -> build prompt -> (windowing SQuAD) -> tokenize 1 lan -> cache
# ============================================================================================
def _read_json_records(path: str, name: str) -> list:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Khong tim thay file du lieu {name}: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "train", "examples"):
            if isinstance(data.get(key), list):
                return data[key]
        raise ValueError(f"File {path} la dict nhung khong co key list nao trong (data/train/examples).")
    return data


# ------------------------------------------------------------------------------------- SNLI
def load_snli_records(data_file: str) -> List[Dict]:
    """List cac object {"premise", "hypothesis", "label": 0/1/2}. Bo qua record thieu field, hoac
    label khong nam trong {0, 1, 2} (SNLI goc dung -1 cho cac cau khong dong thuan)."""
    data = _read_json_records(data_file, "SNLI")
    records: List[Dict] = []
    n_skipped = 0
    for rec in data:
        if not isinstance(rec, dict):
            n_skipped += 1
            continue
        premise = rec.get("premise")
        hypothesis = rec.get("hypothesis")
        label = rec.get("label")
        if premise is None or hypothesis is None or label is None:
            n_skipped += 1
            continue
        try:
            label = int(label)
        except (TypeError, ValueError):
            n_skipped += 1
            continue
        if label not in LABEL_TO_WORD:
            n_skipped += 1
            continue
        records.append({
            "premise": str(premise).strip(),
            "hypothesis": str(hypothesis).strip(),
            "label": label,
        })
    logger.info(f"[SNLI] Da doc {len(records)} sample hop le tu {data_file} "
                f"(bo qua {n_skipped} record loi/label khong hop le).")
    return records


def build_snli_prompt(premise: str, hypothesis: str) -> str:
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Question: What is the relationship between the premise and the hypothesis? "
        f"Choose one: entailment, neutral, or contradiction.\n"
        f"Answer:"
    )


def build_snli_examples(records: List[Dict], tokenizer, eos_token: str, max_length: int) -> List[Dict]:
    examples = []
    for rec in records:
        prompt = build_snli_prompt(rec["premise"], rec["hypothesis"])
        examples.append({"prompt": prompt,
                         "full_text": f"{prompt} {LABEL_TO_WORD[rec['label']]}{eos_token}"})
    return examples


# ------------------------------------------------------------------------------------ SQuAD
def load_squad_records(data_file: str) -> List[Dict]:
    """List cac object {"id", "title", "context", "question", "answers": {"text": [...],
    "answer_start": [...]}}. Bo qua record thieu field, khong co dap an (SQuAD 2.0 unanswerable),
    hoac answer_start khong khop voi noi dung context."""
    data = _read_json_records(data_file, "SQuAD")
    records: List[Dict] = []
    n_skipped_missing = 0
    n_skipped_unanswerable = 0
    n_skipped_mismatch = 0
    for rec in data:
        if not isinstance(rec, dict):
            n_skipped_missing += 1
            continue
        context = rec.get("context")
        question = rec.get("question")
        answers = rec.get("answers")
        if context is None or question is None or not isinstance(answers, dict):
            n_skipped_missing += 1
            continue
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []
        if not texts or not starts:
            n_skipped_unanswerable += 1
            continue

        # SQuAD train thuong chi co 1 dap an/cau hoi; neu co nhieu, lay dap an dau tien.
        answer_text = str(texts[0]).strip()
        try:
            answer_start = int(starts[0])
        except (TypeError, ValueError):
            n_skipped_missing += 1
            continue
        if not answer_text:
            n_skipped_unanswerable += 1
            continue

        context = str(context)
        span = context[answer_start: answer_start + len(answer_text)]
        if span.strip() != answer_text.strip():
            found_at = context.find(answer_text)
            if found_at == -1:
                n_skipped_mismatch += 1
                continue
            answer_start = found_at

        records.append({
            "id": rec.get("id"),
            "context": context.strip(),
            "question": str(question).strip(),
            "answer_text": answer_text,
            "answer_start": answer_start,
        })
    logger.info(
        f"[SQuAD] Da doc {len(records)} sample hop le tu {data_file} "
        f"(bo qua {n_skipped_missing} record thieu field, "
        f"{n_skipped_unanswerable} cau hoi khong co dap an, "
        f"{n_skipped_mismatch} record lech offset answer_start/context)."
    )
    return records


def build_squad_prompt(context: str, question: str) -> str:
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def select_context_window(context: str, answer_start: int, answer_text: str,
                           tokenizer, max_context_tokens: int) -> str:
    """Khi full_text vuot qua --max_length, KHONG truncate tho (se cat mat phan 'Answer: ...' o cuoi
    chuoi), ma chon 1 CUA SO cac TU trong context BAO QUANH vi tri cua answer, roi mo rong dan
    sang trai/phai (giu nguyen tung tu) cho toi khi vua sat ngan sach max_context_tokens."""
    spans = [m.span() for m in WORD_SPAN_PATTERN.finditer(context)]
    if not spans:
        return context

    answer_end = answer_start + len(answer_text)
    left_idx, right_idx = None, None
    for i, (s, e) in enumerate(spans):
        if e > answer_start and left_idx is None:
            left_idx = i
        if s < answer_end:
            right_idx = i
    if left_idx is None or right_idx is None:
        left_idx, right_idx = 0, 0
    lo, hi = left_idx, right_idx

    def window_text(lo, hi):
        return context[spans[lo][0]: spans[hi][1]]

    def token_len(s: str) -> int:
        return len(tokenizer(s, add_special_tokens=False)["input_ids"])

    cur_text = window_text(lo, hi)
    if token_len(cur_text) > max_context_tokens:
        # Ngay ca cua so toi thieu da vuot ngan sach -> ham goi se tu phat hien va skip sample nay.
        return cur_text

    while True:
        moved = False
        if lo > 0:
            candidate = window_text(lo - 1, hi)
            if token_len(candidate) <= max_context_tokens:
                lo -= 1
                cur_text = candidate
                moved = True
        if hi < len(spans) - 1:
            candidate = window_text(lo, hi + 1)
            if token_len(candidate) <= max_context_tokens:
                hi += 1
                cur_text = candidate
                moved = True
        if not moved:
            break
    return cur_text


def compute_lengths(tokenizer, texts: Sequence[str], chunk_size: int = 1000, desc: str = "") -> List[int]:
    lengths: List[int] = []
    for i in tqdm(range(0, len(texts), chunk_size), desc=desc or "Tinh do dai token"):
        enc = tokenizer(list(texts[i:i + chunk_size]), add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


def build_squad_examples(records: List[Dict], tokenizer, eos_token: str, max_length: int) -> List[Dict]:
    """Sample nao co full_text vuot qua max_length se duoc windowing lai context; neu van khong vua
    sau khi windowing (hiem) thi bi bo qua."""
    naive = []
    for rec in records:
        prompt = build_squad_prompt(rec["context"], rec["question"])
        naive.append({
            "prompt": prompt,
            "full_text": f"{prompt} {rec['answer_text']}{eos_token}",
            "answer_text": rec["answer_text"],
            "context": rec["context"],
            "question": rec["question"],
            "answer_start": rec["answer_start"],
        })
    naive_lengths = compute_lengths(tokenizer, [ex["full_text"] for ex in naive],
                                    desc="[SQuAD] Tinh do dai token")

    examples: List[Dict] = []
    n_windowed = 0
    n_dropped = 0
    for ex, naive_len in tqdm(list(zip(naive, naive_lengths)), desc="[SQuAD] Windowing context qua dai"):
        if naive_len <= max_length:
            examples.append({"prompt": ex["prompt"], "full_text": ex["full_text"]})
            continue

        overhead_text = f"Context: \nQuestion: {ex['question']}\nAnswer: {ex['answer_text']}{eos_token}"
        overhead_tokens = len(tokenizer(overhead_text, add_special_tokens=True)["input_ids"])
        max_context_tokens = max_length - overhead_tokens - 8
        if max_context_tokens <= 0:
            n_dropped += 1
            continue

        windowed_context = select_context_window(
            ex["context"], ex["answer_start"], ex["answer_text"], tokenizer, max_context_tokens
        )
        new_prompt = build_squad_prompt(windowed_context, ex["question"])
        new_full_text = f"{new_prompt} {ex['answer_text']}{eos_token}"
        new_len = len(tokenizer(new_full_text, add_special_tokens=True)["input_ids"])
        if new_len > max_length:
            n_dropped += 1
            continue
        examples.append({"prompt": new_prompt, "full_text": new_full_text})
        n_windowed += 1

    logger.info(f"[SQuAD] build_examples: {len(examples)} sample giu lai (trong do {n_windowed} sample "
                f"da windowing vi vuot max_length={max_length}), bo qua {n_dropped} sample van qua dai.")
    return examples


# ------------------------------------------------------------------------------------- MMLU
def load_mmlu_records(data_file: str) -> List[Dict]:
    """List cac object {"question": str, "choices": [str, ...], "answer": int, "subject": str}.
    Bo qua record thieu field, choices khong hop le, hoac answer ngoai khoang [0, len(choices)-1]."""
    data = _read_json_records(data_file, "MMLU")
    records: List[Dict] = []
    n_skipped_missing = 0
    n_skipped_bad_answer = 0
    for rec in data:
        if not isinstance(rec, dict):
            n_skipped_missing += 1
            continue
        question = rec.get("question")
        choices = rec.get("choices")
        answer = rec.get("answer")
        subject = rec.get("subject", "")
        if question is None or not isinstance(choices, list) or len(choices) < 2 or answer is None:
            n_skipped_missing += 1
            continue
        try:
            answer = int(answer)
        except (TypeError, ValueError):
            n_skipped_bad_answer += 1
            continue
        if not (0 <= answer < len(choices)) or len(choices) > len(CHOICE_LETTERS):
            n_skipped_bad_answer += 1
            continue

        records.append({
            "question": str(question).strip(),
            "choices": [str(c).strip() for c in choices],
            "answer": answer,
            "subject": str(subject).strip() if subject else "",
        })
    logger.info(
        f"[MMLU] Da doc {len(records)} sample hop le tu {data_file} "
        f"(bo qua {n_skipped_missing} record thieu field, "
        f"{n_skipped_bad_answer} record answer/choices khong hop le)."
    )
    return records


def build_mmlu_prompt(question: str, choices: List[str], subject: str) -> str:
    """Dung chuan format eval MMLU (lm-evaluation-harness / paper goc), khong kem few-shot."""
    if subject:
        header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    else:
        header = "The following are multiple choice questions (with answers).\n\n"
    choice_lines = "\n".join(f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{header}{question}\n{choice_lines}\nAnswer:"


def build_mmlu_examples(records: List[Dict], tokenizer, eos_token: str, max_length: int) -> List[Dict]:
    examples = []
    for rec in records:
        prompt = build_mmlu_prompt(rec["question"], rec["choices"], rec["subject"])
        examples.append({"prompt": prompt,
                         "full_text": f"{prompt} {CHOICE_LETTERS[rec['answer']]}{eos_token}"})
    return examples


TASK_REGISTRY = {
    "snli": {"file_arg": "snli_file", "loader": load_snli_records, "builder": build_snli_examples},
    "squad": {"file_arg": "squad_file", "loader": load_squad_records, "builder": build_squad_examples},
    "mmlu": {"file_arg": "mmlu_file", "loader": load_mmlu_records, "builder": build_mmlu_examples},
}


def parse_tasks(tasks_arg: str) -> List[str]:
    tasks = [t.strip().lower() for t in tasks_arg.split(",") if t.strip()]
    if not tasks:
        raise ValueError("--tasks rong. Chon tap con cua: " + ",".join(ALL_TASKS))
    unknown = [t for t in tasks if t not in TASK_REGISTRY]
    if unknown:
        raise ValueError(f"Task khong ho tro: {unknown}. Chon trong: {list(ALL_TASKS)}")
    if len(set(tasks)) != len(tasks):
        raise ValueError(f"--tasks bi lap: {tasks}")
    return tasks


def tokenize_examples(examples: List[Dict], tokenizer, max_length: int, task_id: int,
                      task_name: str, chunk_size: int = 2000) -> List[Dict]:
    """Tokenize full_text + prompt DUNG 1 LAN. Moi sample -> {"input_ids": int32[L], "prompt_len",
    "n_lab" (so token nhan = L - prompt_len, gom ca eos), "task"}. Sample co L > max_length bi bo
    qua (KHONG truncate vi se cat mat nhan o cuoi chuoi)."""
    out: List[Dict] = []
    n_too_long = 0
    n_no_label = 0
    for i in tqdm(range(0, len(examples), chunk_size), desc=f"[{task_name}] Tokenize"):
        chunk = examples[i:i + chunk_size]
        full_ids = tokenizer([ex["full_text"] for ex in chunk], add_special_tokens=True)["input_ids"]
        prompt_ids = tokenizer([ex["prompt"] for ex in chunk], add_special_tokens=True)["input_ids"]
        for f_ids, p_ids in zip(full_ids, prompt_ids):
            length, plen = len(f_ids), len(p_ids)
            if length > max_length:
                n_too_long += 1
                continue
            if plen < 1 or plen >= length:
                n_no_label += 1
                continue
            out.append({
                "input_ids": np.asarray(f_ids, dtype=np.int32),
                "prompt_len": plen,
                "n_lab": length - plen,
                "task": task_id,
            })
    logger.info(f"[{task_name}] Tokenize xong: giu {len(out)}/{len(examples)} sample "
                f"(bo {n_too_long} sample vuot max_length={max_length}, {n_no_label} sample khong co nhan).")
    return out


def build_mixed_examples(args, tokenizer, tasks: List[str]) -> List[Dict]:
    all_examples: List[Dict] = []
    for task_id, name in enumerate(tasks):
        spec = TASK_REGISTRY[name]
        records = spec["loader"](getattr(args, spec["file_arg"]))
        if args.max_samples_per_task and len(records) > args.max_samples_per_task:
            random.Random(args.seed + task_id).shuffle(records)
            records = records[: args.max_samples_per_task]
            logger.info(f"[{name}] Gioi han con {len(records)} sample (--max_samples_per_task).")
        if not records:
            raise RuntimeError(f"Task {name} khong doc duoc sample nao — kiem tra lai file du lieu.")
        examples = spec["builder"](records, tokenizer, tokenizer.eos_token, args.max_length)
        tokenized = tokenize_examples(examples, tokenizer, args.max_length, task_id, name)
        if not tokenized:
            raise RuntimeError(f"Task {name} khong con sample nao sau buoc loc do dai — thu tang --max_length.")
        all_examples.extend(tokenized)

    if args.max_samples and len(all_examples) > args.max_samples:
        random.Random(args.seed).shuffle(all_examples)
        all_examples = all_examples[: args.max_samples]
        logger.info(f"Gioi han tong con {len(all_examples)} sample (--max_samples).")
    return all_examples


def _file_signature(path: str):
    try:
        st = os.stat(path)
        return [os.path.abspath(path), st.st_size, st.st_mtime_ns]
    except OSError:
        return [os.path.abspath(path), None, None]


def compute_cache_key(args, tokenizer, tasks: List[str]) -> str:
    payload = {
        "version": DATA_FORMAT_VERSION,
        "tokenizer": getattr(tokenizer, "name_or_path", ""),
        "vocab": len(tokenizer),
        "eos": tokenizer.eos_token,
        "max_length": args.max_length,
        "tasks": tasks,
        "files": {t: _file_signature(getattr(args, TASK_REGISTRY[t]["file_arg"])) for t in tasks},
        "max_samples_per_task": args.max_samples_per_task,
        "max_samples": args.max_samples,
        "seed": args.seed,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def get_or_build_dataset(args, tokenizer, tasks: List[str], rank: int, is_distributed: bool) -> List[Dict]:
    """Rank 0 build + tokenize + luu cache; cac rank con lai cho o barrier roi doc cache (tranh
    tokenize trung lap N lan). Lan chay sau / resume doc thang cache."""
    cache_dir = args.cache_dir or os.path.join(args.output_dir, "cache")
    cache_path = os.path.join(cache_dir, f"mixed_tokenized_{compute_cache_key(args, tokenizer, tasks)}.pkl")

    examples = None
    if is_main_process(rank):
        if os.path.exists(cache_path) and not args.rebuild_cache:
            logger.info(f"Tim thay cache du lieu da tokenize: {cache_path}")
        else:
            examples = build_mixed_examples(args, tokenizer, tasks)
            os.makedirs(cache_dir, exist_ok=True)
            tmp_path = cache_path + ".tmp"
            with open(tmp_path, "wb") as f:
                pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            logger.info(f"Da luu cache du lieu: {cache_path}")
    if is_distributed:
        dist.barrier()
    if examples is None:
        with open(cache_path, "rb") as f:
            examples = pickle.load(f)
    return examples


# ============================================================================================
# Token-budget batch sampler (mix deu giua cac rank, can bang tai theo token)
# ============================================================================================
class TokenBudgetBatchSampler:
    """Ke hoach batch cho 1 epoch (xac dinh hoan toan boi seed + epoch, GIONG NHAU tren moi rank):
      1. Shuffle toan bo index -> cat thanh pool `pool_size` sample.
      2. Trong moi pool: sort theo do dai, gom tham lam thanh batch sao cho
         so_sample * do_dai_pad(sample dai nhat) <= max_tokens va so_sample <= max_batch_size
         => batch it padding, va moi batch co ~ cung so token (tai tinh toan ~ bang nhau).
      3. Shuffle TAT CA batch cua cac pool, roi step s lay world_size batch lien tiep: rank r nhan
         batch thu r. Vi batch tren cac rank thuoc cac vung do dai (=> cac task) ngau nhien, 1
         optimizer step tren nhieu GPU se tron ca 3 task.
    So step/epoch duoc co dinh = min tren cac epoch (batch du ra o cuoi bi bo) de scheduler/resume on dinh.

    plan(epoch) tra ve list (theo step) cac tuple (index_cua_rank_nay, tong_so_token_nhan_toan_cuc_cua_step)."""

    def __init__(self, lengths: np.ndarray, label_tokens: np.ndarray, max_tokens: int,
                 max_batch_size: int, world_size: int, rank: int, seed: int, num_epochs: int,
                 pool_size: int = 20000, pad_multiple: int = 8):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.label_tokens = np.asarray(label_tokens, dtype=np.int64)
        self.padded = ((self.lengths + pad_multiple - 1) // pad_multiple) * pad_multiple
        self._padded_list = self.padded.tolist()
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.world_size = max(world_size, 1)
        self.rank = rank
        self.seed = seed
        self.pool_size = max(pool_size, 1)

        counts = []
        first_batches = None
        for e in range(max(num_epochs, 1)):
            batches = self._global_batches(e)
            if e == 0:
                first_batches = batches
            counts.append(len(batches) // self.world_size)
        self.steps_per_epoch = min(counts)
        if self.steps_per_epoch <= 0:
            raise RuntimeError(
                f"Khong du du lieu de tao 1 step (world_size={self.world_size}, "
                f"max_tokens_per_batch={max_tokens}). Giam --max_tokens_per_batch hoac them du lieu."
            )
        self.n_batches_epoch0 = len(first_batches)
        padded_total = sum(len(b) * self._padded_list[b[-1]] if False else
                           len(b) * max(self._padded_list[i] for i in b) for b in first_batches)
        self.padding_efficiency = float(self.lengths.sum()) / max(padded_total, 1)
        self.mean_batch_size = len(self.lengths) / max(len(first_batches), 1)

    def _global_batches(self, epoch: int) -> List[List[int]]:
        rng = random.Random(self.seed + epoch)
        idx = list(range(len(self.lengths)))
        rng.shuffle(idx)
        batches: List[List[int]] = []
        plist = self._padded_list
        for ps in range(0, len(idx), self.pool_size):
            pool = idx[ps:ps + self.pool_size]
            pool.sort(key=plist.__getitem__)  # sort on dinh -> hoa ngau nhien nho shuffle truoc do
            cur: List[int] = []
            for i in pool:
                pad_len = plist[i]  # tang dan => sample hien tai la dai nhat trong batch
                if cur and ((len(cur) + 1) * pad_len > self.max_tokens or len(cur) >= self.max_batch_size):
                    batches.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def plan(self, epoch: int) -> List[Tuple[List[int], int]]:
        batches = self._global_batches(epoch)
        W = self.world_size
        steps: List[Tuple[List[int], int]] = []
        for s in range(self.steps_per_epoch):
            group = batches[s * W:(s + 1) * W]
            global_label_tokens = int(sum(int(self.label_tokens[b].sum()) for b in group))
            steps.append((group[self.rank], max(global_label_tokens, 1)))
        return steps


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
    """Tra ve (targets, kinds): targets la list ten module Linear duoc chon lam LoRA target trong
    khoang layer_indices, kinds la dict ten module -> "router" / "attention" / "expert"."""
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
    """Hook forward tren cac router Linear (da duoc PEFT wrap LoRA) de lay logits phuc vu tinh
    load-balancing loss. Khong detach de gradient van chay ve LoRA cua router."""
    hooks = []
    router_name_set = set(router_names)
    if not router_name_set:
        return hooks
    for name, module in peft_model.named_modules():
        if any(name.endswith(rn) for rn in router_name_set):
            h = module.register_forward_hook(lambda mod, inp, out, cache=cache_list: cache.append(out))
            hooks.append(h)
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
# MoE loss: LM loss (chi tren phan dap an) + Load Balancing loss (Switch/Mixtral style)
# ============================================================================================
def compute_load_balancing_loss(router_logits_list: List[torch.Tensor], attention_mask: torch.Tensor,
                                 num_experts: int, top_k: int):
    """Loai bo padding truoc khi tinh thong ke, noi (concat) token cua TAT CA router layer da hook
    lai thanh 1 tap DUY NHAT roi moi tinh f_i / P_i / loss (dung cach cua HF load_balancing_loss_func)."""
    mask_flat = attention_mask.reshape(-1).bool()

    valid_logits = []
    for logits in router_logits_list:
        logits = logits.reshape(-1, logits.shape[-1]).to(mask_flat.device)
        if logits.shape[0] == mask_flat.shape[0]:
            logits = logits[mask_flat]
        else:
            logger.warning(
                "compute_load_balancing_loss: kich thuoc router logits "
                f"({logits.shape[0]}) khong khop attention_mask ({mask_flat.shape[0]}) -> "
                "bo qua loc padding cho lan tinh nay."
            )
        if logits.shape[0] > 0:
            valid_logits.append(logits)

    if not valid_logits:
        return torch.tensor(0.0, device=attention_mask.device)

    concatenated_logits = torch.cat(valid_logits, dim=0)
    routing_weights = F.softmax(concatenated_logits.float(), dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_experts).float()
    tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # f_i, da chia trung binh theo token
    avg_prob_per_expert = routing_weights.mean(dim=0)
    return num_experts * torch.sum(tokens_per_expert * avg_prob_per_expert)


# ============================================================================================
# Collate + forward/backward
# ============================================================================================
def collate_examples(sub_examples: List[Dict], pad_id: int, device, pad_multiple: int = 8):
    """List example (da tokenize) -> tensor. Pad ve boi so cua pad_multiple (tensor core), padding
    ben phai. Tra ve (input_ids [n,T], attention_mask [n,T], target_mask [n,T-1], task_ids [n])
    trong do target_mask[i, j] = True neu token o vi tri j+1 la token NHAN (tinh loss)."""
    n = len(sub_examples)
    lens = np.fromiter((len(ex["input_ids"]) for ex in sub_examples), dtype=np.int64, count=n)
    plens = np.fromiter((ex["prompt_len"] for ex in sub_examples), dtype=np.int64, count=n)
    tasks = np.fromiter((ex["task"] for ex in sub_examples), dtype=np.int64, count=n)

    T = int(lens.max())
    T = -(-T // pad_multiple) * pad_multiple
    ids = np.full((n, T), pad_id, dtype=np.int64)
    for i, ex in enumerate(sub_examples):
        ids[i, : lens[i]] = ex["input_ids"]
    pos = np.arange(T, dtype=np.int64)[None, :]
    attn = pos < lens[:, None]
    label = (pos >= plens[:, None]) & attn  # nhan = tu vi tri prompt_len den het chuoi (gom eos)

    def to_dev(a):
        return torch.from_numpy(a).to(device)

    return to_dev(ids), to_dev(attn.astype(np.int64)), to_dev(label[:, 1:]), to_dev(tasks)


def forward_backward_one_chunk(chunk: List[Dict], ctx, lb_weight: float):
    """Forward + backward cho 1 chunk (toan bo batch hoac 1 mieng sau khi chia nho vi OOM) va cong
    don thong ke vao ctx.stats (tren device, KHONG sync CPU).

    L_LM van la cross-entropy TIEU CHUAN next-token, CHI tinh tren phan "<answer><eos>". Thay vi tinh
    logits cho moi vi tri roi mask, ta lay hidden state tu decoder va chi dua cac vi tri co nhan
    qua lm_head -> ket qua giong het nhung bo nho logits giam ~50 lan.

    Loss de backward = sum(CE tren token nhan cua chunk) / TONG token nhan TOAN CUC cua step
                       + lb_loss_coef * L_LB * lb_weight
    (gradient cac rank/chunk duoc CONG lai nen ra dung gradient cua loss trung binh theo token)."""
    ids, attn, tgt_mask, task_ids = collate_examples(chunk, ctx.pad_id, ctx.device)

    ctx.router_cache.clear()
    hidden = ctx.decoder(input_ids=ids, attention_mask=attn, use_cache=False).last_hidden_state

    sel_mask = tgt_mask.to(hidden.device)
    h_sel = hidden[:, :-1][sel_mask]  # [N_label_tokens, H], thu tu row-major theo sample
    targets = ids[:, 1:][tgt_mask]
    logits = ctx.lm_head(h_sel).float()
    targets = targets.to(logits.device)
    ce = F.cross_entropy(logits, targets, reduction="none")
    ce_sum = ce.sum()

    if ctx.router_cache and ctx.num_experts and ctx.top_k:
        lb_loss = compute_load_balancing_loss(ctx.router_cache, attn, ctx.num_experts, ctx.top_k)
    else:
        lb_loss = torch.zeros((), device=ce_sum.device)
    lb_loss = lb_loss.to(ce_sum.device)

    loss = ce_sum / ctx.global_tokens + ctx.lb_loss_coef * lb_loss * lb_weight
    loss.backward()

    # Thong ke (khong anh huong gradient)
    with torch.no_grad():
        st = ctx.stats
        dev = st.device
        T = ctx.n_tasks
        n_lab = tgt_mask.sum(dim=1)
        tok_task = torch.repeat_interleave(task_ids.to(dev), n_lab.to(dev), output_size=int(targets.numel()))
        ce_d = ce.detach().to(dev)
        correct = (logits.argmax(dim=-1) == targets).float().to(dev)

        st[0] += ce_d.sum()
        st[1] += float(targets.numel())
        st[2] += correct.sum()
        st[3] += lb_loss.detach().to(dev) * len(chunk)
        st[4] += float(len(chunk))
        st[6] += float(ids.numel())
        st[_N_BASE_STATS:_N_BASE_STATS + T].index_add_(0, tok_task, ce_d)
        st[_N_BASE_STATS + T:_N_BASE_STATS + 2 * T].index_add_(0, tok_task, torch.ones_like(ce_d))
        st[_N_BASE_STATS + 2 * T:_N_BASE_STATS + 3 * T].index_add_(0, tok_task, correct)


def run_batch_with_dynamic_oom_handling(batch_examples: List[Dict], ctx, optimizer, min_batch_size: int):
    """Chay 1 batch (list example). Neu OOM o bat ky chunk nao: ZERO gradient + stats (de khong con
    gradient do dang cua lan that bai), halve kich thuoc chunk roi chay lai CA BATCH tu dau. Neu OOM
    ngay ca khi chunk <= min_batch_size thi bo cac sample do. Luon quay ve kich thuoc batch goc cho
    batch ke tiep. Khong co collective nao trong ham nay nen moi rank tu xu ly OOM doc lap, khong the
    gay treo NCCL."""
    live = list(batch_examples)
    cap = max(len(live), 1)
    n_skipped = 0

    while True:
        optimizer.zero_grad(set_to_none=False)
        ctx.stats.zero_()
        ctx.stats[5] = float(n_skipped)
        if not live:
            return

        n_live = len(live)
        failed_size = None
        failed_ids = None
        try:
            for i in range(0, n_live, cap):
                chunk = live[i:i + cap]
                failed_size, failed_ids = len(chunk), {id(x) for x in chunk}
                forward_backward_one_chunk(chunk, ctx, lb_weight=(len(chunk) / n_live) / ctx.world_size)
            return
        except RuntimeError as e:
            if not is_oom_error(e):
                raise
        # (da thoat khoi except -> traceback/tensor cua lan that bai duoc giai phong truoc khi retry)
        ctx.router_cache.clear()
        clear_memory()
        if failed_size <= max(min_batch_size, 1):
            logger.warning(f"OOM ngay ca voi chunk size={failed_size} -> skip {failed_size} sample nay.")
            live = [x for x in live if id(x) not in failed_ids]
            n_skipped += failed_size
        else:
            cap = max(failed_size // 2, 1)
            logger.warning(f"OOM voi chunk size={failed_size} -> zero gradient, chay lai batch voi chunk={cap}.")


def stats_to_result(stats: List[float], task_names: List[str], lb_loss_coef: float) -> dict:
    T = len(task_names)
    ce, ntok, correct, lb_sum, ns, nskip, npad = stats[:_N_BASE_STATS]
    lm_loss = ce / max(ntok, 1.0)
    lb_loss = lb_sum / max(ns, 1.0)
    t_ce = stats[_N_BASE_STATS:_N_BASE_STATS + T]
    t_tok = stats[_N_BASE_STATS + T:_N_BASE_STATS + 2 * T]
    t_cor = stats[_N_BASE_STATS + 2 * T:_N_BASE_STATS + 3 * T]
    task_loss, task_acc, task_tokens = {}, {}, {}
    for i, name in enumerate(task_names):
        task_tokens[name] = int(t_tok[i])
        task_loss[name] = (t_ce[i] / t_tok[i]) if t_tok[i] > 0 else None
        task_acc[name] = (t_cor[i] / t_tok[i]) if t_tok[i] > 0 else None
    return {
        "lm_loss": lm_loss,
        "lb_loss": lb_loss,
        "total_loss": lm_loss + lb_loss_coef * lb_loss,
        "label_acc": correct / max(ntok, 1.0),
        "n_processed": int(ns),
        "n_skipped": int(nskip),
        "padded_tokens": int(npad),
        "label_tokens": int(ntok),
        "task_loss": task_loss,
        "task_acc": task_acc,
        "task_tokens": task_tokens,
    }


# ============================================================================================
# Checkpoint / resume
# ============================================================================================
def save_checkpoint(output_dir, model, optimizer, scheduler, epoch, step_in_epoch, global_step,
                    steps_per_epoch, world_size):
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
            "steps_per_epoch": steps_per_epoch,
            "world_size": world_size,
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
# Diagnostics: jsonl + plot (loss tong, accuracy, loss/acc theo task)
# ============================================================================================
def log_step_to_jsonl(jsonl_path, global_step, epoch, result, step_time):
    rec = {
        "step": global_step,
        "epoch": epoch,
        "lm_loss": result["lm_loss"],
        "lb_loss": result["lb_loss"],
        "total_loss": result["total_loss"],
        "label_acc": result["label_acc"],
        "n_processed": result["n_processed"],
        "n_skipped": result["n_skipped"],
        "padded_tokens": result["padded_tokens"],
        "label_tokens": result["label_tokens"],
        "task_loss": result["task_loss"],
        "task_acc": result["task_acc"],
        "task_tokens": result["task_tokens"],
        "step_time": step_time,
        "timestamp": time.time(),
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(jsonl_path) -> List[dict]:
    recs: List[dict] = []
    if not os.path.exists(jsonl_path):
        return recs
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def _bin_mean(xs: List[float], ys: List[float], window: int):
    window = max(window, 1)
    out_x, out_y = [], []
    for i in range(0, len(ys), window):
        cy = ys[i:i + window]
        cx = xs[i:i + len(cy)]
        out_x.append(cx[-1])
        out_y.append(sum(cy) / len(cy))
    return out_x, out_y


def plot_all(jsonl_path, diagnostics_dir, task_names: List[str], window: int):
    recs = _read_jsonl(jsonl_path)
    if not recs:
        return
    steps = [r["step"] for r in recs]
    lm = [r["lm_loss"] for r in recs]
    lb = [r["lb_loss"] for r in recs]
    total = [r["total_loss"] for r in recs]
    acc = [r.get("label_acc", 0.0) for r in recs]
    title = "Qwen1.5-MoE-A2.7B-Finetuning (SNLI+SQuAD+MMLU)"

    # 1. Loss tho tung step
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lm, label="L_LM (chi tren token dap an)")
    plt.plot(steps, lb, label="L_LB")
    plt.plot(steps, total, label="L_Total")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"{title} - loss (tho, tung step)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # 2. Loss lam min
    if window > 1:
        plt.figure(figsize=(10, 6))
        for name, ys in (("L_LM", lm), ("L_LB", lb), ("L_Total", total)):
            bx, by = _bin_mean(steps, ys, window)
            plt.plot(bx, by, label=f"{name} (mean)", marker="o", markersize=3)
        plt.xlabel(f"Training step (moi diem = trung binh cua {window} step lien tiep)")
        plt.ylabel("Loss (trung binh)")
        plt.title(f"{title} - loss trung binh moi {window} step")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(diagnostics_dir, "loss_curve_smoothed.png"), dpi=150)
        plt.close()

    # 3. Accuracy tong
    bx, by = _bin_mean(steps, acc, window)
    plt.figure(figsize=(10, 6))
    plt.plot(bx, by, label="Answer token accuracy (train, xap xi)", color="green", marker="o", markersize=3)
    plt.xlabel(f"Training step (moi diem = trung binh cua {window} step lien tiep)")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title(f"{title} - ty le token dung cua phan dap an tren train batch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(diagnostics_dir, "accuracy_curve.png"), dpi=150)
    plt.close()

    # 4. Loss + accuracy THEO TASK (chi tinh cac step co task do xuat hien)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for name in task_names:
        pts_l = [(r["step"], r["task_loss"][name]) for r in recs
                 if r.get("task_loss") and r["task_loss"].get(name) is not None]
        pts_a = [(r["step"], r["task_acc"][name]) for r in recs
                 if r.get("task_acc") and r["task_acc"].get(name) is not None]
        if pts_l:
            bx, by = _bin_mean([p[0] for p in pts_l], [p[1] for p in pts_l], window)
            axes[0].plot(bx, by, label=name, marker="o", markersize=3)
        if pts_a:
            bx, by = _bin_mean([p[0] for p in pts_a], [p[1] for p in pts_a], window)
            axes[1].plot(bx, by, label=name, marker="o", markersize=3)
    axes[0].set_title("L_LM theo task (trung binh moi diem)")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss")
    axes[1].set_title("Answer token accuracy theo task (xap xi)")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(f"{title} - diagnostics theo task")
    fig.tight_layout()
    fig.savefig(os.path.join(diagnostics_dir, "task_curves_smoothed.png"), dpi=150)
    plt.close(fig)


# ============================================================================================
# Hugging Face Hub push (bat dong bo, khong upload optimizer state)
# ============================================================================================
def build_model_card(args, tasks, task_counts, num_experts, top_k, layer_start, layer_end, num_layers) -> str:
    counts_str = ", ".join(f"{t}: {task_counts.get(t, 0)}" for t in tasks)
    return f"""---
license: apache-2.0
base_model: {args.model_name_or_path}
datasets:
- stanfordnlp/snli
- rajpurkar/squad
- cais/mmlu
tags:
- lora
- peft
- moe
- mixture-of-experts
- multi-task
- nli
- question-answering
- multiple-choice
- fine-tuned
---

# Qwen1.5-MoE-A2.7B-Finetuning

LoRA adapter finetune tu [`{args.model_name_or_path}`](https://huggingface.co/{args.model_name_or_path})
(Mixture-of-Experts) tren **mix 3 task**: SNLI (NLI), SQuAD (extractive QA), MMLU (trac nghiem),
train chung trong 1 lan (cac tap duoc tron lai, shuffle).

Task: `{", ".join(tasks)}` — so sample sau khi loc: {counts_str}.

## Prompt format (giu nguyen nhu khi train de eval zero-shot)
SNLI:
```
Premise: <premise>
Hypothesis: <hypothesis>
Question: What is the relationship between the premise and the hypothesis? Choose one: entailment, neutral, or contradiction.
Answer: <label>
```
SQuAD:
```
Context: <context>
Question: <question>
Answer: <answer>
```
MMLU:
```
The following are multiple choice questions (with answers) about <subject>.

<question>
A. <choice_A>
B. <choice_B>
C. <choice_C>
D. <choice_D>
Answer: <letter>
```
Phan sau `Answer:` la phan model sinh ra va la phan DUY NHAT duoc tinh loss.

## Cau hinh LoRA
- Layer duoc finetune: `[{layer_start}, {layer_end})` trong tong so `{num_layers}` layer (khoang 1L/3 -> 2L/3).
- Module gan LoRA: **attention** (r = {args.lora_r_attention}), **router** (r = {args.lora_r_router}),
  **experts** (r = {args.lora_r_experts}) trong khoang layer tren (rank rieng qua `rank_pattern` cua PEFT).
- alpha = {args.lora_alpha}, dropout = {args.lora_dropout}

## Loss
`L_total = L_LM + lb_loss_coef * L_LB`
- `L_LM`: cross-entropy chuan, chi tren phan `<answer><eos>`, chuan hoa theo tong so token nhan cua
  global batch.
- `L_LB`: load balancing loss chuan cua MoE (Switch/Mixtral style) tren cac router trong khoang layer duoc finetune.
- `lb_loss_coef` = {args.lb_loss_coef}, `num_experts` = {num_experts}, `top_k` = {top_k}

## Training
- {args.num_train_epochs} epoch, lr = {args.learning_rate}, warmup = {args.warmup_ratio}, grad clip = {args.gradient_clip_norm}
- Token-budget batching: toi da {args.max_tokens_per_batch} token (sau pad) va {args.max_batch_size} sample moi batch moi GPU.
- `max_length` = {args.max_length}; context SQuAD qua dai duoc windowing quanh vi tri dap an.

## Diagnostics
Xem thu muc `diagnostics/`: `loss_log.jsonl` (tung step, gom loss/acc theo task), `loss_curve.png`,
`loss_curve_smoothed.png`, `accuracy_curve.png`, `task_curves_smoothed.png`. Accuracy o day la ty le
token dung (teacher forcing) tren train batch, KHONG phai do chinh xac chuan tren tap eval.
"""


def push_to_hub(local_ckpt_dir, diagnostics_dir, hub_model_id, private, readme_path, token=None):
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub chua duoc cai, bo qua buoc push_to_hub.")
        return
    api = HfApi(token=token)
    api.create_repo(repo_id=hub_model_id, private=private, exist_ok=True)
    # Khong upload trainer_state.pt (optimizer state nang, chi can de resume local).
    api.upload_folder(folder_path=local_ckpt_dir, repo_id=hub_model_id, path_in_repo=".",
                      ignore_patterns=["trainer_state.pt"],
                      commit_message=f"Update checkpoint: {os.path.basename(local_ckpt_dir)}")
    if os.path.isdir(diagnostics_dir):
        api.upload_folder(folder_path=diagnostics_dir, repo_id=hub_model_id,
                          path_in_repo="diagnostics", commit_message="Update diagnostics")
    api.upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=hub_model_id,
                    commit_message="Update model card")


class AsyncHubPusher:
    """Push len hub trong thread nen de GPU khong phai doi luc upload. Moi lan submit se cho lan
    push truoc xong (de khong chong len nhau). Loi push chi duoc log, khong lam dung training."""

    def __init__(self, diagnostics_dir, hub_model_id, private, readme_path, token):
        self.args = (diagnostics_dir, hub_model_id, private, readme_path, token)
        self.thread: Optional[threading.Thread] = None

    def _run(self, ckpt_dir):
        diagnostics_dir, hub_model_id, private, readme_path, token = self.args
        try:
            push_to_hub(ckpt_dir, diagnostics_dir, hub_model_id, private, readme_path, token=token)
            logger.info(f"Da push checkpoint len hub: {hub_model_id} ({os.path.basename(ckpt_dir)})")
        except Exception as e:
            logger.error(f"Push checkpoint len hub that bai (checkpoint local van o {ckpt_dir}): {e}")

    def submit(self, ckpt_dir):
        self.wait()
        self.thread = threading.Thread(target=self._run, args=(ckpt_dir,), daemon=False)
        self.thread.start()

    def wait(self):
        if self.thread is not None and self.thread.is_alive():
            logger.info("Doi lan push hub truoc do hoan tat ...")
        if self.thread is not None:
            self.thread.join()
            self.thread = None


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

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tasks = parse_tasks(args.tasks)

    hf_token = load_hf_token(args.env_file, args.hf_token) if (args.push_to_hub and is_main_process(rank)) else None
    if args.push_to_hub and is_main_process(rank) and hf_token is None:
        logger.warning(
            "push_to_hub=True nhung khong tim thay HF token nao (--hf_token / bien moi truong / .env). "
            "Buoc push co the that bai voi 401 Unauthorized neu repo chua ton tai hoac chua co quyen ghi san."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    diagnostics_dir = args.diagnostics_dir or os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = os.path.join(diagnostics_dir, "loss_log.jsonl")
    readme_path = os.path.join(args.output_dir, "README.md")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # -------------------------------------------------------------------------------- tokenizer
    logger.info(f"Dang load tokenizer tu {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                               trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pad_id = tokenizer.pad_token_id

    # ------------------------------------------------------------------------------------ data
    # Lam du lieu TRUOC khi load model 14B: loi duong dan/du lieu se bao ngay, va rank 0 tokenize
    # trong luc cac rank khac cho o barrier (chua ton VRAM).
    logger.info(f"Dang chuan bi du lieu mix {tasks} ...")
    examples = get_or_build_dataset(args, tokenizer, tasks, rank, is_distributed)
    if len(examples) == 0:
        raise RuntimeError("Khong co sample nao sau khi build du lieu.")

    lengths = np.fromiter((len(ex["input_ids"]) for ex in examples), dtype=np.int64, count=len(examples))
    label_tokens = np.fromiter((ex["n_lab"] for ex in examples), dtype=np.int64, count=len(examples))
    task_counts = {t: int(sum(1 for ex in examples if ex["task"] == i)) for i, t in enumerate(tasks)}
    logger.info("So sample moi task sau khi loc: " + ", ".join(f"{t}={c}" for t, c in task_counts.items())
                + f" | tong={len(examples)}")

    batch_sampler = TokenBudgetBatchSampler(
        lengths, label_tokens, max_tokens=args.max_tokens_per_batch, max_batch_size=args.max_batch_size,
        world_size=world_size, rank=rank, seed=args.seed, num_epochs=args.num_train_epochs,
        pool_size=args.pool_size,
    )
    steps_per_epoch = batch_sampler.steps_per_epoch
    total_steps = steps_per_epoch * args.num_train_epochs
    logger.info(
        f"Token-budget batching: ~{batch_sampler.mean_batch_size:.1f} sample/batch, "
        f"padding efficiency ~{batch_sampler.padding_efficiency * 100:.1f}%, "
        f"{steps_per_epoch} step/epoch x {args.num_train_epochs} epoch = {total_steps} step "
        f"(world_size={world_size})."
    )

    # ---------------------------------------------------------------------------------- model
    logger.info(f"Dang load model tu {args.model_name_or_path} ...")
    model_kwargs = dict(torch_dtype=dtype, trust_remote_code=args.trust_remote_code,
                        attn_implementation=args.attn_implementation)
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
    logger.info(f"lb_loss_coef (lambda load balancing) = {args.lb_loss_coef}")

    # ---------------------------------------------------------------------------- resume / LoRA
    resume_dir = find_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_dir:
        logger.info(f"Resume LoRA adapter tu checkpoint: {resume_dir}")
        model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
    else:
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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_distributed:
        # Khong dung DDP: dong bo LoRA tu rank 0 mot lan, sau do moi step all-reduce gradient 1 lan.
        broadcast_trainable_params(trainable_params)

    # Lay thang decoder + lm_head de chi tinh logits tai cac vi tri co nhan (LoRA da duoc inject
    # vao cac module ben trong nen forward qua decoder van di qua LoRA binh thuong).
    inner_lm = model.get_base_model()
    decoder = inner_lm.get_decoder() if hasattr(inner_lm, "get_decoder") else inner_lm.model
    lm_head = inner_lm.get_output_embeddings()
    model_device = next(model.parameters()).device

    # ------------------------------------------------------------------------------- optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                  weight_decay=args.weight_decay, foreach=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    start_epoch, start_step_in_epoch, global_step = 0, 0, 0
    if resume_dir:
        state_path = os.path.join(resume_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
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
            if (state.get("steps_per_epoch") not in (None, steps_per_epoch)
                    or state.get("world_size") not in (None, world_size)):
                logger.warning(
                    f"Ke hoach batch hien tai (steps_per_epoch={steps_per_epoch}, world_size={world_size}) "
                    f"KHAC luc luu checkpoint (steps_per_epoch={state.get('steps_per_epoch')}, "
                    f"world_size={state.get('world_size')}) -> vi tri resume trong epoch co the khong khop. "
                    f"Nen giu nguyen world_size/--max_tokens_per_batch/--max_batch_size/--pool_size/--seed/--tasks."
                )
            if start_step_in_epoch >= steps_per_epoch:
                start_epoch += 1
                start_step_in_epoch = 0

    pusher = None
    if is_main_process(rank):
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(build_model_card(args, tasks, task_counts, num_experts, top_k,
                                     layer_start, layer_end, num_layers))
        if args.push_to_hub:
            pusher = AsyncHubPusher(diagnostics_dir, args.hub_model_id, args.hub_private,
                                    readme_path, hf_token)

    n_tasks = len(tasks)
    ctx = SimpleNamespace(
        decoder=decoder, lm_head=lm_head, router_cache=router_logits_cache,
        num_experts=num_experts, top_k=top_k, lb_loss_coef=args.lb_loss_coef,
        pad_id=pad_id, device=model_device, n_tasks=n_tasks, world_size=world_size,
        stats=torch.zeros(_N_BASE_STATS + 3 * n_tasks, dtype=torch.float32, device=model_device),
        global_tokens=1.0,
    )

    # ------------------------------------------------------------------------------ training loop
    model.train()
    last_epoch, last_step_in_epoch = start_epoch, max(start_step_in_epoch - 1, 0)
    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            plan = batch_sampler.plan(epoch)
            step_offset = start_step_in_epoch if epoch == start_epoch else 0

            pbar = tqdm(
                range(step_offset, steps_per_epoch),
                total=steps_per_epoch,
                initial=step_offset,
                desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
                disable=not is_main_process(rank),
            )
            for step_in_epoch in pbar:
                t0 = time.time()
                batch_idx, global_label_tokens = plan[step_in_epoch]
                batch_examples = [examples[i] for i in batch_idx]
                ctx.global_tokens = float(global_label_tokens)

                run_batch_with_dynamic_oom_handling(batch_examples, ctx, optimizer, args.min_batch_size)

                if is_distributed:
                    dist.all_reduce(ctx.stats, op=dist.ReduceOp.SUM)
                result = stats_to_result(ctx.stats.tolist(), tasks, args.lb_loss_coef)
                do_step = result["n_processed"] > 0  # giong nhau tren moi rank sau all-reduce

                if do_step:
                    if is_distributed:
                        sync_grads_across_ranks(trainable_params)
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
                    optimizer.step()
                scheduler.step()
                global_step += 1
                last_epoch, last_step_in_epoch = epoch, step_in_epoch

                step_time = time.time() - t0
                if is_main_process(rank):
                    postfix = {
                        "L_LM": f"{result['lm_loss']:.4f}",
                        "L_LB": f"{result['lb_loss']:.4f}",
                        "tok_acc": f"{result['label_acc']:.3f}",
                        "ktok/s": f"{result['padded_tokens'] / max(step_time, 1e-6) / 1000:.1f}",
                        "skipped": result["n_skipped"],
                    }
                    for name in tasks:
                        v = result["task_loss"][name]
                        if v is not None:
                            postfix[name] = f"{v:.3f}"
                    pbar.set_postfix(postfix)

                    if result["n_processed"] > 0:
                        log_step_to_jsonl(jsonl_path, global_step, epoch, result, step_time)

                    if global_step % args.plot_every == 0:
                        plot_all(jsonl_path, diagnostics_dir, tasks, args.smooth_window)

                    if global_step % args.save_steps == 0:
                        ckpt_dir = save_checkpoint(args.output_dir, model, optimizer, scheduler, epoch,
                                                   step_in_epoch, global_step, steps_per_epoch, world_size)
                        plot_all(jsonl_path, diagnostics_dir, tasks, args.smooth_window)
                        logger.info(f"Da luu checkpoint local: {ckpt_dir}")
                        if pusher is not None:
                            pusher.submit(ckpt_dir)  # khong chan training

                if is_distributed and global_step % args.save_steps == 0:
                    dist.barrier()

            start_step_in_epoch = 0

        if is_main_process(rank):
            final_ckpt = save_checkpoint(args.output_dir, model, optimizer, scheduler,
                                         args.num_train_epochs - 1, steps_per_epoch - 1, global_step,
                                         steps_per_epoch, world_size)
            plot_all(jsonl_path, diagnostics_dir, tasks, args.smooth_window)
            if pusher is not None:
                pusher.submit(final_ckpt)
                pusher.wait()
            logger.info("Training hoan tat.")
        if is_distributed:
            dist.barrier()

    except KeyboardInterrupt:
        logger.warning("Nhan KeyboardInterrupt — luu checkpoint khan cap truoc khi thoat ...")
        if is_main_process(rank):
            save_checkpoint(args.output_dir, model, optimizer, scheduler, last_epoch,
                            last_step_in_epoch, global_step, steps_per_epoch, world_size)
            plot_all(jsonl_path, diagnostics_dir, tasks, args.smooth_window)
        raise
    finally:
        if pusher is not None:
            pusher.wait()
        for h in hooks:
            h.remove()
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()