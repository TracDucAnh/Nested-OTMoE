"""
MidAlign baseline cho Qwen/Qwen1.5-MoE-A2.7B (Qwen2MoeForCausalLM) — Alternate Training:

    step CHAN (0, 2, 4, ...) = TASK step     : GIONG HET Qwen1.5-MoE-A2.7B.py (finetuning english MIX task:
                                               SNLI + SQuAD + MMLU, LM loss CHI tren phan dap an)
    step LE   (1, 3, 5, ...) = ALIGN step    : contrastive Eq.(1) MidAlign (Liu & Niehues, 2025) tai 1 middle layer
                                               tren cap song song english-other  <-- PHAN THEM VAO DUY NHAT

NGUYEN TAC: bo phan ALIGN ra thi chuoi TASK step phai la chuoi step cua file finetuning english mix task.
Vi vay moi thu cua task step duoc lay NGUYEN VAN tu file finetuning:
  * doc du lieu, build prompt, windowing SQuAD, loc do dai, tokenize 1 lan, mask nhan (prompt = khong tinh loss)
  * thu tu task mac dinh snli,squad,mmlu (-> cung thu tu noi du lieu)
  * TokenBudgetBatchSampler (batch co dinh --batch_size sample / rank, pool sort theo do dai, shuffle batch);
    cung seed + world_size + batch_size => cung chuoi batch nhu file finetuning
  * L_task = sum CE(token nhan) / TONG token nhan TOAN CUC cua step + lb_loss_coef * L_LB
    (L_LB tinh gop token cua moi router layer nhu HF load_balancing_loss_func), gradient all-reduce SUM
  * dynamic OOM handling, clip grad 1.0, AdamW (weight_decay 0)
  * hyper-param: lr 2e-4, warmup 3% roi GIAM TUYEN TINH ve 0, 3 epoch, lb_loss_coef 0.01,
    LoRA router r=4 / attention r=16 / experts r=16, alpha=32, dropout=0.05, layers = [L/3, 2L/3) hoac --layers
  * KHONG LoRA len shared_expert (file finetuning khong co) — bat lai bang --include_shared_expert

Lich xen ke: moi epoch co N task step (N = steps_per_epoch cua sampler finetuning) va N align step,
step_in_epoch chan = task, le = align. LR scheduler chi tien 1 buoc moi CAP (task k, align k), nen task step
thu k co DUNG learning rate ma file finetuning dung o step k; align step k dung chung LR voi task step k.
Optimizer (AdamW) la 1 doi tuong chung cho ca 2 loai step (nhu paper), moi step 1 lan optimizer.step().

ALIGN (giu nguyen tu ban truoc): Eq.(1) paper, 1 chieu, cosine / temperature (1.5), mean-pooled hidden state tai
--align_layer (mac dinh = layer giua cua khoang layer LoRA), negative chi trong mini-batch 32 (--align_micro_batch_size),
pool nho seed co dinh (--align_pairs_per_lang cap / ngon ngu tu flores+ntrex+ted), lap lai de du N align step / epoch,
resample deu giua cac ngon ngu (--align_lang_balance). Align step dung gradient TRUNG BINH qua rank.

Dong bo gradient: all-reduce thu cong (khong DDP). Task step: SUM (loss da chuan hoa theo token nhan toan cuc);
align step: MEAN.

Vi du chay (8 GPU) — task step khop file finetuning (--batch_size 8):
    torchrun --standalone --nproc_per_node=8 Qwen1.5-MoE-A2.7B-midalign.py \\
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \\
        --data_dir data/processed_alignment --alignment_data flores ntrex ted \\
        --align_pairs_per_lang 500 --seed 42 \\
        --snli_file data/english_task/snli/train.json \\
        --squad_file data/english_task/squad/train.json \\
        --mmlu_file data/english_task/mmlu/auxiliary_train.json \\
        --batch_size 8 --align_batch_size 32 --push_to_hub

Smoke test (1 GPU, du lieu nho):
    python Qwen1.5-MoE-A2.7B-midalign.py --max_samples_per_task 2000 --align_pairs_per_lang 50 --no_push_to_hub

Resume:
    torchrun ... --resume_from_checkpoint auto
(giu nguyen world_size, --batch_size, --align_batch_size, --seed, --tasks, du lieu — se bao loi neu steps_per_epoch lech.)
"""

import argparse
import gc
import glob
import hashlib
import json
import logging
import os
import pickle
import random
import re
import shutil
import string
import time
from datetime import timedelta
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
logger = logging.getLogger("midalign_qwen15_moe")


# ============================================================================================
# HF token + mapping nguon alignment (giu nguyen tu ban MidAlign truoc)
# ============================================================================================
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


ALIGNMENT_DATASET_FILES = {
    "flores": "flores.json",
    "ntrex": "ntrex.json",
    "ted": "ted.json",      # giong code finetuning: --alignment_data ... ted -> <data_dir>/ted.json
}


def resolve_data_files(alignment_data: Sequence[str], data_files: Optional[Sequence[str]]) -> List[str]:
    """--data_files (neu duoc truyen thu cong) luon THANG the va duoc dung nguyen ven.
    Nguoc lai, suy ra danh sach file JSON tu --alignment_data qua ALIGNMENT_DATASET_FILES,
    giu thu tu xuat hien dau tien va loai trung (vd --alignment_data flores flores ntrex ->
    chi doc flores.json + ntrex.json 1 lan)."""
    if data_files:
        return list(data_files)
    resolved: List[str] = []
    for name in alignment_data:
        fname = ALIGNMENT_DATASET_FILES[name]
        if fname not in resolved:
            resolved.append(fname)
    return resolved


# ============================================================================================
# Argparse
# ============================================================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MidAlign: alternate [task step = finetuning english mix SNLI+SQuAD+MMLU] / [align step = contrastive] cho Qwen1.5-MoE-A2.7B"
    )

    # Model / output
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--output_dir", type=str,
                    default="training/MidAlign/checkpoints/Qwen1.5-MoE-A2.7B")

    # Du lieu ALIGNMENT (contrastive step, bitext english-other)
    p.add_argument("--data_dir", type=str, default="data/processed_alignment")
    p.add_argument("--alignment_data", type=str, nargs="+",
                    choices=sorted(ALIGNMENT_DATASET_FILES.keys()),
                    default=["flores", "ntrex", "ted"],
                    help="Nguon du lieu alignment: flores | ntrex | ted (1 hoac nhieu). Mac dinh flores ntrex ted "
                         "(doc <data_dir>/flores.json, ntrex.json, ted.json).")
    p.add_argument("--data_files", type=str, nargs="+", default=None,
                    help="[Nang cao] Ghi de --alignment_data bang danh sach file JSON trong --data_dir.")
    p.add_argument("--eng_key", type=str, default="eng_Latn")
    p.add_argument("--max_lang_pairs_per_record", type=int, default=None,
                    help="Gioi han so ngon ngu ghep voi eng_key trong 1 record (None = dung het).")
    p.add_argument("--ted_pairs_per_record", type=int, default=0,
                    help="RIENG cho file TED (ten file bat dau bang 'ted'): moi record chi lay NGAU NHIEN "
                         "N cap eng-other thay vi to hop het cac ngon ngu. MAC DINH 0 = lay het (de buoc "
                         "boc mau theo --align_pairs_per_lang la dong deu tren moi cap hop le cua 3 tap).")
    p.add_argument("--align_pairs_per_lang", type=int, default=500,
                    help="So cap eng-other lay NGAU NHIEN (seed = --seed, reservoir sampling) cho MOI ngon ngu "
                         "tu HOP cua tat ca file alignment. Paper chi dung vai tram cau song song / ngon ngu "
                         "(Javanese 264, Swahili 371, Welsh 823) nen mac dinh 500. Ngon ngu co it cap hon thi "
                         "lay het. <= 0 = khong gioi han (dung het, KHONG khuyen dung — trai paper).")
    p.add_argument("--max_samples", type=int, default=None,
                    help="[Debug] Gioi han TONG so cap bitext sau khi da boc --align_pairs_per_lang, "
                         "None = dung het.")

    # Du lieu TASK (task step) — DUNG GIONG HET file finetuning english mix task
    p.add_argument("--tasks", type=str, default="snli,squad,mmlu",
                    help="Task mix cho task step, cach nhau dau phay (tap con cua snli,squad,mmlu). THU TU quyet dinh "
                         "task_id + thu tu noi du lieu -> giu mac dinh de chuoi batch khop file finetuning.")
    p.add_argument("--snli_file", type=str, default="data/english_task/snli/train.json")
    p.add_argument("--squad_file", type=str, default="data/english_task/squad/train.json")
    p.add_argument("--mmlu_file", type=str, default="data/english_task/mmlu/auxiliary_train.json")
    p.add_argument("--max_samples_per_task", type=int, default=None,
                    help="[Debug] Gioi han so sample MOI task, None = dung het.")

    # Hugging Face Hub
    p.add_argument("--push_to_hub", action="store_true", default=True)
    p.add_argument("--no_push_to_hub", dest="push_to_hub", action="store_false")
    p.add_argument("--hub_model_id", type=str, default="ducanhdinh/Qwen1.5-MoE-A2.7B-MidAlign")
    p.add_argument("--hub_private", action="store_true")
    p.add_argument("--env_file", type=str, default=".env")
    p.add_argument("--hf_token", type=str, default=None)

    # Training schedule + batch (xem docstring dau file ve cach tinh so step)
    p.add_argument("--num_train_epochs", type=int, default=3)
    # --- TASK step (khop file finetuning)
    p.add_argument("--batch_size", type=int, default=8,
                    help="TASK step: so sample cua 1 batch tren MOI rank (co dinh). Batch toan cuc moi task step = "
                         "batch_size x world_size. Giu bang --batch_size cua file finetuning de khop task step.")
    p.add_argument("--max_tokens_per_batch", type=int, default=0,
                    help="MAC DINH 0 = TAT (batch co dinh --batch_size). > 0 = token-budget batching (giong file finetuning).")
    p.add_argument("--max_batch_size", type=int, default=512, help="Chi dung khi --max_tokens_per_batch > 0.")
    p.add_argument("--min_batch_size", type=int, default=1,
                    help="Khi OOM, chunk <= gia tri nay ma van OOM thi bo qua chunk do.")
    p.add_argument("--pool_size", type=int, default=20000,
                    help="Moi epoch: shuffle toan bo sample, cat pool N sample, sort theo do dai trong pool.")
    p.add_argument("--task_max_length", type=int, default=512,
                    help="max_length (token) cua 1 sample task (SQuAD qua dai duoc windowing, SNLI/MMLU qua dai bi bo).")
    # --- ALIGN step (phan them vao)
    p.add_argument("--align_batch_size", type=int, default=128,
                    help="ALIGN step: so cap align / rank / align step (paper: effective 128). Chia thanh mini-batch "
                         "--align_micro_batch_size de tinh contrastive loss + cong don gradient. Moi epoch co N align "
                         "step (N = so task step / epoch) -> can N * world_size * align_batch_size cap, lay tu pool nho "
                         "bang cach LAP LAI. Chi phi align ~ align_batch_size, giam neu align step qua cham so voi task step.")
    p.add_argument("--align_micro_batch_size", type=int, default=32,
                    help="Kich thuoc MINI-BATCH contrastive (paper: 32). Negative chi nam trong mini-batch; "
                         "gradient cong don qua cac mini-batch cua 1 align step. 0 = khong chia.")
    p.add_argument("--max_length", type=int, default=256,
                    help="max_length cho cau alignment (contrastive step).")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)

    # MidAlign: alignment objective
    p.add_argument("--align_layer", type=int, default=None,
                    help="Layer lay hidden state cho contrastive loss (hidden_states[align_layer]). "
                         "Chi so 1-indexed (hidden_states[i] = output cua block i-1). None = TU DONG lay "
                         "layer GIUA khoang layer finetune: block (min+max)//2 cua --layers (vd 7-14 -> "
                         "block 10 0-indexed -> align_layer = 11); khong co --layers thi dung [L/3, 2L/3).")
    p.add_argument("--align_temperature", type=float, default=1.5,
                    help="tau (paper: tim trong {0.1,1.0,1.5,2.0}; Qwen dung 1.5, Llama 0.1).")
    p.add_argument("--align_loss", type=str, default="oneway", choices=["oneway", "symmetric"],
                    help="oneway = Eq.(1) paper (query = cau Anh, key = cau target trong mini-batch); "
                         "symmetric = trung binh 2 chieu (khong co trong paper).")
    p.add_argument("--align_global_negatives", dest="align_global_negatives",
                    action="store_true", default=False,
                    help="[Mac dinh TAT, dung paper] all_gather embedding qua cac rank de negative la "
                         "mini-batch TOAN CUC (world_size * align_micro_batch_size) thay vi mini-batch local.")
    p.add_argument("--no_align_global_negatives", dest="align_global_negatives", action="store_false")
    p.add_argument("--align_lang_balance", dest="align_lang_balance", action="store_true", default=True,
                    help="[Mac dinh BAT, dung paper] resample du lieu align moi epoch ve phan phoi xap xi "
                         "deu giua cac ngon ngu (tong so cap/epoch khong doi).")
    p.add_argument("--no_align_lang_balance", dest="align_lang_balance", action="store_false")
    p.add_argument("--mask_false_negatives", dest="mask_false_negatives",
                    action="store_true", default=True,
                    help="[Mac dinh BAT] Loai khoi softmax cac 'negative' thuc ra la cung 1 cau Anh "
                         "(hoac cung 1 cau target) voi positive — hay gap vi du lieu multiway ghep "
                         "1 cau Anh voi nhieu ngon ngu.")
    p.add_argument("--no_mask_false_negatives", dest="mask_false_negatives", action="store_false")

    # MoE loss (phu tro cho task step)
    p.add_argument("--lb_loss_coef", type=float, default=0.01,
                    help="He so load-balancing loss cua TASK step. Mac dinh 0.01 nhu file finetuning. "
                         "Truyen -1 de lay config.router_aux_loss_coef cua model.")
    p.add_argument("--num_local_experts", type=int, default=None)
    p.add_argument("--num_experts_per_tok", type=int, default=None)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16, help="Rank fallback (khong dung khi 3 nhom deu co rank rieng).")
    p.add_argument("--lora_r_router", type=int, default=4, help="Rank LoRA cho router/gating (finetuning: 4).")
    p.add_argument("--lora_r_attn", "--lora_r_attention", dest="lora_r_attn", type=int, default=16,
                    help="Rank LoRA cho attention Q/K/V/O (finetuning: 16).")
    p.add_argument("--lora_r_expert", "--lora_r_experts", dest="lora_r_expert", type=int, default=16,
                    help="Rank LoRA cho cac expert FFN (finetuning: 16).")
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--include_shared_expert", action="store_true", default=False,
                    help="Them LoRA len shared_expert. MAC DINH TAT vi file finetuning english mix task KHONG co "
                         "(de tap tham so train duoc giong het).")
    p.add_argument("--layers", type=str, default=None,
                    help="Danh sach layer (block, chi so tu 0) duoc finetune LoRA. Vi du: '8,9,10' hoac "
                         "'8-15' hoac '0-3,10,12-14' (khoang a-b gom ca a va b). Neu KHONG truyen thi "
                         "mac dinh finetune cac layer [L/3, 2L/3). Khac voi --align_layer (layer tinh "
                         "contrastive loss).")

    # Checkpoint / resume
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--resume_from_checkpoint", type=str, default=None,
                    help="'auto' de tu tim checkpoint moi nhat trong output_dir, hoac duong dan cu the")

    # Distributed
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--nccl_timeout_minutes", type=int, default=30)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_implementation", type=str, default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"], help="Giong file finetuning (sdpa).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Chi dung khi CHAY DON PROCESS (khong qua torchrun)")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--diagnostics_dir", type=str, default=None)
    p.add_argument("--log_every", type=int, default=10,
                    help="Cap nhat postfix cua thanh tien do moi N step.")
    p.add_argument("--plot_every", type=int, default=0,
                    help="Ve lai 2 graph moi N step (0 = chi ve khi luu checkpoint / ket thuc). Ve graph "
                         "doc lai TOAN BO loss_log.jsonl + luu 2 PNG nen rank 0 bi cham, ca cac rank khac "
                         "phai cho o all-reduce -> khong nen de nho.")
    p.add_argument("--cache_dir", type=str, default=None,
                    help="Thu muc cache pool alignment/task da build (mac dinh <output_dir>/cache). "
                         "Rank 0 build 1 lan, cac rank khac + cac lan chay sau doc lai.")
    p.add_argument("--no_cache", action="store_true", help="Bo qua cache, build lai du lieu tu dau.")
    p.add_argument("--smooth_window", type=int, default=50,
                    help="Cua so trung binh truot (so diem cua MOI duong, task va align tinh rieng) cho duong dam tren 2 graph.")
    return p


def sync_grads_across_ranks(trainable_params: List[torch.Tensor], world_size: int, average: bool = True):
    """All-reduce gradient THU CONG (average=True: TRUNG BINH — dung cho ALIGN step; average=False: SUM — dung cho
    TASK step, vi loss task da chia cho TONG token nhan TOAN CUC nen tong gradient cac rank la dung gradient can tim), gop thanh 1 buffer lien tuc theo dtype.
    Tham so khong co grad (None) — vd LoRA cua expert khong nhan token nao tren rank nay — duoc
    dien 0 TRUOC khi gop, de MOI rank luon all-reduce buffer cung kich thuoc va cung thu tu."""
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    for p in trainable_params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    grads_by_dtype: Dict[torch.dtype, List[torch.Tensor]] = {}
    for p in trainable_params:
        grads_by_dtype.setdefault(p.grad.dtype, []).append(p.grad)
    for _, grads in grads_by_dtype.items():
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        if average:
            flat.div_(world_size)
        for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)


def broadcast_trainable_params(trainable_params: List[torch.Tensor], src: int = 0):
    """Dong bo gia tri khoi tao LoRA tu rank 0 (khong con DDP tu broadcast luc khoi tao)."""
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    by_dtype: Dict[torch.dtype, List[torch.Tensor]] = {}
    for p in trainable_params:
        by_dtype.setdefault(p.dtype, []).append(p)
    for ps in by_dtype.values():
        datas = [p.data for p in ps]
        flat = _flatten_dense_tensors(datas)
        dist.broadcast(flat, src=src)
        for d, synced in zip(datas, _unflatten_dense_tensors(flat, datas)):
            d.copy_(synced)



# ============================================================================================
# Utils + distributed (khong con DDP — xem docstring dau file)
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


def setup_distributed(args) -> Tuple[bool, int, int, int, torch.device]:
    """Tra ve (is_distributed, local_rank, global_rank, world_size, device).

    Neu duoc khoi chay qua torchrun (WORLD_SIZE > 1 trong bien moi truong), khoi tao
    process group va tra ve thong tin distributed. Nguoc lai, chay don process nhu binh
    thuong (tuong thich nguoc, khong bat buoc phai co torchrun)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Mac dinh PyTorch chi cho 10 phut cho moi thao tac NCCL; tang len --nccl_timeout_minutes
        # de chiu duoc luc rank 0 luu checkpoint / push len Hub trong khi cac rank khac cho o barrier.
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=args.nccl_timeout_minutes),
        )
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


def transformers_at_least(ver: str) -> bool:
    import transformers
    from packaging.version import Version
    return Version(transformers.__version__.split("+")[0]) >= Version(ver)


def file_signature(paths: Sequence[str]) -> List:
    sig = []
    for p in paths:
        try:
            st = os.stat(p)
            sig.append([os.path.abspath(p), st.st_size, int(st.st_mtime)])
        except OSError:
            sig.append([os.path.abspath(p), None, None])
    return sig


def cached_build(cache_path: Optional[str], build_fn, is_main: bool, is_distributed: bool):
    """Rank 0 build (hoac doc cache neu co), luu cache, roi cac rank khac doc lai tu cache.
    Tranh viec MOI rank tu doc json 22M cap + tokenize 700k sample (log cu: 2 rank lam trung nhau
    ~4 phut + gap doi RAM). cache_path=None -> khong cache (moi rank tu build, nhu cu)."""
    def _load():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if cache_path is None:
        return build_fn()
    if os.path.exists(cache_path):
        if is_main:
            logger.info(f"Dung cache: {cache_path}")
        return _load()

    obj, err = None, None
    if is_main:
        try:
            obj = build_fn()
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp = cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
            logger.info(f"Da luu cache: {cache_path}")
        except BaseException as e:  # van phai toi barrier de cac rank khac khong treo toi timeout
            err = e
    if is_distributed:
        dist.barrier()
    if err is not None:
        raise err
    if not is_main:
        if not os.path.exists(cache_path):
            raise RuntimeError(f"Rank 0 build du lieu that bai (khong co {cache_path}).")
        obj = _load()
    return obj


# ============================================================================================
# Du lieu ALIGNMENT: bitext english-other tu JSON multiway-parallel
# ============================================================================================
def load_bitext_pairs(data_dir: str, data_files: Sequence[str], eng_key: str,
                       max_lang_pairs_per_record: Optional[int] = None,
                       seed: int = 42, ted_pairs_per_record: Optional[int] = None,
                       pairs_per_lang: Optional[int] = None):
    """Tra ve (pairs, src_ids): pairs = [(eng, other, lang)], src_ids = np.uint8 (chi so file nguon).

    pairs_per_lang > 0 (mac dinh cua CLI: 500, nhu paper — "vai tram cau" / ngon ngu): voi MOI ngon ngu,
      gom TAT CA cap eng-other hop le tu HOP cua cac file roi boc NGAU NHIEN dung pairs_per_lang cap bang
      reservoir sampling (Algorithm R) voi random.Random(seed) -> moi cap hop le co cung xac suat duoc chon,
      ket qua tai lap duoc 100% theo (seed, noi dung file, thu tu file). Bo nho ~ O(so_ngon_ngu * pairs_per_lang)
      thay vi giu hang chuc trieu cap. Ngon ngu co it cap hon pairs_per_lang -> lay het.
    pairs_per_lang None/<=0: doc TOAN BO cap hop le (hanh vi cu, trai paper)."""
    use_res = bool(pairs_per_lang) and pairs_per_lang > 0
    pairs: List[Tuple[str, str, str]] = []
    src_ids: List[int] = []
    res: Dict[str, List[Tuple[str, str, int]]] = {}    # lang -> reservoir [(eng, other, file_idx)]
    seen: Dict[str, int] = {}                          # lang -> tong so cap hop le da duyet
    rng = random.Random(seed)        # chon cap/record (TED, max_lang_pairs_per_record)
    rng_res = random.Random(seed)    # reservoir sampling (--seed 42)
    for fi, fname in enumerate(data_files):
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            logger.warning(f"Khong tim thay file {path}, bo qua.")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = list(data.values()) if isinstance(data, dict) else data
        del data

        is_ted = os.path.basename(fname).lower().startswith("ted")
        n_no_eng = 0
        n_valid = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue
            eng_text = rec.get(eng_key)
            if not isinstance(eng_text, str) or not eng_text.strip():
                n_no_eng += 1
                continue
            eng_text = eng_text.strip()
            other_keys = [k for k in rec.keys() if k not in ("id", eng_key)]
            if is_ted and ted_pairs_per_record and ted_pairs_per_record > 0:
                # TED: chon ngau nhien trong cac ngon ngu CO cau hop le (khong boc trung khoa rong)
                valid = [k for k in other_keys if isinstance(rec.get(k), str) and rec[k].strip()]
                other_keys = rng.sample(valid, min(ted_pairs_per_record, len(valid)))
            elif max_lang_pairs_per_record is not None and len(other_keys) > max_lang_pairs_per_record:
                other_keys = rng.sample(other_keys, max_lang_pairs_per_record)
            for k in other_keys:
                v = rec.get(k)
                if not (isinstance(v, str) and v.strip()):
                    continue
                v = v.strip()
                n_valid += 1
                if use_res:
                    n_seen = seen.get(k, 0) + 1
                    seen[k] = n_seen
                    bucket = res.setdefault(k, [])
                    if len(bucket) < pairs_per_lang:
                        bucket.append((eng_text, v, fi))
                    else:
                        j = rng_res.randrange(n_seen)
                        if j < pairs_per_lang:
                            bucket[j] = (eng_text, v, fi)
                else:
                    pairs.append((eng_text, v, k))
                    src_ids.append(fi)

        if n_no_eng:
            ex = next((list(r.keys())[:6] for r in records if isinstance(r, dict)), [])
            logger.warning(f"{fname}: {n_no_eng}/{len(records)} record KHONG co khoa '{eng_key}' "
                           f"(hoac rong) -> bi bo qua. Vi du cac khoa cua 1 record: {ex}. Neu ca file "
                           f"bi bo qua, dung --eng_key de chi dinh dung ten khoa tieng Anh.")
        logger.info(f"{fname}: {n_valid} cap bitext hop le ({eng_key}-other), "
                    f"tong so record = {len(records)}")
        del records
        gc.collect()

    if use_res:
        for lang in sorted(res):
            for e, o, fi in res[lang]:
                pairs.append((e, o, lang))
                src_ids.append(fi)
        n_short = {l: seen[l] for l in res if seen[l] < pairs_per_lang}
        logger.info(f"[align] boc NGAU NHIEN (seed={seed}) toi da {pairs_per_lang} cap / ngon ngu tu "
                    f"{sum(seen.values())} cap hop le cua {len(res)} ngon ngu -> pool align = {len(pairs)} cap.")
        if n_short:
            logger.info(f"[align] {len(n_short)} ngon ngu co < {pairs_per_lang} cap nen lay het: "
                        f"{dict(sorted(n_short.items(), key=lambda kv: kv[1])[:10])}"
                        f"{' ...' if len(n_short) > 10 else ''}")
    return pairs, np.asarray(src_ids, dtype=np.uint8)


# ============================================================================================
# Doc du lieu TASK — COPY NGUYEN VAN tu file finetuning english mix task (Qwen1.5-MoE-A2.7B.py)
# ============================================================================================
# Tang DATA_FORMAT_VERSION neu doi prompt/cach build du lieu de vo hieu hoa cache cu.
ALL_TASKS = ("snli", "squad", "mmlu")
DATA_FORMAT_VERSION = 1
LABEL_TO_WORD = {0: "entailment", 1: "neutral", 2: "contradiction"}
CHOICE_LETTERS = string.ascii_uppercase
WORD_SPAN_PATTERN = re.compile(r"\S+")
# Layout vector thong ke moi task step (all-reduce 1 lan): [ce_sum, n_label_tok, n_correct, lb_sum,
# n_samples, n_skipped, n_padded_tok] + [task_ce_sum]*T + [task_tok]*T + [task_correct]*T
_N_BASE_STATS = 7


def is_oom_error(e: RuntimeError) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or ("cuda error" in msg and "memory" in msg)



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
        examples = spec["builder"](records, tokenizer, tokenizer.eos_token, args.task_max_length)
        tokenized = tokenize_examples(examples, tokenizer, args.task_max_length, task_id, name)
        if not tokenized:
            raise RuntimeError(f"Task {name} khong con sample nao sau buoc loc do dai — thu tang --task_max_length.")
        all_examples.extend(tokenized)

    return all_examples




# ============================================================================================
# Tu dong tim target module LoRA (range layer) + MoE helpers (giu nguyen tu ban MidAlign truoc)
# ============================================================================================
LAYER_IDX_PATTERN = re.compile(r"\.(?:layers|h|blocks|block)\.(\d+)\.")


def get_num_layers(config) -> int:
    for attr in ("num_hidden_layers", "num_layers", "n_layer", "n_layers"):
        if hasattr(config, attr):
            return int(getattr(config, attr))
    raise ValueError("Khong tim thay so luong layer trong model.config. Hay kiem tra ten attribute.")


def parse_layers_arg(spec: str, num_layers: int) -> List[int]:
    """Parse chuoi --layers ('8,9,10', '8-15', '0-3,10,12-14') thanh list layer da sap xep, khong trung.
    Khoang a-b la dong (gom ca a va b). Bao loi neu sai cu phap hoac chi so ngoai [0, num_layers)."""
    indices = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        try:
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if lo > hi:
                    raise ValueError(f"khoang '{part}' co dau > cuoi")
                indices.update(range(lo, hi + 1))
            else:
                indices.add(int(part))
        except ValueError as e:
            raise ValueError(f"--layers '{spec}' khong hop le tai '{part}': {e}") from e
    if not indices:
        raise ValueError(f"--layers '{spec}' khong chua layer nao.")
    bad = sorted(i for i in indices if i < 0 or i >= num_layers)
    if bad:
        raise ValueError(f"--layers co chi so ngoai pham vi [0, {num_layers - 1}]: {bad}")
    return sorted(indices)


def format_layers(layers: Sequence[int]) -> str:
    """[8,9,10,12] -> '8-10, 12' (de log / model card)."""
    parts, start, prev = [], None, None
    for i in sorted(layers):
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            parts.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = i
    if start is not None:
        parts.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(parts)


def infer_align_layer_from_lora_range(lora_layers: Sequence[int]) -> int:
    """Tu dong chon align_layer = layer GIUA cua KHOANG layer finetune LoRA [min, max] (khoang bao
    gom ca 2 dau, ke ca khi --layers khong lien tuc). Block 0-indexed o giua = (min + max) // 2
    (so layer chan -> lay layer giua BEN TRAI), vd khoang 7..14 -> block 10.
    Tra ve align_layer theo quy uoc 1-indexed cua hidden_states (= block 0-indexed + 1), vd 7..14 -> 11.
    Voi mac dinh [L/3, 2L/3) cua backbone 24 layer = 8..15 -> block 11 -> align_layer 12 (nhu truoc).
    """
    return (min(lora_layers) + max(lora_layers)) // 2 + 1


def infer_middle_layer(num_layers: int) -> int:
    """Tu dong suy ra 'middle layer' (1-indexed, dung quy uoc hidden_states[i],
    Layer ID 0 = embedding giong Figure 1/4 paper MidAlign) TU num_layers THUC TE
    cua backbone dang load, thay vi hard-code mot con so co dinh (vd 12, chi dung
    cho backbone 24-layer). Cong thuc: floor(num_layers / 2) — vd 24 layer -> 12,
    28 layer -> 14, 32 layer -> 16 (Llama-3-8B). Duoc dung lam gia tri mac dinh
    khi nguoi dung khong tu chi dinh --align_layer; van co the override thu cong."""
    return num_layers // 2


def is_routed_expert_name(name: str) -> bool:
    return ".experts." in name or ".expert." in name


def is_shared_expert_name(name: str) -> bool:
    return ".shared_expert." in name or ".shared_experts." in name


def is_expert_name(name: str, include_shared: bool = False) -> bool:
    """Mac dinh CHI routed experts (mlp.experts.N.*) — dung nhu file finetuning english mix task.
    include_shared=True them shared expert (mlp.shared_expert.*) cua Qwen1.5-MoE."""
    return is_routed_expert_name(name) or (include_shared and is_shared_expert_name(name))


def is_router_leaf_name(name: str) -> bool:
    leaf = name.split(".")[-1]
    return leaf in ("gate", "router", "gating") and not is_routed_expert_name(name) and not is_shared_expert_name(name)


def build_lora_target_modules(model, layer_indices: set, include_shared_expert: bool = False) -> List[str]:
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
        is_expert = is_expert_name(name, include_shared_expert)
        is_router = is_router_leaf_name(name)
        if is_attn or is_expert or is_router:
            targets.append(name)
    return targets


def build_rank_pattern(attn_names, expert_names, router_names, r_attn, r_expert, r_router) -> Dict[str, int]:
    """rank_pattern gon bang REGEX (vai key) thay vi 1480 ten day du.
    PEFT tra rank cho tung module bang re.match voi MOI key cua rank_pattern (khong cache duoc vi
    >512 pattern khac nhau) -> O(n_target^2) ~ 1.1M lan bien dich regex ~ 2 PHUT khoi tao LoRA (log:
    04:03:14 -> 04:05:08). Gom cac ten thanh vai pattern 'layers\\.\\d+\\.mlp\\.experts\\.\\d+\\.gate_proj'..."""
    def pat(name: str) -> str:
        parts = (name.split(".", 1)[1] if "." in name else name).split(".")
        return r"\.".join(r"\d+" if p.isdigit() else re.escape(p) for p in parts)
    rp: Dict[str, int] = {}
    for names, r in ((router_names, r_router), (attn_names, r_attn), (expert_names, r_expert)):
        for n in names:
            rp[pat(n)] = r
    return rp


def categorize_lora_targets(target_modules: Sequence[str],
                            include_shared_expert: bool = False) -> Tuple[List[str], List[str], List[str]]:
    """Chia target_modules thanh attention / experts / router — DUNG CUNG dieu kien voi build_lora_target_modules."""
    attn_names, expert_names, router_names = [], [], []
    for name in target_modules:
        if is_router_leaf_name(name):
            router_names.append(name)
        elif is_expert_name(name, include_shared_expert):
            expert_names.append(name)
        elif re.search(r"(self_attn|attention|attn)\.", name):
            attn_names.append(name)
    return attn_names, expert_names, router_names


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
# Align step: contrastive loss tai DUNG 1 middle layer (giu nguyen tu ban MidAlign truoc)
# ============================================================================================
def mean_pool_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


class _StopForward(Exception):
    """Dung forward ngay sau block align_layer (khong chay cac layer sau + norm + lm_head)."""


_DECODER_LAYERS_CACHE: Dict[int, torch.nn.ModuleList] = {}


def _find_decoder_layers(model, num_layers: int) -> Optional[torch.nn.ModuleList]:
    key = id(model)
    if key in _DECODER_LAYERS_CACHE:
        return _DECODER_LAYERS_CACHE[key]
    found = None
    for name, mod in model.named_modules():
        if name.endswith(".layers") and isinstance(mod, torch.nn.ModuleList) and len(mod) == num_layers:
            found = mod
            break
    _DECODER_LAYERS_CACHE[key] = found
    return found


def encode_layer_representation(texts: List[str], tokenizer, model, align_layer: int,
                                 max_length: int, device, router_logits_cache: list) -> torch.Tensor:
    """Mean-pooled hidden state tai align_layer. Quy uoc: hidden_states[i] = output SAU decoder block
    0-based (i-1) (Layer ID 0 = embedding nhu paper MidAlign).

    TOI UU: thay vi chay het 24 layer + lm_head (logits [B, T, 151936] rat ton VRAM/thoi gian,
    chi de vut di), ta hook output cua block (align_layer-1) roi nem _StopForward de dung forward
    ngay do. Autograd van dung (graph chi gom cac layer da chay). Gia tri hidden state GIONG HET
    outputs.hidden_states[align_layer] (voi align_layer < num_layers, do la output tho cua block,
    chua qua norm cuoi). Neu khong tim thay layer list hoac align_layer == num_layers thi quay ve
    duong cu (chay day du + output_hidden_states)."""
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attention_mask = enc["attention_mask"].to(device, non_blocking=True)

    router_logits_cache.clear()  # khong dung cho align step, chi de tranh cache tich luy
    num_layers = get_num_layers(get_underlying_model(model).config)
    layers = _find_decoder_layers(model, num_layers) if align_layer < num_layers else None

    if layers is None:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[align_layer]
        return mean_pool_hidden(hidden, attention_mask)

    captured = {}

    def _grab(mod, inp, out):
        captured["h"] = out[0] if isinstance(out, (tuple, list)) else out
        raise _StopForward()

    handle = layers[align_layer - 1].register_forward_hook(_grab)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except _StopForward:
        pass
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError("Hook tai align layer khong duoc kich hoat — kiem tra --align_layer.")
    return mean_pool_hidden(captured["h"], attention_mask)


def _text_id(text: str) -> int:
    """Hash on dinh (khong phu thuoc PYTHONHASHSEED -> giong nhau tren moi rank) cua 1 cau, vua int64."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=7).digest(), "big")


def _gather_ids(ids: List[int], device) -> torch.Tensor:
    t = torch.tensor(ids, dtype=torch.int64, device=device)
    if not (dist.is_available() and dist.is_initialized()):
        return t
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t)
    return torch.cat(out, dim=0)


def _gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """all_gather CO autograd (backward = reduce_scatter) -> [world * b, d], noi theo thu tu rank.
    Gradient tu loss cua MOI rank deu chay nguoc ve embedding local cua rank so huu."""
    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
        return x
    from torch.distributed.nn.functional import all_gather as _all_gather_autograd
    return torch.cat(_all_gather_autograd(x), dim=0)


def compute_alignment_step(eng_texts: List[str], other_texts: List[str], tokenizer, model,
                            align_layer: int, max_length: int, device, temperature: float,
                            router_logits_cache: list, global_negatives: bool = False,
                            mask_false_negatives: bool = True, symmetric: bool = False):
    """Contrastive loss Eq.(1) MidAlign (mac dinh MOT CHIEU Anh->target; symmetric=True: 2 chieu) tai align_layer.

    global_negatives=True: all_gather embedding (CO gradient) tu TAT CA rank -> batch contrastive
      la batch TOAN CUC (world_size * batch_local), vd 8 GPU x 8 cap = 64 cap / 63 negative moi
      cap, bang batch size 64 cua paper. Moi rank chi tinh loss cho cac dong (query) cua rank do
      (local) nhung ung vien (key) la toan bo batch. Gradient tham so SAU all-reduce TRUNG BINH
      qua rank chinh xac bang gradient cua loss tren batch toan cuc (autograd all_gather cong
      don gradient cua cac rank vao embedding local roi all-reduce chia W).
    global_negatives=False: moi rank tu tinh tren batch local (hanh vi cu).

    mask_false_negatives=True: voi du lieu multiway, cung 1 cau Anh (hoac cung 1 cau target) co
      the xuat hien o 2 cap khac nhau trong cung batch -> cap (i, j) nhu vay KHONG phai negative
      that; loai khoi mau so softmax (logit = -inf). Vi tri duong cheo (positive) luon giu."""
    # TOI UU: gop eng + other thanh 1 batch 2b -> 1 forward thay vi 2 (MoE+LoRA bi gioi han boi so
    # kernel launch, nen gan nhu giam nua thoi gian align step). Padding phai (causal) + mean-pool co
    # mask nen gia tri tung cau khong doi so voi 2 forward rieng.
    n_pair = len(eng_texts)
    pooled = encode_layer_representation(list(eng_texts) + list(other_texts), tokenizer, model,
                                          align_layer, max_length, device, router_logits_cache)
    pooled_eng, pooled_other = pooled[:n_pair], pooled[n_pair:]

    a_loc = F.normalize(pooled_eng.float(), dim=-1)     # [b, d]
    b_loc = F.normalize(pooled_other.float(), dim=-1)   # [b, d]
    n_loc = a_loc.size(0)

    if global_negatives:
        a_all, b_all = _gather_with_grad(a_loc), _gather_with_grad(b_loc)
        rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        offset = rank * n_loc            # rank r giu cac chi so toan cuc [r*b, (r+1)*b)
    else:
        a_all, b_all, offset = a_loc, b_loc, 0

    sim_e2o = torch.matmul(a_loc, b_all.t()) / temperature   # [b, B]  query = Anh local
    sim_o2e = torch.matmul(b_loc, a_all.t()) / temperature if symmetric else None   # query = target local
    target = offset + torch.arange(n_loc, device=sim_e2o.device)

    if mask_false_negatives:
        eng_ids = [_text_id(t) for t in eng_texts]
        oth_ids = [_text_id(t) for t in other_texts]
        if global_negatives:
            eng_all, oth_all = _gather_ids(eng_ids, device), _gather_ids(oth_ids, device)
        else:
            eng_all = torch.tensor(eng_ids, dtype=torch.int64, device=device)
            oth_all = torch.tensor(oth_ids, dtype=torch.int64, device=device)
        eng_loc, oth_loc = eng_all[offset:offset + n_loc], oth_all[offset:offset + n_loc]
        dup = (eng_loc[:, None] == eng_all[None, :]) | (oth_loc[:, None] == oth_all[None, :])
        dup[torch.arange(n_loc, device=dup.device), target] = False   # giu positive
        sim_e2o = sim_e2o.masked_fill(dup, float("-inf"))
        if sim_o2e is not None:
            sim_o2e = sim_o2e.masked_fill(dup, float("-inf"))

    loss_e2o = F.cross_entropy(sim_e2o, target)   # Eq.(1) paper: s = Anh, mau so = sum_v exp(sim(h_s, h_t^v))
    if sim_o2e is None:
        return loss_e2o
    return (loss_e2o + F.cross_entropy(sim_o2e, target)) / 2.0


def align_step_accumulate(eng_texts: List[str], other_texts: List[str], tokenizer, model,
                           align_layer: int, max_length: int, device, temperature: float,
                           router_logits_cache: list, micro_bs: int, global_negatives: bool,
                           mask_false_negatives: bool, symmetric: bool) -> float:
    """1 align step = effective batch (vd 128 cap) chia thanh cac MINI-BATCH contrastive (paper: 32);
    negative CHI nam trong mini-batch; backward tung mini-batch (loss nhan ty le size/n) de cong don
    gradient — bo nho chi phu thuoc kich thuoc mini-batch. Tra ve loss trung binh (float)."""
    n = len(eng_texts)
    n_mb = 1 if (micro_bs is None or micro_bs <= 0 or n <= micro_bs) else -(-n // micro_bs)
    sizes = split_sizes(n, n_mb)          # deu nhau (chenh <= 1) -> khong sinh mini-batch 1 cap
    total, pos = 0.0, 0
    for sz in sizes:
        loss = compute_alignment_step(
            eng_texts[pos:pos + sz], other_texts[pos:pos + sz], tokenizer, model, align_layer,
            max_length, device, temperature, router_logits_cache,
            global_negatives=global_negatives, mask_false_negatives=mask_false_negatives,
            symmetric=symmetric)
        (loss * (sz / n)).backward()
        total += float(loss.detach()) * (sz / n)
        pos += sz
        router_logits_cache.clear()
    return total


# ============================================================================================
# TASK side: sampler + collate + forward/backward + dynamic OOM + stats — COPY NGUYEN VAN tu file finetuning
# ============================================================================================
class TokenBudgetBatchSampler:
    """Ke hoach batch cho 1 epoch (xac dinh hoan toan boi seed + epoch, GIONG NHAU tren moi rank).
    Mac dinh batch CO DINH `batch_size` sample/rank; neu max_tokens > 0 thi chuyen sang token-budget.
    Cac buoc:
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
                 pool_size: int = 20000, pad_multiple: int = 8, batch_size: int = 8):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.label_tokens = np.asarray(label_tokens, dtype=np.int64)
        self.padded = ((self.lengths + pad_multiple - 1) // pad_multiple) * pad_multiple
        self._padded_list = self.padded.tolist()
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.batch_size = max(int(batch_size), 1)  # chi dung khi max_tokens <= 0 (batch co dinh)
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
                f"batch_size={self.batch_size}, max_tokens_per_batch={max_tokens}). "
                f"Giam --batch_size (hoac --max_tokens_per_batch) hoac them du lieu."
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
                if self.max_tokens > 0:  # che do token-budget
                    flush = bool(cur) and ((len(cur) + 1) * pad_len > self.max_tokens
                                           or len(cur) >= self.max_batch_size)
                else:  # che do batch co dinh: du batch_size sample thi dong batch
                    flush = len(cur) >= self.batch_size
                if flush:
                    batches.append(cur)
                    cur = []
                cur.append(i)
            # batch co dinh: bo batch cuoi chua du batch_size (toi da batch_size-1 sample/pool/epoch)
            if cur and (self.max_tokens > 0 or len(cur) >= self.batch_size):
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
# Collate + forward/backward (task step) — nguyen van file finetuning
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
# ALIGN side: pool nho, lap lai de du N align step / epoch (N = so task step / epoch cua sampler finetuning)
# ============================================================================================
def split_sizes(total: int, parts: int) -> List[int]:
    """Chia `total` thanh `parts` phan chenh nhau toi da 1 (phan dau nhan them 1 neu du)."""
    base, rem = divmod(total, parts)
    return [base + 1 if i < rem else base for i in range(parts)]


class AlignPlan:
    """Ke hoach align cho 1 epoch (xac dinh hoan toan theo (seed, epoch) -> resume giua epoch ra dung batch cu).
    Moi epoch can align_per_epoch = N * W * align_batch_size cap, sinh tu pool n_align (nho) bang cach lap lai pool:
      - lang_ids != None (paper 4.3): moi ngon ngu nhan ~align_per_epoch / L cap, quay vong qua nhieu hoan vi
        noi tiep -> phan phoi ngon ngu xap xi DEU;
      - nguoc lai: nhieu hoan vi noi tiep cua ca pool.
    Moi lat align = W * align_batch_size cap, chia rank theo buoc W -> moi rank dung align_batch_size cap
    (all_gather cua contrastive loss, neu bat, can cung shape tren moi rank).
    Plan nay KHONG anh huong toi chuoi TASK step (task do TokenBudgetBatchSampler quyet dinh)."""

    def __init__(self, n_align, steps_per_epoch_side, world_size, rank, seed, align_batch_size, lang_ids=None):
        self.n_align = n_align
        self.lang_ids = None if lang_ids is None else np.asarray(lang_ids)
        self._lang_groups = None
        if self.lang_ids is not None:
            order = np.argsort(self.lang_ids, kind="stable")
            _, starts = np.unique(self.lang_ids[order], return_index=True)
            self._lang_groups = np.split(order, starts[1:])
        self.N, self.W, self.rank, self.seed = steps_per_epoch_side, world_size, rank, seed
        self.align_per_step = align_batch_size * world_size            # cap align TOAN CUC / align step
        self.align_per_epoch = self.align_per_step * steps_per_epoch_side

    def _align_epoch_stream(self, epoch: int) -> np.ndarray:
        M = self.align_per_epoch
        rng = np.random.default_rng([self.seed, epoch, 2])
        if self._lang_groups is None:
            reps = -(-M // self.n_align)
            return np.concatenate([rng.permutation(self.n_align) for _ in range(reps)])[:M]
        L = len(self._lang_groups)
        quota = split_sizes(M, L)
        rng.shuffle(quota)                           # ngon ngu nao nhan phan du thay doi theo epoch
        parts = []
        for g, q in zip(self._lang_groups, quota):
            reps = -(-q // len(g))
            parts.append(np.concatenate([rng.permutation(g) for _ in range(reps)])[:q])
        return rng.permutation(np.concatenate(parts))

    def epoch_batches(self, epoch: int) -> List[np.ndarray]:
        a_stream = self._align_epoch_stream(epoch)
        return [a_stream[k * self.align_per_step:(k + 1) * self.align_per_step][self.rank::self.W]
                for k in range(self.N)]


def reduce_metrics(vals: List[float], device, is_distributed: bool) -> List[float]:
    t = torch.tensor(vals, dtype=torch.float64, device=device)
    if is_distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()


# ============================================================================================
# Checkpoint / resume — chi giu 1 checkpoint moi nhat
# ============================================================================================
def save_checkpoint_and_rotate(output_dir, model, optimizer, scheduler, epoch, step_in_epoch,
                                global_step, steps_per_epoch, world_size,
                                prev_checkpoint_dir: Optional[str]) -> str:
    underlying = get_underlying_model(model)
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    underlying.save_pretrained(ckpt_dir)  # PeftModel: chi luu adapter LoRA
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,   # step CUOI CUNG DA HOAN THANH trong epoch (-1 = chua co)
            "global_step": global_step,
            "steps_per_epoch": steps_per_epoch,
            "world_size": world_size,
        },
        os.path.join(ckpt_dir, "trainer_state.pt"),
    )
    with open(os.path.join(output_dir, "latest_checkpoint.txt"), "w") as f:
        f.write(ckpt_dir)
    if prev_checkpoint_dir and os.path.isdir(prev_checkpoint_dir) and prev_checkpoint_dir != ckpt_dir:
        shutil.rmtree(prev_checkpoint_dir, ignore_errors=True)
        logger.info(f"Da xoa checkpoint cu: {prev_checkpoint_dir}")
    return ckpt_dir


# ============================================================================================
# Diagnostics: jsonl + plot (tach task loss va align loss vi 2 loai step khac nhau)
# ============================================================================================
def log_step_to_jsonl(jsonl_f, global_step, epoch, step_type, lm_loss=None, lb_loss=None,
                       task_total_loss=None, token_acc=None, align_loss=None, n_samples=None, extra=None):
    rec = {
        "step": global_step, "epoch": epoch, "step_type": step_type,
        "lm_loss": lm_loss, "lb_loss": lb_loss, "task_total_loss": task_total_loss,
        "token_acc": token_acc, "align_loss": align_loss, "n_samples": n_samples,
        "timestamp": time.time(),
    }
    if extra:
        rec.update(extra)
    jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")     # handle mo san, line-buffered


def _moving_avg(vals: List[float], window: int) -> List[float]:
    """Trung binh truot (trailing) — diem i = mean cua toi da `window` diem ket thuc tai i."""
    if window <= 1:
        return list(vals)
    out, acc = [], 0.0
    for i, v in enumerate(vals):
        acc += v
        if i >= window:
            acc -= vals[i - window]
        out.append(acc / min(i + 1, window))
    return out


def _read_step_logs(jsonl_path):
    task = {"step": [], "lm": [], "lb": [], "tot": [], "acc": []}
    align = {"step": [], "loss": []}
    if not os.path.exists(jsonl_path):
        return task, align
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["step_type"] == "task":
                task["step"].append(rec["step"])
                task["lm"].append(rec["lm_loss"])
                task["lb"].append(rec["lb_loss"])
                task["tot"].append(rec["task_total_loss"])
                task["acc"].append(rec.get("token_acc") or 0.0)
            else:
                align["step"].append(rec["step"])
                align["loss"].append(rec["align_loss"])
    return task, align


def _line_with_smooth(ax, xs, ys, label, window, color=None):
    """Duong tho (mo) + duong trung binh truot (dam) cung 1 mau, tren cung 1 truc."""
    raw, = ax.plot(xs, ys, alpha=0.25, linewidth=0.8, color=color)
    ax.plot(xs, _moving_avg(ys, window), linewidth=1.8, color=raw.get_color(), label=label)


def plot_task_graph(jsonl_path, out_png, smooth_window: int):
    """GRAPH 1 — chi cac TASK step (step chan): L_LM, L_LB, L_task_total (tren) + token acc (duoi)."""
    task, _ = _read_step_logs(jsonl_path)
    if not task["step"]:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    _line_with_smooth(ax1, task["step"], task["lm"], "L_LM (SQuAD+SNLI+MMLU)", smooth_window)
    _line_with_smooth(ax1, task["step"], task["tot"], "L_task_total = L_LM + coef*L_LB", smooth_window)
    ax1.set_ylabel("Loss")
    ax1.set_title(f"Task steps (step chan) — dam = trung binh truot {smooth_window} diem, mo = tung step")
    ax1.grid(alpha=0.3)
    # L_LB co thang do rieng (~1) nen ve tren truc phai de khong lam det L_LM
    ax1b = ax1.twinx()
    raw, = ax1b.plot(task["step"], task["lb"], alpha=0.2, linewidth=0.8, color="tab:green")
    ax1b.plot(task["step"], _moving_avg(task["lb"], smooth_window), linewidth=1.5,
              color="tab:green", linestyle="--", label="L_LB (truc phai)")
    ax1b.set_ylabel("L_LB")
    # Gop legend cua 2 truc thanh 1 (tranh 2 legend chong len nhau)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.9)
    _line_with_smooth(ax2, task["step"], task["acc"], "token acc tren phan dap an", smooth_window)
    ax2.set_xlabel("Global training step"); ax2.set_ylabel("Token acc")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_align_graph(jsonl_path, out_png, align_layer, smooth_window: int):
    """GRAPH 2 — chi cac ALIGN step (step le): contrastive loss tai align_layer."""
    _, align = _read_step_logs(jsonl_path)
    if not align["step"]:
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    _line_with_smooth(ax, align["step"], align["loss"],
                      f"L_align (InfoNCE @ layer {align_layer})", smooth_window, color="tab:red")
    ax.set_xlabel("Global training step"); ax.set_ylabel("Loss")
    ax.set_title(f"Align steps (step le) — dam = trung binh truot {smooth_window} diem, mo = tung step")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_all(jsonl_path, task_png, align_png, align_layer, smooth_window):
    plot_task_graph(jsonl_path, task_png, smooth_window)
    plot_align_graph(jsonl_path, align_png, align_layer, smooth_window)


# ============================================================================================
# Hugging Face Hub push
# ============================================================================================
def build_model_card(args, tasks, task_counts, num_experts, top_k, align_layer_0based, num_layers,
                      lora_layers, n_align, steps_per_side, world_size) -> str:
    task_lines = "\n".join(f"  - {k}: {v} sample" for k, v in task_counts.items())
    shared = "attention / router / routed experts + shared expert" if args.include_shared_expert \
        else "attention / router / routed experts (khong co shared expert)"
    return f"""---
license: apache-2.0
base_model: {args.model_name_or_path}
tags:
- lora
- peft
- moe
- mixture-of-experts
- qwen
- cross-lingual-alignment
- midalign
---

# Qwen1.5-MoE-A2.7B-MidAlign (task = SNLI + SQuAD + MMLU)

LoRA adapter finetune tu `{args.model_name_or_path}`, Alternate Training: step chan = **task step** (giong het
finetuning english mix task), step le = **align step** (contrastive MidAlign, phan them vao).

## Task step (chan) — giong file finetuning english mix task
- `L_task = L_LM + lb_loss_coef * L_LB`, `L_LM` chi tinh tren phan dap an (prompt bi mask), chuan hoa theo tong token nhan toan cuc.
- Tasks: {", ".join(tasks)}
{task_lines}
- `lb_loss_coef` = {args.lb_loss_coef}, `num_experts` = {num_experts}, `top_k` = {top_k}
- batch co dinh {args.batch_size} sample / rank (token-budget = {args.max_tokens_per_batch or "tat"}), pool_size = {args.pool_size}.

## Align step (le)
- Contrastive Eq.(1) MidAlign ({args.align_loss}, negative trong mini-batch {args.align_micro_batch_size}) giua mean-pooled hidden state
  cau tieng Anh va cau target tai layer {args.align_layer} (block 0-indexed = {align_layer_0based} / {num_layers} layer), temperature = {args.align_temperature}.
- Pool nho {n_align} cap (nguon: {", ".join(args.alignment_data)}; {args.align_pairs_per_lang} cap/ngon ngu, seed {args.seed}), lap lai de du
  {steps_per_side * world_size * args.align_batch_size} cap align / epoch. global_negatives = {args.align_global_negatives},
  mask_false_negatives = {args.mask_false_negatives}, lang_balance = {args.align_lang_balance}.

## Lich
- Moi epoch: {steps_per_side} task step + {steps_per_side} align step. LR scheduler (linear warmup {args.warmup_ratio} -> 0) tien 1 buoc moi cap (task, align).
- world_size = {world_size}, task batch = {args.batch_size}/rank, align batch = {args.align_batch_size}/rank.

## LoRA
- Layer: `{format_layers(lora_layers)}` ({len(lora_layers)} layer, 0-indexed){' - chi dinh qua --layers' if args.layers else ' - mac dinh [L/3, 2L/3)'}, {shared}.
- Rank: attention {args.lora_r_attn}, router {args.lora_r_router}, experts {args.lora_r_expert}; alpha = {args.lora_alpha}, dropout = {args.lora_dropout}.

## Training
- {args.num_train_epochs} epoch, lr {args.learning_rate}, gradient all-reduce thu cong (khong DDP), checkpoint chi giu ban moi nhat.
- Diagnostics (`diagnostics/loss_log.jsonl`):

### Task steps (L_LM, L_LB, L_task_total, token acc)
![task loss](diagnostics/task_loss_curve.png)

### Align steps (contrastive loss)
![align loss](diagnostics/align_loss_curve.png)
"""


def push_to_hub(local_ckpt_dir, diagnostics_dir, hub_model_id, private, readme_text, token=None):
    if not HF_HUB_AVAILABLE:
        logger.warning("huggingface_hub chua duoc cai, bo qua buoc push_to_hub.")
        return
    api = HfApi(token=token)
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
# Main
# ============================================================================================
def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)
    # LoRA adapter giu fp32 (PEFT autocast_adapter_dtype) -> ~3000 matmul LoRA chay fp32; TF32 nhanh hon
    # nhieu tren Ampere+ va khong anh huong bf16 cua backbone.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    is_distributed, local_rank, rank, world_size, device = setup_distributed(args)
    is_main = (rank == 0)
    if not is_main:
        logger.setLevel(logging.WARNING)

    tasks = parse_tasks(args.tasks)
    args.data_files = resolve_data_files(args.alignment_data, args.data_files)
    logger.info(f"--alignment_data={args.alignment_data} -> data_files={args.data_files}")
    logger.info(f"Task mix (task step) = {tasks}")

    hf_token = load_hf_token(args.env_file, args.hf_token) if args.push_to_hub else None

    diagnostics_dir = args.diagnostics_dir or os.path.join(args.output_dir, "diagnostics")
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = os.path.join(diagnostics_dir, "loss_log.jsonl")
    task_plot_path = os.path.join(diagnostics_dir, "task_loss_curve.png")
    align_plot_path = os.path.join(diagnostics_dir, "align_loss_curve.png")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # ---------------------------------------------------------------------------------- tokenizer
    logger.info(f"Dang load tokenizer va model tu {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                               trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pad_id = int(tokenizer.pad_token_id)

    # ---------------------------------------------------------------------------------- model
    load_kwargs = dict(trust_remote_code=args.trust_remote_code, attn_implementation=args.attn_implementation)
    load_kwargs["dtype" if transformers_at_least("4.56.0") else "torch_dtype"] = dtype
    base_model = None
    if device.type == "cuda":
        # Load thang len GPU cua rank (khong qua RAM CPU: 2 rank x ~29GB bf16 neu qua CPU roi .to()).
        dev_idx = device.index if device.index is not None else torch.cuda.current_device()
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, device_map={"": dev_idx}, **load_kwargs)
        except (ImportError, ValueError) as e:
            logger.warning(f"Khong load truc tiep len GPU duoc ({e}) -> load qua CPU roi .to(device).")
    if base_model is None:
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    base_model.to(device)

    num_layers = get_num_layers(base_model.config)
    if args.layers:
        lora_layers = parse_layers_arg(args.layers, num_layers)
        layers_src = "chon qua --layers"
    else:
        # Mac dinh: [L/3, 2L/3) (exclusive o cuoi)
        lora_layers = list(range(num_layers // 3, (2 * num_layers) // 3))
        layers_src = "mac dinh [L/3, 2L/3)"
    layer_indices = set(lora_layers)

    if args.align_layer is None:
        args.align_layer = infer_align_layer_from_lora_range(lora_layers)
        logger.info(f"--align_layer khong duoc chi dinh -> lay layer GIUA khoang LoRA "
                    f"[{min(lora_layers)}, {max(lora_layers)}]: block {args.align_layer - 1} (0-indexed) "
                    f"-> align_layer = {args.align_layer} (num_hidden_layers={num_layers}).")
    if not (1 <= args.align_layer <= num_layers):
        raise ValueError(f"--align_layer={args.align_layer} phai nam trong [1, {num_layers}].")
    align_layer_0based = args.align_layer - 1

    logger.info(f"Tong so layer = {num_layers}. Align tai layer {args.align_layer} "
                f"(block 0-indexed {align_layer_0based}). LoRA tren layer {layers_src}: "
                f"{format_layers(lora_layers)} ({len(lora_layers)} layer).")
    if min(lora_layers) > align_layer_0based:
        logger.warning(f"[canh bao] moi layer LoRA ({format_layers(lora_layers)}) deu nam SAU align layer "
                       f"(block {align_layer_0based}) -> contrastive loss khong day gradient vao "
                       f"tham so LoRA nao.")
    elif align_layer_0based not in layer_indices:
        logger.warning(f"[canh bao] align layer (block {align_layer_0based}) khong thuoc cac layer LoRA "
                       f"({format_layers(lora_layers)}); chi cac layer LoRA <= block {align_layer_0based} "
                       f"nhan gradient tu contrastive loss.")

    target_modules = build_lora_target_modules(base_model, layer_indices, args.include_shared_expert)
    if not target_modules:
        raise RuntimeError("Khong tim thay module attention/router/experts nao trong range layer. "
                           "Kiem tra regex trong build_lora_target_modules().")
    attn_names, expert_names, router_names = categorize_lora_targets(target_modules, args.include_shared_expert)
    rank_pattern = build_rank_pattern(attn_names, expert_names, router_names,
                                       args.lora_r_attn, args.lora_r_expert, args.lora_r_router)
    logger.info(f"rank_pattern ({len(rank_pattern)} key regex): {rank_pattern}")
    logger.info(f"{len(target_modules)} target module LoRA: {len(attn_names)} attention "
                f"(r={args.lora_r_attn}), {len(expert_names)} experts (r={args.lora_r_expert}"
                f"{', gom shared_expert' if args.include_shared_expert else ', KHONG gom shared_expert'}), "
                f"{len(router_names)} router (r={args.lora_r_router}).")

    num_experts, top_k = infer_moe_dims(base_model.config, args)
    lb_loss_coef = args.lb_loss_coef
    if lb_loss_coef < 0:
        lb_loss_coef = float(getattr(base_model.config, "router_aux_loss_coef", 0.01))
    args.lb_loss_coef = lb_loss_coef
    logger.info(f"lb_loss_coef (task step) = {lb_loss_coef}")

    resume_dir = find_resume_checkpoint(args.output_dir, args.resume_from_checkpoint)
    if resume_dir:
        logger.info(f"Resume LoRA adapter tu checkpoint: {resume_dir}")
        model = PeftModel.from_pretrained(base_model, resume_dir, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
            rank_pattern=rank_pattern,
        )
        model = get_peft_model(base_model, lora_config)
    if is_main:
        model.print_trainable_parameters()
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_distributed:
        broadcast_trainable_params(trainable_params, src=0)  # dong bo init LoRA (khong con DDP)

    router_logits_cache: list = []
    hooks = register_router_hooks(model, router_names, router_logits_cache)

    # Decoder + lm_head de task step chi tinh logits tai cac vi tri co nhan (giong file finetuning).
    inner_lm = model.get_base_model()
    decoder = getattr(inner_lm, "model", None)
    if decoder is None and hasattr(inner_lm, "get_decoder"):
        decoder = inner_lm.get_decoder()
    if decoder is None:
        raise RuntimeError(f"Khong tim thay decoder trong {type(inner_lm).__name__}")
    lm_head = inner_lm.get_output_embeddings()

    # ------------------------------------------------------------------------------------ data
    cache_dir = None if args.no_cache else (args.cache_dir or os.path.join(args.output_dir, "cache"))

    def _cache_path(prefix: str, sig: dict) -> Optional[str]:
        if cache_dir is None:
            return None
        h = hashlib.md5(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return os.path.join(cache_dir, f"{prefix}_{h}.pkl")

    # ---- ALIGN pool (phan them vao)
    def _build_align():
        logger.info(f"Dang doc bitext {args.eng_key}-other tu {args.data_dir} ({args.data_files}) ...")
        prs, src = load_bitext_pairs(args.data_dir, args.data_files, args.eng_key,
                                      args.max_lang_pairs_per_record, args.seed,
                                      args.ted_pairs_per_record, args.align_pairs_per_lang)
        if args.max_samples and len(prs) > args.max_samples:
            sel = np.random.default_rng(args.seed).permutation(len(prs))[: args.max_samples]
            prs, src = [prs[i] for i in sel], src[sel]
        if not prs:
            raise RuntimeError("Khong doc duoc cap bitext nao — kiem tra --data_dir / --data_files / --eng_key.")
        return {"pairs": prs, "src": src}

    align_sig = {"v": 5, "ppl": args.align_pairs_per_lang, "ted_ppr": args.ted_pairs_per_record,
                 "files": file_signature([os.path.join(args.data_dir, f) for f in args.data_files]),
                 "eng_key": args.eng_key, "mlp": args.max_lang_pairs_per_record, "seed": args.seed,
                 "max_samples": args.max_samples}
    align_data = cached_build(_cache_path("align", align_sig), _build_align, is_main, is_distributed)
    pairs, align_src = align_data["pairs"], align_data["src"]
    del align_data

    # ---- TASK data: build_mixed_examples cua file finetuning (cung doc/prompt/windowing/tokenize/loc)
    def _build_task():
        logger.info(f"Dang chuan bi du lieu mix {tasks} (nhu file finetuning) ...")
        exs = build_mixed_examples(args, tokenizer, tasks)
        if len(exs) == 0:
            raise RuntimeError("Khong co sample task nao sau khi build du lieu.")
        return exs

    task_sig = {"v": DATA_FORMAT_VERSION, "tasks": tasks,
                "files": file_signature([getattr(args, TASK_REGISTRY[t]["file_arg"]) for t in tasks]),
                "max_len": args.task_max_length, "mspt": args.max_samples_per_task, "seed": args.seed,
                "tok": [tokenizer.name_or_path, len(tokenizer), tokenizer.eos_token]}
    examples = cached_build(_cache_path("task", task_sig), _build_task, is_main, is_distributed)

    lengths = np.fromiter((len(ex["input_ids"]) for ex in examples), dtype=np.int64, count=len(examples))
    label_tokens = np.fromiter((ex["n_lab"] for ex in examples), dtype=np.int64, count=len(examples))
    task_counts = {t: int(sum(1 for ex in examples if ex["task"] == i)) for i, t in enumerate(tasks)}
    logger.info("So sample moi task sau khi loc: " + ", ".join(f"{t}={c}" for t, c in task_counts.items())
                + f" | tong={len(examples)}")

    # ---- Lich: TASK sampler cua file finetuning quyet dinh N; ALIGN chay N step xen ke
    batch_sampler = TokenBudgetBatchSampler(
        lengths, label_tokens, max_tokens=args.max_tokens_per_batch, max_batch_size=args.max_batch_size,
        world_size=world_size, rank=rank, seed=args.seed, num_epochs=args.num_train_epochs,
        pool_size=args.pool_size, batch_size=args.batch_size,
    )
    N = batch_sampler.steps_per_epoch               # so task step / epoch (= so align step / epoch)
    steps_per_epoch = 2 * N                         # luon chan -> epoch nao cung bat dau bang task step
    total_task_steps = N * args.num_train_epochs    # == total_steps cua file finetuning (cung cau hinh)
    total_steps = steps_per_epoch * args.num_train_epochs

    n_align, align_bs = len(pairs), int(args.align_batch_size)
    if align_bs < 2:
        raise ValueError(f"--align_batch_size phai >= 2 (contrastive can >= 2 cap/rank; nhan duoc {align_bs}).")
    align_per_epoch = N * world_size * align_bs
    align_cover = align_per_epoch / n_align
    src_counts = np.bincount(align_src.astype(np.int64)).tolist() if len(align_src) else []
    batch_mode = (f"token-budget (max_tokens={args.max_tokens_per_batch})" if args.max_tokens_per_batch > 0
                  else f"batch co dinh {args.batch_size}/rank (toan cuc {args.batch_size * world_size})")
    logger.info(
        f"[schedule] TASK step = finetuning english mix: {len(examples)} sample, {batch_mode}, ~{batch_sampler.mean_batch_size:.1f} "
        f"sample/batch, padding eff ~{batch_sampler.padding_efficiency * 100:.1f}% | N={N} task step + {N} align step / epoch "
        f"(steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, task_steps_total={total_task_steps}) | "
        f"align: pool {n_align} cap (seed={args.seed}, {args.align_pairs_per_lang}/ngon ngu; theo file {args.data_files}: "
        f"{src_counts}), batch {align_bs}/rank, can {align_per_epoch} cap/epoch = {align_cover:.1f}x pool (LAP LAI).")
    if align_cover > 100:
        logger.warning(
            f"[canh bao] Moi cap align bi lap ~{align_cover:.0f} lan / epoch (~{align_cover * args.num_train_epochs:.0f} "
            f"lan trong {args.num_train_epochs} epoch) vi task step nho (--batch_size {args.batch_size}) cho nhieu step. "
            f"Neu align loss ve ~0 som (overfit) hoac align step qua cham: giam --align_batch_size, "
            f"tang --align_pairs_per_lang, hoac giam --num_train_epochs.")

    lang_ids = None
    if args.align_lang_balance:
        lang_names = sorted({p[2] for p in pairs})
        lmap = {l: i for i, l in enumerate(lang_names)}
        lang_ids = np.fromiter((lmap[p[2]] for p in pairs), dtype=np.int32, count=len(pairs))
        cnt = np.bincount(lang_ids, minlength=len(lang_names))
        logger.info(f"[align] resample DEU {len(lang_names)} ngon ngu: ~{align_per_epoch // len(lang_names)} "
                    f"cap/ngon ngu/epoch (pool: min={int(cnt.min())}, max={int(cnt.max())} cap/ngon ngu).")
    align_plan = AlignPlan(n_align, N, world_size, rank, args.seed, align_bs, lang_ids)

    # ------------------------------------------------------------------------------- optimizer
    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                   weight_decay=args.weight_decay,
                                   fused=True if use_fused else None, foreach=None if use_fused else True)
    # Giong file finetuning: linear warmup (3%) -> linear decay ve 0, tinh theo TASK step. scheduler.step() chi goi
    # 1 lan moi CAP (task k, align k) => task step k dung dung LR cua file finetuning o step k.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_task_steps * args.warmup_ratio),
        num_training_steps=total_task_steps,
    )

    start_epoch, start_step_in_epoch, global_step = 0, 0, 0
    prev_checkpoint_dir = resume_dir
    if resume_dir:
        state_path = os.path.join(resume_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            try:
                state = torch.load(state_path, map_location="cpu", weights_only=False)
            except TypeError:                      # torch cu khong co weights_only
                state = torch.load(state_path, map_location="cpu")
            if state.get("steps_per_epoch") not in (None, steps_per_epoch):
                raise RuntimeError(
                    f"Checkpoint co steps_per_epoch={state['steps_per_epoch']} nhung cau hinh hien "
                    f"tai cho {steps_per_epoch} (world_size/batch size/du lieu da doi?) -> khong the "
                    f"resume chinh xac.")
            optimizer.load_state_dict(state["optimizer"])
            for group in optimizer.param_groups:   # load_state_dict ghi de co fused/foreach cua checkpoint cu
                group["fused"] = True if use_fused else None
                group["foreach"] = None if use_fused else True
            if state.get("scheduler"):
                scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"]
            start_step_in_epoch = state["step_in_epoch"] + 1
            global_step = state["global_step"]
            logger.info(f"Da resume: epoch={start_epoch}, step_in_epoch={start_step_in_epoch}, "
                        f"global_step={global_step}")
            if start_step_in_epoch >= steps_per_epoch:
                start_epoch += 1
                start_step_in_epoch = 0

    # RNG rieng moi rank (dropout LoRA khac nhau giua cac rank)
    torch.manual_seed(args.seed + 7919 * rank + global_step)

    n_tasks = len(tasks)
    ctx = SimpleNamespace(
        decoder=decoder, lm_head=lm_head, router_cache=router_logits_cache,
        num_experts=num_experts, top_k=top_k, lb_loss_coef=lb_loss_coef,
        pad_id=pad_id, device=device, n_tasks=n_tasks, world_size=world_size,
        stats=torch.zeros(_N_BASE_STATS + 3 * n_tasks, dtype=torch.float32, device=device),
        global_tokens=1.0,
    )

    readme_text = build_model_card(args, tasks, task_counts, num_experts, top_k, align_layer_0based, num_layers,
                                    lora_layers, n_align, N, world_size)

    def _save_and_push(epoch_, step_in_epoch_, final=False):
        nonlocal prev_checkpoint_dir
        ckpt = save_checkpoint_and_rotate(
            args.output_dir, model, optimizer, scheduler, epoch_, step_in_epoch_, global_step,
            steps_per_epoch, world_size, prev_checkpoint_dir)
        prev_checkpoint_dir = ckpt
        plot_all(jsonl_path, task_plot_path, align_plot_path, args.align_layer, args.smooth_window)
        logger.info(f"Da luu checkpoint local: {ckpt}")
        if args.push_to_hub:
            try:
                push_to_hub(ckpt, diagnostics_dir, args.hub_model_id, args.hub_private,
                            readme_text, token=hf_token)
                logger.info(f"Da push checkpoint len hub: {args.hub_model_id}")
            except Exception as e:  # khong lam gian doan training
                logger.error(f"Push len hub that bai (checkpoint local van o {ckpt}): {e}")

    # ------------------------------------------------------------------------------ training loop
    last_done = (start_epoch, start_step_in_epoch - 1)  # (epoch, step_in_epoch) cuoi cung HOAN THANH
    jsonl_f = open(jsonl_path, "a", buffering=1, encoding="utf-8") if is_main else None
    model.train()
    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            step_offset = start_step_in_epoch if epoch == start_epoch else 0
            task_plan = batch_sampler.plan(epoch)            # N phan tu (batch_idx_cua_rank, global_label_tokens)
            align_batches = align_plan.epoch_batches(epoch)  # N mang chi so align cua rank nay

            pbar = tqdm(range(step_offset, steps_per_epoch), total=steps_per_epoch,
                        initial=step_offset, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
                        disable=not is_main)
            for step_in_epoch in pbar:
                model.train()
                optimizer.zero_grad(set_to_none=True)

                # Alternate Training: step chan = task, step le = align (steps_per_epoch chan)
                k = step_in_epoch // 2
                if step_in_epoch % 2 == 0:
                    # ======== TASK step: y het vong lap cua file finetuning ========
                    step_type = "task"
                    batch_idx, global_label_tokens = task_plan[k]
                    batch_examples = [examples[i] for i in batch_idx]
                    ctx.global_tokens = float(global_label_tokens)

                    run_batch_with_dynamic_oom_handling(batch_examples, ctx, optimizer, args.min_batch_size)

                    if is_distributed:
                        dist.all_reduce(ctx.stats, op=dist.ReduceOp.SUM)
                    result = stats_to_result(ctx.stats.tolist(), tasks, lb_loss_coef)
                    do_step = result["n_processed"] > 0  # giong nhau tren moi rank sau all-reduce
                    if do_step:
                        if is_distributed:
                            sync_grads_across_ranks(trainable_params, world_size, average=False)  # SUM
                    log_kwargs = dict(
                        lm_loss=result["lm_loss"], lb_loss=result["lb_loss"],
                        task_total_loss=result["total_loss"], token_acc=result["label_acc"],
                        align_loss=None, n_samples=result["n_processed"],
                        extra={"task_loss": result["task_loss"], "n_skipped": result["n_skipped"]})
                    postfix = {"type": "task", "L_LM": f"{result['lm_loss']:.4f}",
                               "L_LB": f"{result['lb_loss']:.4f}", "acc": f"{result['label_acc']:.3f}"}
                else:
                    # ======== ALIGN step (phan them vao) ========
                    step_type = "align"
                    batch = [pairs[i] for i in align_batches[k].tolist()]
                    eng_texts = [b[0] for b in batch]
                    other_texts = [b[1] for b in batch]
                    align_loss_val = align_step_accumulate(
                        eng_texts, other_texts, tokenizer, model, args.align_layer,
                        args.max_length, device, args.align_temperature, router_logits_cache,
                        micro_bs=args.align_micro_batch_size,
                        global_negatives=args.align_global_negatives and is_distributed,
                        mask_false_negatives=args.mask_false_negatives,
                        symmetric=(args.align_loss == "symmetric"))
                    router_logits_cache.clear()
                    sums = reduce_metrics([align_loss_val, 1.0, len(batch)], device, is_distributed)
                    do_step = True
                    if is_distributed:
                        sync_grads_across_ranks(trainable_params, world_size, average=True)  # MEAN
                    log_kwargs = dict(align_loss=sums[0] / sums[1], n_samples=int(sums[2]))
                    postfix = {"type": "align", "L_align": f"{log_kwargs['align_loss']:.4f}"}

                if do_step:
                    torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
                    optimizer.step()
                if step_in_epoch % 2 == 1:
                    scheduler.step()      # 1 buoc LR / cap (task k, align k)
                global_step += 1
                last_done = (epoch, step_in_epoch)

                if is_main:
                    if global_step % args.log_every == 0:
                        pbar.set_postfix(postfix)
                    if do_step:
                        log_step_to_jsonl(jsonl_f, global_step, epoch, step_type, **log_kwargs)
                    if args.plot_every and global_step % args.plot_every == 0:
                        plot_all(jsonl_path, task_plot_path, align_plot_path, args.align_layer,
                                 args.smooth_window)
                    if global_step % args.save_steps == 0:
                        _save_and_push(epoch, step_in_epoch)
                if is_distributed and global_step % args.save_steps == 0:
                    dist.barrier()  # cac rank cho rank 0 ghi checkpoint/push xong

        if is_main:
            _save_and_push(args.num_train_epochs - 1, steps_per_epoch - 1, final=True)
            logger.info("Training hoan tat.")
        if is_distributed:
            dist.barrier()

    except KeyboardInterrupt:
        logger.warning("Nhan KeyboardInterrupt — luu checkpoint khan cap truoc khi thoat ...")
        if is_main:
            _save_and_push(*last_done)
        raise
    finally:
        if jsonl_f is not None:
            jsonl_f.close()
        for h in hooks:
            h.remove()
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()