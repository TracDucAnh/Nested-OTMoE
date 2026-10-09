"""
MidAlign baseline cho mo hinh Mixture-of-Experts Qwen/Qwen1.5-MoE-A2.7B
(kien truc Qwen2MoeForCausalLM) — TASK STEP = LM loss tren SQuAD + SNLI + MMLU (English-only).

Adapt tu "Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned
LLMs" (Liu & Niehues, 2025), dung Alternate Training (Figure 2 trong paper): moi optimizer
step CHI toi uu MOT trong hai objective, xen ke theo step:

  - step CHAN (0, 2, 4, ...) = TASK step:
        L_task = L_LM (+ lb_loss_coef * L_LB, phu tro cho backbone MoE)
        L_LM  : cross-entropy chuan, CHI tinh tren phan dap an (<answer>/<label>/<letter><eos>),
                prompt bi mask -100 — DUNG cach doc du lieu + build prompt + mask cua cac file
                finetuning english-task-only (Qwen1.5-MoE-A2.7B-SQuAD/SNLI/MMLU.py):
                  SQuAD: "Context: ...\nQuestion: ...\nAnswer: <answer><eos>" (co context windowing)
                  SNLI : "Premise/Hypothesis/Question/Answer: <entailment|neutral|contradiction><eos>"
                  MMLU : "The following are multiple choice questions ... Answer: <letter><eos>"
        Ba tap duoc GOP THANH 1 POOL DUY NHAT, moi task step lay 1 batch tu pool nay.
  - step LE (1, 3, 5, ...) = ALIGN (contrastive) step:
        L_align = Eq.(1) paper (MOT CHIEU) giua mean-pooled hidden state cua cau tieng Anh va cau
        target (cap english-other), tai DUNG 1 layer (--align_layer, mac dinh TU DONG
        = layer GIUA cua khoang layer finetune LoRA: block (min+max)//2 cua --layers, vd --layers 7-14
        -> block 10 (0-indexed) = --align_layer 11; khong co --layers thi khoang [L/3, 2L/3)).

VAI TRO DATA (DUNG NHU PAPER): TASK LA ANCHOR, ALIGNMENT DATA NHO VA LAP LAI
-----------------------------------------------------------------------------
Paper: task data quyet dinh do dai training (toi da 5 epoch tren task data); alignment data chi la
"vai tram cau song song" / ngon ngu (Javanese 264, Swahili 371, Welsh 823 — Appendix B) va duoc
LAP LAI de bat kip so task step. Ban nay lam dung nhu vay:

  * TASK = anchor. N = ceil(n_task / (task_batch_size * world_size)) = so task step MOI epoch.
    Moi epoch la 1 hoan vi (seed, epoch) cua TOAN BO pool task -> MOI sample task duoc dung >= 1 lan /
    epoch (chi toi da task_batch_size*world_size - 1 sample bi dem them 1 lan de du batch cuoi).
  * ALIGN = pool NHO, lay NGAU NHIEN theo --seed (42): doc het flores + ntrex + ted (mac dinh), voi MOI
    ngon ngu gom tat ca cap eng-other tu ca 3 tap roi boc ngau nhien dung --align_pairs_per_lang cap
    (mac dinh 500 — cung bac "vai tram" cua paper; ngon ngu co it hon thi lay het) bang reservoir
    sampling seed co dinh -> pool tai lap duoc 100%.
  * Moi epoch co N align step xen ke voi N task step; moi align step can align_batch_size * world_size
    cap -> moi epoch can N * world_size * align_batch_size cap align, lay tu pool nho bang cach LAP LAI
    pool (nhieu luot hoan vi noi tiep). Neu --align_lang_balance (mac dinh bat, paper 4.3) thi moi ngon
    ngu nhan ~1/L tong so mau moi epoch (resample ve phan phoi xap xi deu).
  * Ke hoach batch xac dinh hoan toan theo (seed, epoch) nen resume giua epoch ra dung cac batch cu.

CAC DIEU CHINH KHAC SO VOI PAPER (giu nguyen tu ban truoc)
----------------------------------------------------------
  * Align loss = Eq.(1) MOT CHIEU: -log exp(sim(h_s,h_t)) / sum_{v in B} exp(sim(h_s,h_v)), sim = cosine,
    chia temperature (mac dinh 1.5 cho Qwen). --align_loss symmetric de quay lai InfoNCE doi xung.
  * Batch: --align_batch_size / --task_batch_size la PER-RANK (mac dinh 128); contrastive chi dung
    MINI-BATCH 32 (--align_micro_batch_size): negative CHI nam trong mini-batch 32 cap, gradient cong don
    qua 4 mini-batch roi moi optimizer.step() (footnote 11 paper). Voi world_size > 1 effective batch =
    128 * world_size (paper: 128 tong) — giam --*_batch_size neu muon khop effective batch cua paper.
  * LoRA: r=8, alpha=16, dropout=0.1, nhung van gioi han o range layer [L/3, 2L/3) theo yeu cau.
  * Toi uu: LR 5e-4, lich inverse-sqrt, warmup 0.03 (paper Appendix D.1).
  * Khong co early stopping (khong co dev set trong pipeline nay); dung --save_steps/--num_train_epochs.

Batch task cua moi rank tu dong chia thanh nhieu micro-batch (theo ngan sach token
--task_micro_batch_tokens, sort theo do dai de giam padding) va cong don gradient truoc
optimizer.step(). Contrastive step khong chia micro-batch duoc ngoai mini-batch (can in-batch negatives).

Dong bo gradient: all-reduce gradient thu cong (khong DDP). Sau backward, MOI tham so trainable
(ke ca LoRA cua expert khong nhan token nao trong step do -> grad None) deu duoc dien 0 roi
all-reduce 1 lan duy nhat -> khong con nguy co NCCL watchdog / "marked ready twice".

Cac dieu kien giu nguyen tu ban MidAlign truoc:
  1. LoRA mac dinh ap dung cho RANGE layer [L/3, 2L/3), tach bach voi layer tinh alignment loss.
     Co the chi dinh chinh xac cac layer can finetune bang --layers (vd: --layers 8-15 hoac
     --layers 4,5,6,10-12; chi so block tinh tu 0, khoang a-b dong ca 2 dau); khong truyen thi
     dung mac dinh [L/3, 2L/3). Attention / router / experts moi nhom 1 rank rieng.
  2. Cap ngon ngu english - other, doc tu du lieu multiway-parallel JSON (flores/ntrex/ted).
  3. Checkpoint chi giu ban moi nhat, push len HF Hub, resume tu checkpoint.

Vi du chay (8 GPU):
    torchrun --standalone --nproc_per_node=8 Qwen1.5-MoE-A2.7B.py \\
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \\
        --data_dir data/processed_alignment --alignment_data flores ntrex ted \\
        --align_pairs_per_lang 500 --seed 42 \\
        --squad_file data/english_task/squad/train.json \\
        --snli_file data/english_task/snli/train.json \\
        --mmlu_file data/english_task/mmlu/auxiliary_train.json \\
        --push_to_hub

Smoke test (1 GPU, du lieu nho):
    python Qwen1.5-MoE-A2.7B.py --max_task_samples 5000 --align_pairs_per_lang 50 --no_push_to_hub

Resume:
    torchrun --standalone --nproc_per_node=8 Qwen1.5-MoE-A2.7B.py --resume_from_checkpoint auto
(Resume yeu cau giu nguyen world_size, batch size va du lieu de N khong doi — se bao loi neu lech.)
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
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
        description="MidAlign (alternate task-LM[SQuAD+SNLI+MMLU] / contrastive align) cho Qwen1.5-MoE-A2.7B"
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

    # Du lieu TASK (task step, LM loss) — doc giong het cac file finetuning english-task-only
    p.add_argument("--task_datasets", type=str, nargs="+", choices=["squad", "snli", "mmlu"],
                    default=["squad", "snli", "mmlu"],
                    help="Cac tap duoc gop vao pool task step.")
    p.add_argument("--squad_file", type=str, default="data/english_task/squad/train.json")
    p.add_argument("--snli_file", type=str, default="data/english_task/snli/train.json")
    p.add_argument("--mmlu_file", type=str, default="data/english_task/mmlu/auxiliary_train.json")
    p.add_argument("--max_task_samples", type=int, default=None,
                    help="Gioi han so sample MOI tap task (debug), None = dung het.")

    # Hugging Face Hub
    p.add_argument("--push_to_hub", action="store_true", default=True)
    p.add_argument("--no_push_to_hub", dest="push_to_hub", action="store_false")
    p.add_argument("--hub_model_id", type=str, default="ducanhdinh/Qwen1.5-MoE-A2.7B-MidAlign")
    p.add_argument("--hub_private", action="store_true")
    p.add_argument("--env_file", type=str, default=".env")
    p.add_argument("--hf_token", type=str, default=None)

    # Training schedule + batch (xem docstring dau file ve cach tinh so step)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--align_batch_size", type=int, default=128,
                    help="EFFECTIVE batch size align per-rank (paper: 128). Duoc chia thanh mini-batch "
                         "--align_micro_batch_size de tinh contrastive loss + cong don gradient. Moi epoch "
                         "can N * world_size * align_batch_size cap align (N tinh tu TASK); pool align nho "
                         "duoc LAP LAI de du so cap nay (dung nhu paper).")
    p.add_argument("--task_batch_size", type=int, default=128,
                    help="Batch size task per-rank (so sample task / rank / task step). TASK LA ANCHOR: "
                         "N = ceil(n_task / (task_batch_size * world_size)) = so task step = so align step "
                         "moi epoch; MOI sample task duoc dung >= 1 lan / epoch.")
    p.add_argument("--align_micro_batch_size", type=int, default=32,
                    help="Kich thuoc MINI-BATCH contrastive (paper: 32). Negative chi nam trong mini-batch; "
                         "gradient cong don qua cac mini-batch cua 1 align step. 0 = khong chia.")
    p.add_argument("--task_micro_batch_tokens", type=int, default=16384,
                    help="Ngan sach token (so sample * do dai da padding) cho 1 micro-batch cua task "
                         "step. Giam neu OOM, tang neu con du VRAM.")
    p.add_argument("--max_length", type=int, default=256,
                    help="max_length cho cau alignment (contrastive step).")
    p.add_argument("--task_max_length", type=int, default=512,
                    help="max_length cho sample task (SQuAD can 512; sample SQuAD dai hon duoc "
                         "windowing, MMLU/SNLI dai hon bi bo).")
    p.add_argument("--learning_rate", type=float, default=5e-4)
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
    p.add_argument("--lb_loss_coef", type=float, default=None,
                    help="He so load-balancing loss. None = config.router_aux_loss_coef, fallback "
                         "0.01. Dat 0 de task loss la LM thuan tuy.")
    p.add_argument("--num_local_experts", type=int, default=None)
    p.add_argument("--num_experts_per_tok", type=int, default=None)

    # LoRA
    p.add_argument("--lora_r", type=int, default=8, help="Rank fallback (paper: 8).")
    p.add_argument("--lora_r_router", type=int, default=8)
    p.add_argument("--lora_r_attn", type=int, default=8)
    p.add_argument("--lora_r_expert", type=int, default=8,
                    help="Rank cho gate/up/down_proj cua experts VA shared_expert.")
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.1)
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


def sync_grads_across_ranks(trainable_params: List[torch.Tensor], world_size: int):
    """All-reduce (trung binh) gradient THU CONG, gop thanh 1 buffer lien tuc theo dtype.
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


_BACKBONE_CACHE: Dict[int, Tuple] = {}


def get_backbone_and_head(model):
    """(backbone, lm_head) cua CausalLM goc ben duoi PeftModel -> chay backbone roi CHI nhan lm_head
    tren cac vi tri co nhan (phan dap an). (None, None) neu khong tim thay (se quay ve forward day du)."""
    key = id(model)
    if key not in _BACKBONE_CACHE:
        m = get_underlying_model(model)
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
        backbone, head = getattr(m, "model", None), getattr(m, "lm_head", None)
        _BACKBONE_CACHE[key] = (backbone, head) if (backbone is not None and head is not None) else (None, None)
    return _BACKBONE_CACHE[key]


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
# Doc du lieu TASK — COPY NGUYEN VAN tu cac file finetuning english-task-only
# (chi doi ten build_prompt/build_full_text/build_examples them tien to squad_/snli_/mmlu_ de khong trung ten)
# ============================================================================================
WORD_SPAN_PATTERN = re.compile(r"\S+")
CHOICE_LETTERS = string.ascii_uppercase


LABEL_TO_WORD = {0: "entailment", 1: "neutral", 2: "contradiction"}


# ---------------------------------------- SQuAD ----------------------------------------
def load_squad_records(data_file: str) -> List[Dict]:
    """Doc file JSON dang list cac object SQuAD chuan:
        {"id", "title", "context", "question",
         "answers": {"text": [...], "answer_start": [...]}}
    Bo qua record thieu field, khong co dap an (SQuAD 2.0 "unanswerable", answers.text rong —
    khong phu hop voi muc tieu zero-shot tren cac benchmark luon-co-dap-an nhu XQuAD), hoac
    answer_start khong khop voi noi dung context (loi du lieu)."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Khong tim thay file du lieu SQuAD: {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
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

        # SQuAD train thuong chi co 1 dap an/cau hoi; neu co nhieu, lay dap an dau tien de train
        # (giong quy uoc pho bien khi finetune generative QA tren SQuAD).
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
        # Kiem tra answer_start co thuc su tro dung vao answer_text trong context khong (an
        # toan du lieu — mot so file SQuAD-like bi lech offset do tien xu ly khac nhau).
        span = context[answer_start: answer_start + len(answer_text)]
        if span.strip() != answer_text.strip():
            # Thu tim lai vi tri chinh xac trong context (fallback) truoc khi bo qua han.
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
        f"Da doc {len(records)} sample hop le tu {data_file} "
        f"(bo qua {n_skipped_missing} record thieu field, "
        f"{n_skipped_unanswerable} cau hoi khong co dap an, "
        f"{n_skipped_mismatch} record lech offset answer_start/context)."
    )
    return records


def squad_build_prompt(context: str, question: str) -> str:
    """Prompt dang instruction cho extractive QA. Phan sau 'Answer:' la phan model phai sinh ra
    va la phan DUY NHAT duoc tinh loss (xem squad_build_full_text + mask trong
    forward_backward_one_subbatch). Dung format nay de tuong thich truc tiep voi cach eval
    zero-shot pho bien cho SQuAD/XQuAD (prompt giong het khi generate o eval, chi bo phan dap an)."""
    return (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def squad_build_full_text(context: str, question: str, answer_text: str, eos_token: str) -> Tuple[str, str]:
    prompt = squad_build_prompt(context, question)
    full_text = f"{prompt} {answer_text}{eos_token}"
    return prompt, full_text


def select_context_window(context: str, answer_start: int, answer_text: str,
                           tokenizer, max_context_tokens: int) -> str:
    """Khi full_text vuot qua --max_length, KHONG truncate tho tu tokenizer (se cat mat phan
    'Answer: ...' nam o cuoi chuoi), ma chon 1 CUA SO cac TU trong context BAO QUANH vi tri
    cua answer (dua vao answer_start), roi mo rong dan sang trai/phai (giu nguyen tung tu) cho
    toi khi vua sat ngan sach max_context_tokens. Nho vay context van luon chua answer.

    TOI UU: ban cu tokenize lai ca cua so sau MOI lan mo rong 1 tu (O(so_tu) lan goi tokenizer /
    sample, ~50 giay cho 213 sample SQuAD). Gio: tokenize tung tu 1 lan (batch) -> tong tich luy ->
    mo rong bang uoc luong O(1); roi KIEM TRA lai bang tokenizer that va thu hep neu uoc luong
    thap hon thuc te, nen ket qua van dam bao <= max_context_tokens."""
    spans = [m.span() for m in WORD_SPAN_PATTERN.finditer(context)]
    if not spans:
        return context

    answer_end = answer_start + len(answer_text)
    left_idx, right_idx = None, None
    for i, (s_, e_) in enumerate(spans):
        if e_ > answer_start and left_idx is None:
            left_idx = i
        if s_ < answer_end:
            right_idx = i
    if left_idx is None or right_idx is None:
        left_idx, right_idx = 0, 0
    lo, hi = left_idx, right_idx

    def window_text(lo, hi):
        return context[spans[lo][0]: spans[hi][1]]

    def token_len(t: str) -> int:
        return len(tokenizer(t, add_special_tokens=False)["input_ids"])

    cur_text = window_text(lo, hi)
    if token_len(cur_text) > max_context_tokens:
        # Ngay ca cua so toi thieu (chi vua du cac tu cua answer) da vuot ngan sach -> tra ve
        # nguyen trang, ham goi se tu phat hien full_text van qua dai va skip sample nay.
        return cur_text

    words = [" " + context[a_:b_] for a_, b_ in spans]
    wl = tokenizer(words, add_special_tokens=False)["input_ids"]
    cum = [0]
    for ids in wl:
        cum.append(cum[-1] + len(ids))

    def est(lo, hi):
        return cum[hi + 1] - cum[lo]

    while True:
        moved = False
        if lo > 0 and est(lo - 1, hi) <= max_context_tokens:
            lo -= 1
            moved = True
        if hi < len(spans) - 1 and est(lo, hi + 1) <= max_context_tokens:
            hi += 1
            moved = True
        if not moved:
            break

    # Kiem tra bang tokenizer that (uoc luong theo tung tu co the lech vai token); thu hep neu can.
    while (lo, hi) != (left_idx, right_idx) and token_len(window_text(lo, hi)) > max_context_tokens:
        if lo < left_idx and (hi == right_idx or (left_idx - lo) >= (hi - right_idx)):
            lo += 1
        else:
            hi -= 1
    return window_text(lo, hi)


def compute_token_ids(tokenizer, texts: Sequence[str], chunk_size: int = 4000,
                       desc: str = "Tokenize") -> List[np.ndarray]:
    """Tokenize batch (fast tokenizer) -> list np.int32. GIU LAI ids (khong chi do dai) de luc train
    khong phai tokenize lai tung step. Khong tao attention_mask / token_type_ids (khong dung)."""
    out: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), chunk_size), desc=desc):
        enc = tokenizer(list(texts[i:i + chunk_size]), add_special_tokens=True,
                        return_attention_mask=False)["input_ids"]
        out.extend(np.asarray(x, dtype=np.int32) for x in enc)
    return out


def squad_build_examples(records: List[Dict], tokenizer, eos_token: str, max_length: int) -> Tuple[List[Dict], List[int]]:
    """Tien xu ly: moi record -> {"prompt", "full_text", "answer_text"}.
    Sample nao co full_text vuot qua max_length se duoc "windowing" lai context (xem
    select_context_window); neu van khong vua sau khi windowing (hiem) thi bi bo qua.
    Tra ve (examples, lengths) da loc, dong bo index voi nhau — dung truc tiep cho
    LengthGroupedBatchSampler, tranh phai tokenize lai toan bo lan nua."""
    # Buoc 1: build naive (chua windowing) cho toan bo, tinh do dai token 1 lan (batch, nhanh).
    naive_examples = []
    for rec in records:
        prompt, full_text = squad_build_full_text(rec["context"], rec["question"], rec["answer_text"], eos_token)
        naive_examples.append({
            "prompt": prompt,
            "full_text": full_text,
            "answer_text": rec["answer_text"],
            "context": rec["context"],
            "question": rec["question"],
            "answer_start": rec["answer_start"],
        })
    naive_ids = compute_token_ids(tokenizer, [ex["full_text"] for ex in naive_examples],
                                  desc="SQuAD: tokenize")
    naive_lengths = [len(x) for x in naive_ids]

    # Buoc 2: chi ap dung windowing (co the cham hon, goi tokenizer nhieu lan) cho phan THIEU SO
    # sample vuot qua max_length — da so sample SQuAD se vua trong 1 lan, khong can qua buoc nay.
    examples: List[Dict] = []
    lengths: List[int] = []
    n_windowed = 0
    n_dropped_too_long = 0
    for ex, naive_len, naive_id in zip(naive_examples, naive_lengths, naive_ids):
        if naive_len <= max_length:
            examples.append({"prompt": ex["prompt"], "full_text": ex["full_text"], "ids": naive_id})
            lengths.append(naive_len)
            continue

        # Ngan sach token danh cho context = max_length tru phan "Context: \nQuestion: ...\n
        # Answer: <answer><eos>" (moi thu tru context), tru them margin an toan cho cac dac thu
        # tokenization (BOS/khoang trang noi tu ...).
        overhead_text = f"Context: \nQuestion: {ex['question']}\nAnswer: {ex['answer_text']}{eos_token}"
        overhead_tokens = len(tokenizer(overhead_text, add_special_tokens=True)["input_ids"])
        max_context_tokens = max_length - overhead_tokens - 8
        if max_context_tokens <= 0:
            n_dropped_too_long += 1
            continue

        windowed_context = select_context_window(
            ex["context"], ex["answer_start"], ex["answer_text"], tokenizer, max_context_tokens
        )
        new_prompt, new_full_text = squad_build_full_text(windowed_context, ex["question"], ex["answer_text"], eos_token)
        new_ids = np.asarray(tokenizer(new_full_text, add_special_tokens=True)["input_ids"], dtype=np.int32)
        new_len = len(new_ids)
        if new_len > max_length:
            n_dropped_too_long += 1
            continue

        examples.append({"prompt": new_prompt, "full_text": new_full_text, "ids": new_ids})
        lengths.append(new_len)
        n_windowed += 1

    logger.info(
        f"squad_build_examples: {len(examples)} sample giu lai (trong do {n_windowed} sample da duoc "
        f"windowing context vi vuot max_length={max_length}), bo qua {n_dropped_too_long} sample "
        f"van qua dai ngay ca sau khi windowing."
    )
    return examples, lengths


# ---------------------------------------- SNLI -----------------------------------------
def load_snli_records(data_file: str) -> List[Dict]:
    """Doc file JSON dang list cac object {"premise": ..., "hypothesis": ..., "label": 0/1/2}.
    Bo qua record thieu field, hoac label khong nam trong {0, 1, 2} (SNLI goc dung -1 cho
    cac cau khong dong thuan giua annotator)."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Khong tim thay file du lieu SNLI: {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
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
    logger.info(f"Da doc {len(records)} sample hop le tu {data_file} (bo qua {n_skipped} record loi/label khong hop le).")
    return records


def snli_build_prompt(premise: str, hypothesis: str) -> str:
    """Prompt dang instruction cho NLI. Phan sau 'Answer:' la phan model phai sinh ra va la
    phan DUY NHAT duoc tinh loss (xem snli_build_full_text + mask trong forward_backward_one_subbatch)."""
    return (
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        f"Question: What is the relationship between the premise and the hypothesis? "
        f"Choose one: entailment, neutral, or contradiction.\n"
        f"Answer:"
    )


def snli_build_full_text(premise: str, hypothesis: str, label: int, eos_token: str) -> Tuple[str, str]:
    prompt = snli_build_prompt(premise, hypothesis)
    label_word = LABEL_TO_WORD[label]
    full_text = f"{prompt} {label_word}{eos_token}"
    return prompt, full_text


def snli_build_examples(records: List[Dict], eos_token: str) -> List[Dict]:
    """Tien xu ly 1 lan: moi record -> {"prompt", "full_text", "label", "label_word"}.
    Tranh phai build lai chuoi prompt/full_text moi lan __getitem__/moi epoch."""
    examples = []
    for rec in records:
        prompt, full_text = snli_build_full_text(rec["premise"], rec["hypothesis"], rec["label"], eos_token)
        examples.append({
            "prompt": prompt,
            "full_text": full_text,
            "label": rec["label"],
            "label_word": LABEL_TO_WORD[rec["label"]],
        })
    return examples


# ---------------------------------------- MMLU -----------------------------------------
def load_mmlu_records(data_file: str) -> List[Dict]:
    """Doc file JSON dang list cac object MMLU chuan:
        {"question": str, "choices": [str, ...], "answer": int, "subject": str}
    Bo qua record thieu field, choices khong hop le (khong phai list, < 2 phan tu), hoac answer
    khong nam trong khoang [0, len(choices)-1]."""
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Khong tim thay file du lieu MMLU: {data_file}")
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
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
        if not (0 <= answer < len(choices)):
            n_skipped_bad_answer += 1
            continue
        if len(choices) > len(CHOICE_LETTERS):
            n_skipped_bad_answer += 1
            continue

        records.append({
            "question": str(question).strip(),
            "choices": [str(c).strip() for c in choices],
            "answer": answer,
            "subject": str(subject).strip() if subject else "",
        })
    logger.info(
        f"Da doc {len(records)} sample hop le tu {data_file} "
        f"(bo qua {n_skipped_missing} record thieu field, "
        f"{n_skipped_bad_answer} record answer/choices khong hop le)."
    )
    return records


def mmlu_build_prompt(question: str, choices: List[str], subject: str) -> str:
    """Prompt dang trac nghiem, DUNG CHUAN format pho bien khi eval MMLU (lm-evaluation-harness
    / paper goc), de tuong thich truc tiep voi zero-shot eval tren MMLU/MMMLU. Phan sau
    'Answer:' la phan model phai sinh ra va la phan DUY NHAT duoc tinh loss (xem mmlu_build_full_text
    + mask trong forward_backward_one_subbatch)."""
    if subject:
        header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    else:
        header = "The following are multiple choice questions (with answers).\n\n"
    choice_lines = "\n".join(f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{header}{question}\n{choice_lines}\nAnswer:"


def mmlu_build_full_text(question: str, choices: List[str], subject: str, answer_idx: int,
                     eos_token: str) -> Tuple[str, str]:
    prompt = mmlu_build_prompt(question, choices, subject)
    letter = CHOICE_LETTERS[answer_idx]
    full_text = f"{prompt} {letter}{eos_token}"
    return prompt, full_text


def mmlu_build_examples(records: List[Dict], tokenizer, eos_token: str, max_length: int) -> Tuple[List[Dict], List[int]]:
    """Tien xu ly 1 lan: moi record -> {"prompt", "full_text"}. Khac SQuAD (khong co "context"
    dai can windowing) — MMLU prompt thuong ngan, sample nao (hiem) vuot max_length se bi BO QUA
    hoan toan (KHONG truncate tho, vi truncation se cat mat dung phan "Answer: <letter>" o cuoi
    chuoi). Tra ve (examples, lengths) da loc, dong bo index — dung truc tiep cho
    LengthGroupedBatchSampler."""
    full_texts, prompts = [], []
    for rec in records:
        prompt, full_text = mmlu_build_full_text(rec["question"], rec["choices"], rec["subject"],
                                             rec["answer"], eos_token)
        prompts.append(prompt)
        full_texts.append(full_text)

    ids_all = compute_token_ids(tokenizer, full_texts, desc="MMLU: tokenize")

    examples: List[Dict] = []
    lengths: List[int] = []
    n_dropped_too_long = 0
    for prompt, full_text, ids in zip(prompts, full_texts, ids_all):
        if len(ids) > max_length:
            n_dropped_too_long += 1
            continue
        examples.append({"prompt": prompt, "full_text": full_text, "ids": ids})
        lengths.append(len(ids))

    logger.info(
        f"mmlu_build_examples: {len(examples)} sample giu lai, bo qua {n_dropped_too_long} sample "
        f"vuot qua max_length={max_length} (khong truncate de tranh cat mat nhan)."
    )
    return examples, lengths


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


def is_router_leaf_name(name: str) -> bool:
    leaf = name.split(".")[-1]
    return leaf in ("gate", "router", "gating") and not is_expert_name(name)


def is_expert_name(name: str) -> bool:
    """Linear projection cua routed experts (mlp.experts.N.*) HOAC shared expert (mlp.shared_expert.*).
    (ban truoc chi bat '.experts.'/'.expert.' nen BO SOT shared_expert cua Qwen1.5-MoE.)"""
    return (".experts." in name or ".expert." in name or ".shared_expert." in name
            or ".shared_experts." in name)


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
        is_expert = is_expert_name(name)
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


def categorize_lora_targets(target_modules: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """Chia danh sach target_modules (da duoc build_lora_target_modules loc) thanh 3 nhom rieng
    biet — attention / experts / router — dung LAI CHINH XAC cung dieu kien nhu trong
    build_lora_target_modules(), de dam bao khop 1-1 voi target_modules truyen vao LoraConfig.
    Dung cho rank_pattern (dieu kien thay doi #2: moi nhom co the mang 1 rank LoRA rieng)."""
    attn_names, expert_names, router_names = [], [], []
    for name in target_modules:
        if is_router_leaf_name(name):
            router_names.append(name)
        elif is_expert_name(name):
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
    """attention_mask: [batch, seq_len] (1 = token that, 0 = padding). Padding bi loai bang TRONG SO
    (0/1) thay vi boolean-index logits[mask] (cach cu moi lan goi gay 1 device sync / layer). Ket qua
    bang het: mean tren cac token that cua [so token chon expert] va [xac suat router]."""
    w_full = attention_mask.reshape(-1).to(torch.float32)

    losses = []
    for logits in router_logits_list:
        logits = logits.reshape(-1, logits.shape[-1]).float()
        if logits.shape[0] == w_full.shape[0]:
            w = w_full
        else:
            logger.warning(
                "compute_load_balancing_loss: kich thuoc router logits "
                f"({logits.shape[0]}) khong khop attention_mask ({w_full.shape[0]}) -> "
                "bo qua loc padding cho lan tinh nay."
            )
            w = torch.ones(logits.shape[0], device=logits.device, dtype=torch.float32)
        n_valid = w.sum().clamp(min=1.0)
        routing_weights = F.softmax(logits, dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        expert_mask = torch.zeros_like(routing_weights).scatter_(1, selected_experts, 1.0)   # [T, E]
        tokens_per_expert = (expert_mask * w[:, None]).sum(dim=0) / n_valid
        avg_prob_per_expert = (routing_weights * w[:, None]).sum(dim=0) / n_valid
        losses.append(num_experts * torch.sum(tokens_per_expert * avg_prob_per_expert))
    if not losses:
        return torch.tensor(0.0, device=attention_mask.device)
    return torch.stack(losses).mean()


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
# Task pool: gop SQuAD + SNLI + MMLU thanh 1 danh sach example duy nhat {"prompt","full_text","src"}
# (doc/build prompt/windowing/loc do dai bang CHINH cac ham cua cac file finetuning goc)
# ============================================================================================
def build_task_pool(args, tokenizer) -> Dict:
    eos = tokenizer.eos_token
    rng = random.Random(args.seed)
    examples: List[Dict] = []
    lengths: List[int] = []
    stats: Dict[str, Dict[str, int]] = {}

    def _cap(records):
        if args.max_task_samples and len(records) > args.max_task_samples:
            records = list(records)
            rng.shuffle(records)
            records = records[: args.max_task_samples]
        return records

    src_order: List[str] = []

    def _add(src: str, exs: List[Dict], lens: List[int], n_loaded: int):
        for ex, ln in zip(exs, lens):
            examples.append({"prompt": ex["prompt"], "ids": ex["ids"], "src": len(src_order)})
            lengths.append(ln)
        src_order.append(src)
        stats[src] = {"loaded": n_loaded, "kept": len(exs), "dropped": n_loaded - len(exs)}

    if "squad" in args.task_datasets:
        recs = _cap(load_squad_records(args.squad_file))
        exs, lens = squad_build_examples(recs, tokenizer, eos, args.task_max_length)
        _add("squad", exs, lens, len(recs))
    if "snli" in args.task_datasets:
        recs = _cap(load_snli_records(args.snli_file))
        all_exs = snli_build_examples(recs, eos)
        all_ids = compute_token_ids(tokenizer, [e["full_text"] for e in all_exs], desc="SNLI: tokenize")
        keep = []
        for e, ids in zip(all_exs, all_ids):
            if len(ids) <= args.task_max_length:
                e["ids"] = ids
                keep.append(e)
        _add("snli", keep, [len(e["ids"]) for e in keep], len(recs))
    if "mmlu" in args.task_datasets:
        recs = _cap(load_mmlu_records(args.mmlu_file))
        exs, lens = mmlu_build_examples(recs, tokenizer, eos, args.task_max_length)
        _add("mmlu", exs, lens, len(recs))

    for src, st in stats.items():
        logger.info(f"[task pool] {src}: doc {st['loaded']}, giu {st['kept']}, "
                    f"bo (qua dai khong the cat) {st['dropped']}")

    # Dong goi: 1 mang phang int32 + offsets (gon, pickle nhanh) + do dai PROMPT (de mask nhan).
    # prompt_len = so token cua prompt tokenize RIENG (dung quy uoc cu: mask prompt = -100).
    # Tinh 1 lan o day (duoc cache) thay vi tokenize lai prompt cua tung sample moi step.
    prompt_ids = compute_token_ids(tokenizer, [e["prompt"] for e in examples], desc="Tokenize prompt")
    prompt_len = np.asarray([len(x) for x in prompt_ids], dtype=np.int32)
    del prompt_ids
    lens_np = np.asarray(lengths, dtype=np.int64)
    offs = np.zeros(len(examples) + 1, dtype=np.int64)
    np.cumsum(lens_np, out=offs[1:])
    flat = np.empty(int(offs[-1]), dtype=np.int32)
    for i, e in enumerate(examples):
        flat[offs[i]:offs[i + 1]] = e["ids"]
    pool = {
        "flat": flat, "offs": offs, "lengths": lens_np.astype(np.int32),
        "prompt_len": np.minimum(prompt_len, lens_np.astype(np.int32)),
        "src": np.asarray([e["src"] for e in examples], dtype=np.uint8),
        "src_names": src_order, "stats": stats,
    }
    return pool


# ============================================================================================
# Lich xen ke task/align: ALIGN la anchor — moi epoch = N align step dung HET pool align + N task step
# (task data duoc phep lap lai)
# ============================================================================================
def solve_schedule(n_task: int, world_size: int, task_batch_size: int) -> int:
    """Tra ve N = so task step = so align step MOI epoch. TASK la anchor (dung nhu paper):
        N = max(1, ceil(n_task / (task_batch_size * world_size)))
    -> moi sample task duoc dung >= 1 lan / epoch. Align data (pool nho) khong tham gia chon N;
    no duoc LAP LAI de du N align step."""
    return max(1, -(-int(n_task) // (int(task_batch_size) * int(world_size))))


def split_sizes(total: int, parts: int) -> List[int]:
    """Chia `total` thanh `parts` phan chenh nhau toi da 1 (phan dau nhan them 1 neu du)."""
    base, rem = divmod(total, parts)
    return [base + 1 if i < rem else base for i in range(parts)]


class AlternatePlan:
    """Ke hoach batch cho 1 epoch (xac dinh hoan toan theo (seed, epoch) -> resume giua epoch ra dung
    cac batch cu). Dung numpy.

    TASK (anchor): hoan vi (seed, epoch) TOAN BO n_task sample, cat thanh N lat, moi lat
    W * task_batch_size sample (phan thieu o lat cuoi, < 1 step, duoc lap vong tu dau hoan vi) -> MOI sample
    task xuat hien >= 1 lan / epoch. Moi lat sort theo do dai TRUOC khi chia rank (can bang tai + thuan loi
    cho micro-batching theo token).

    ALIGN (pool nho, lap lai nhu paper): moi epoch can align_per_epoch = N * W * align_batch_size cap. Luong cap
    nay duoc sinh tu pool n_align (nho) bang cach lap lai pool:
      - align_lang_balance (lang_ids != None, paper 4.3): moi ngon ngu nhan ~align_per_epoch / L cap, trong ngon
        ngu do cac cap duoc quay vong qua nhieu hoan vi noi tiep (deu nhau: moi cap xuat hien floor hoac
        ceil lan) -> phan phoi ngon ngu xap xi DEU;
      - nguoc lai: nhieu hoan vi noi tiep cua ca pool.
    Moi lat align = W * align_batch_size cap, chia rank theo buoc W -> moi rank dung align_batch_size cap
    (all_gather cua contrastive loss can cung shape tren moi rank)."""

    def __init__(self, n_task, n_align, task_lengths, steps_per_side, world_size, rank, seed,
                 task_batch_size, align_batch_size, lang_ids=None):
        self.n_task, self.n_align = n_task, n_align
        self.lang_ids = None if lang_ids is None else np.asarray(lang_ids)
        self._lang_groups = None
        if self.lang_ids is not None:
            order = np.argsort(self.lang_ids, kind="stable")
            _, starts = np.unique(self.lang_ids[order], return_index=True)
            self._lang_groups = np.split(order, starts[1:])
        self.task_lengths = np.asarray(task_lengths)
        self.N, self.W, self.rank, self.seed = steps_per_side, world_size, rank, seed
        self.task_per_step = task_batch_size * world_size              # sample task TOAN CUC / task step
        self.task_per_epoch = self.task_per_step * steps_per_side
        self.align_per_step = align_batch_size * world_size            # cap align TOAN CUC / align step
        self.align_per_epoch = self.align_per_step * steps_per_side    # M

    def _align_epoch_stream(self, epoch: int) -> np.ndarray:
        """Day M = align_per_epoch chi so align (lap lai pool nho). Co lang_ids: phan phoi DEU giua cac ngon ngu."""
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

    def epoch_batches(self, epoch: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # ---- task (anchor): 1 hoan vi cua TOAN BO pool, N lat bang nhau
        t_perm = np.random.default_rng([self.seed, epoch, 1]).permutation(self.n_task)
        t_perm = np.resize(t_perm, self.task_per_epoch)      # lap vong phan thieu (< 1 task step)
        task_batches, align_batches = [], []
        for k in range(self.N):
            chunk = t_perm[k * self.task_per_step:(k + 1) * self.task_per_step]
            chunk = chunk[np.argsort(self.task_lengths[chunk], kind="stable")]
            task_batches.append(chunk[self.rank::self.W])
        # ---- align: pool nho, lap lai de du N align step
        a_stream = self._align_epoch_stream(epoch)
        for k in range(self.N):
            chunk = a_stream[k * self.align_per_step:(k + 1) * self.align_per_step]
            align_batches.append(chunk[self.rank::self.W])
        return task_batches, align_batches


# ============================================================================================
# Task step: LM loss (chi tren phan dap an) tren batch tron SQuAD/SNLI/MMLU (+ L_LB phu tro MoE)
# Chia batch cua rank thanh micro-batch theo ngan sach token, cong don gradient.
# ============================================================================================
def make_micro_batches(sorted_idx: List[int], lengths: List[int], token_budget: int) -> List[List[int]]:
    """sorted_idx da sap xep TANG DAN theo do dai -> sample moi them vao luon la dai nhat nen
    kich thuoc padded cua micro-batch = so_sample * do_dai_sample_moi."""
    micro, cur = [], []
    for i in sorted_idx:
        if cur and (len(cur) + 1) * int(lengths[i]) > token_budget:
            micro.append(cur)
            cur = []
        cur.append(i)
    if cur:
        micro.append(cur)
    return micro


def collate_task_micro(pool: Dict, idx: List[int], pad_id: int):
    """Ghep micro-batch tu ids DA TOKENIZE SAN (pool["flat"]/["offs"]) bang numpy — khong con goi
    tokenizer 2 lan/sample moi step. Right-pad bang pad_id; nhan = token cua phan dap an
    (prompt + padding = -100) — giong het cach mask cu."""
    flat, offs, plen, lens_all = pool["flat"], pool["offs"], pool["prompt_len"], pool["lengths"]
    lens = lens_all[idx]
    B, T = len(idx), int(lens.max())
    ids = np.full((B, T), pad_id, dtype=np.int64)
    att = np.zeros((B, T), dtype=np.int64)
    lab = np.full((B, T), -100, dtype=np.int64)
    for r, i in enumerate(idx):
        L, st, pl = int(lens[r]), int(offs[i]), int(plen[i])
        row = flat[st:st + L]
        ids[r, :L] = row
        att[r, :L] = 1
        lab[r, pl:L] = row[pl:]
    return torch.from_numpy(ids), torch.from_numpy(att), torch.from_numpy(lab)


def forward_backward_task_micro(ids_cpu, att_cpu, lab_cpu, model, device,
                                 router_logits_cache, num_experts, top_k, lb_loss_coef,
                                 loss_weight):
    """Forward + backward cho 1 micro-batch. L_LM = cross-entropy CHI tren token cua phan dap an
    ("<answer|label|letter><eos>"); prompt + padding bi mask -100 (giong cac file finetuning goc).

    TOI UU: chay BACKBONE roi chi nhan lm_head tren cac vi tri co nhan (~1-5% token) thay vi tinh
    logits [B, T, 151936] cho TOAN BO token (~5 GB bf16 / 16k token + them ~5 GB cho grad) roi moi
    chon. Gia tri loss giong het (lm_head la ham theo tung vi tri). Cac chi so vi tri duoc tinh tren CPU
    nen khong co device sync. Tra ve (stats[lm, lb, total, n_correct] tensor float64, n_answer_tokens)."""
    input_ids = ids_cpu.to(device, non_blocking=True)
    attention_mask = att_cpu.to(device, non_blocking=True)

    shift_cpu = lab_cpu[:, 1:]
    rows, cols = (shift_cpu != -100).nonzero(as_tuple=True)       # tren CPU
    n_ans = int(rows.numel())
    sel_labels = shift_cpu[rows, cols].to(device, non_blocking=True)
    rows_d, cols_d = rows.to(device, non_blocking=True), cols.to(device, non_blocking=True)

    router_logits_cache.clear()
    backbone, lm_head = get_backbone_and_head(model)
    if backbone is not None:
        hidden = backbone(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False).last_hidden_state            # [B, T, H]
        sel_logits = lm_head(hidden[:, :-1][rows_d, cols_d]) if n_ans else None   # [n_ans, V]
        anchor = hidden.sum() * 0.0
    else:                                                               # fallback: forward day du
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        sel_logits = logits[:, :-1][rows_d, cols_d] if n_ans else None
        anchor = logits.sum() * 0.0

    lm_loss = F.cross_entropy(sel_logits.float(), sel_labels) if n_ans else anchor

    if router_logits_cache and num_experts and top_k:
        lb_loss = compute_load_balancing_loss(router_logits_cache, attention_mask, num_experts, top_k)
        lb_loss = lb_loss.to(lm_loss.device)
    else:
        lb_loss = torch.zeros((), device=lm_loss.device)

    total_loss = lm_loss + lb_loss_coef * lb_loss
    (total_loss * loss_weight).backward()

    with torch.no_grad():
        n_correct = (sel_logits.argmax(dim=-1) == sel_labels).sum().to(torch.float64) if n_ans \
            else torch.zeros((), dtype=torch.float64, device=lm_loss.device)
        stats = torch.stack([lm_loss.detach().double(), lb_loss.detach().double(),
                             total_loss.detach().double(), n_correct])
    router_logits_cache.clear()
    return stats, n_ans


def compute_task_step(batch_idx: List[int], pool: Dict, pad_id: int, model, token_budget: int, device,
                       router_logits_cache: list, num_experts, top_k, lb_loss_coef: float) -> dict:
    n = len(batch_idx)
    acc = torch.zeros(4, dtype=torch.float64, device=device)     # cong don tren GPU, 1 lan .tolist() cuoi step
    n_ans_total = 0
    for mb in make_micro_batches(batch_idx, pool["lengths"], token_budget):
        ids, att, lab = collate_task_micro(pool, mb, pad_id)
        st, cnt = forward_backward_task_micro(
            ids, att, lab, model, device, router_logits_cache,
            num_experts, top_k, lb_loss_coef, loss_weight=len(mb) / n)
        w = len(mb)
        acc += st * torch.tensor([w, w, w, 1.0], dtype=torch.float64, device=device)
        n_ans_total += cnt
    lm, lb, tot, correct = acc.tolist()
    return {"lm": lm / n, "lb": lb / n, "tot": tot / n, "correct": correct, "ans": n_ans_total}


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
                       task_total_loss=None, token_acc=None, align_loss=None, n_samples=None):
    rec = {
        "step": global_step, "epoch": epoch, "step_type": step_type,
        "lm_loss": lm_loss, "lb_loss": lb_loss, "task_total_loss": task_total_loss,
        "token_acc": token_acc, "align_loss": align_loss, "n_samples": n_samples,
        "timestamp": time.time(),
    }
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
def build_model_card(args, num_experts, top_k, align_layer_0based, num_layers,
                      lora_layers, n_task, n_align, steps_per_side,
                      task_stats, world_size) -> str:
    task_lines = "\n".join(f"  - {k}: {v['kept']} sample (doc {v['loaded']}, bo {v['dropped']})"
                           for k, v in task_stats.items())
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

# Qwen1.5-MoE-A2.7B-MidAlign (task = SQuAD + SNLI + MMLU)

LoRA adapter finetune tu `{args.model_name_or_path}` theo baseline **MidAlign**, Alternate
Training: step chan = task step (LM loss), step le = contrastive align step.

## Alternate Training
- **Task step (chan)**: `L_task = L_LM + lb_loss_coef * L_LB`. `L_LM` chi tinh tren phan dap an
  (prompt bi mask), tren pool gop cua cac tap: {", ".join(args.task_datasets)}.
{task_lines}
  - `lb_loss_coef` = {args.lb_loss_coef}, `num_experts` = {num_experts}, `top_k` = {top_k}
- **Align step (le)**: contrastive Eq.(1) MidAlign ({args.align_loss}, negative trong mini-batch {args.align_micro_batch_size}) giua mean-pooled hidden state cau
  tieng Anh va cau target tai layer {args.align_layer} (block 0-indexed = {align_layer_0based}
  / {num_layers} layer), temperature = {args.align_temperature}.
- Moi epoch: {steps_per_side} task step + {steps_per_side} align step. TASK la anchor: moi epoch dung HET {n_task} sample
  task (SQuAD/SNLI/MMLU, English-only) >= 1 lan. Align = pool nho {n_align} cap (nguon: {", ".join(args.alignment_data)};
  {args.align_pairs_per_lang} cap/ngon ngu, boc ngau nhien seed {args.seed}), lap lai de du
  {steps_per_side * world_size * args.align_batch_size} cap align / epoch (dung nhu paper).
- Contrastive: effective batch {args.align_batch_size}/rank, mini-batch {args.align_micro_batch_size}, global_negatives = {args.align_global_negatives}, mask_false_negatives = {args.mask_false_negatives}, lang_balance = {args.align_lang_balance}.
- world_size = {world_size}, align_batch_size = {args.align_batch_size} (per-rank, anchor),
  task_batch_size = {args.task_batch_size} (per-rank), task_micro_batch_tokens = {args.task_micro_batch_tokens}.

## LoRA
- Layer finetune: `{format_layers(lora_layers)}` ({len(lora_layers)} layer, 0-indexed){' - chi dinh qua --layers' if args.layers else ' - mac dinh [L/3, 2L/3)'}, attention / router / experts + shared expert.
- Rank: attention {args.lora_r_attn}, router {args.lora_r_router}, experts {args.lora_r_expert};
  alpha = {args.lora_alpha}, dropout = {args.lora_dropout}.

## Training
- {args.num_train_epochs} epoch, gradient all-reduce thu cong (khong DDP), checkpoint chi giu ban moi nhat.
- Diagnostics (log theo tung step: `diagnostics/loss_log.jsonl`):

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

    args.data_files = resolve_data_files(args.alignment_data, args.data_files)
    logger.info(f"--alignment_data={args.alignment_data} -> data_files={args.data_files}")

    hf_token = load_hf_token(args.env_file, args.hf_token) if args.push_to_hub else None

    diagnostics_dir = args.diagnostics_dir or os.path.join(args.output_dir, "diagnostics")
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = os.path.join(diagnostics_dir, "loss_log.jsonl")
    task_plot_path = os.path.join(diagnostics_dir, "task_loss_curve.png")
    align_plot_path = os.path.join(diagnostics_dir, "align_loss_curve.png")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    # ---------------------------------------------------------------------------------- model
    logger.info(f"Dang load tokenizer va model tu {args.model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                               trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = dict(trust_remote_code=args.trust_remote_code)
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

    target_modules = build_lora_target_modules(base_model, layer_indices)
    if not target_modules:
        raise RuntimeError("Khong tim thay module attention/router/experts nao trong range layer. "
                           "Kiem tra regex trong build_lora_target_modules().")
    attn_names, expert_names, router_names = categorize_lora_targets(target_modules)
    rank_pattern = build_rank_pattern(attn_names, expert_names, router_names,
                                       args.lora_r_attn, args.lora_r_expert, args.lora_r_router)
    logger.info(f"rank_pattern ({len(rank_pattern)} key regex): {rank_pattern}")
    logger.info(f"{len(target_modules)} target module LoRA: {len(attn_names)} attention "
                f"(r={args.lora_r_attn}), {len(expert_names)} experts (r={args.lora_r_expert}), "
                f"{len(router_names)} router (r={args.lora_r_router}).")

    num_experts, top_k = infer_moe_dims(base_model.config, args)
    lb_loss_coef = args.lb_loss_coef
    if lb_loss_coef is None:
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

    # ------------------------------------------------------------------------------------ data
    cache_dir = None if args.no_cache else (args.cache_dir or os.path.join(args.output_dir, "cache"))

    def _cache_path(prefix: str, sig: dict) -> Optional[str]:
        if cache_dir is None:
            return None
        h = hashlib.md5(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:12]
        return os.path.join(cache_dir, f"{prefix}_{h}.pkl")

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

    align_sig = {"v": 5, "ppl": args.align_pairs_per_lang, "ted_ppr": args.ted_pairs_per_record, "files": file_signature([os.path.join(args.data_dir, f) for f in args.data_files]),
                 "eng_key": args.eng_key, "mlp": args.max_lang_pairs_per_record, "seed": args.seed,
                 "max_samples": args.max_samples}
    align_data = cached_build(_cache_path("align", align_sig), _build_align, is_main, is_distributed)
    pairs, align_src = align_data["pairs"], align_data["src"]
    del align_data

    def _build_task():
        logger.info(f"Dang doc + build pool task ({args.task_datasets}) ...")
        pl = build_task_pool(args, tokenizer)
        if len(pl["lengths"]) == 0:
            raise RuntimeError("Pool task rong — kiem tra --squad_file / --snli_file / --mmlu_file.")
        return pl

    task_sig = {"v": 2, "files": file_signature([args.squad_file, args.snli_file, args.mmlu_file]),
                "ds": args.task_datasets, "max_len": args.task_max_length, "mts": args.max_task_samples,
                "seed": args.seed, "tok": [tokenizer.name_or_path, len(tokenizer), tokenizer.eos_token]}
    pool = cached_build(_cache_path("task", task_sig), _build_task, is_main, is_distributed)
    task_lengths, task_stats = pool["lengths"], pool["stats"]
    pad_id = int(tokenizer.pad_token_id)

    n_task, n_align = len(task_lengths), len(pairs)
    task_bs, align_bs = int(args.task_batch_size), int(args.align_batch_size)
    if task_bs < 1:
        raise ValueError(f"--task_batch_size phai >= 1 (nhan duoc {args.task_batch_size}).")
    if align_bs < 2:
        raise ValueError(f"--align_batch_size phai >= 2 (contrastive can >= 2 cap/rank; nhan duoc {align_bs}).")
    N = solve_schedule(n_task, world_size, task_bs)                      # TASK la anchor
    steps_per_epoch = 2 * N                       # luon chan -> epoch nao cung bat dau bang task step
    total_steps = steps_per_epoch * args.num_train_epochs
    task_per_epoch = N * world_size * task_bs     # >= n_task (dem them < 1 task step)
    align_per_epoch = N * world_size * align_bs   # cap align can moi epoch (lap lai pool nho)
    align_cover = align_per_epoch / n_align
    src_counts = np.bincount(align_src.astype(np.int64)).tolist() if len(align_src) else []
    logger.info(
        f"[schedule] ANCHOR = TASK | n_task={n_task} (SQuAD/SNLI/MMLU) -> moi sample dung >= 1 lan/epoch | "
        f"N={N} task step + {N} align step / epoch (steps_per_epoch={steps_per_epoch}, "
        f"total_steps={total_steps}) | batch/rank: task={task_bs}, align={align_bs} | "
        f"align: pool nho {n_align} cap (seed={args.seed}, {args.align_pairs_per_lang}/ngon ngu; theo file "
        f"{args.data_files}: {src_counts}), moi epoch can {align_per_epoch} cap = {align_cover:.1f}x pool "
        f"(LAP LAI, dung nhu paper)."
    )
    if align_cover > 100:
        logger.warning(
            f"[canh bao] Moi cap align bi lap ~{align_cover:.0f} lan / epoch (~{align_cover * args.num_train_epochs:.0f} "
            f"lan trong {args.num_train_epochs} epoch). Paper cung lap pool nho nhieu lan, nhung neu thay align "
            f"loss ve ~0 som (overfit) hay giam --num_train_epochs hoac tang --align_pairs_per_lang.")

    lang_ids = None
    if args.align_lang_balance:
        lang_names = sorted({p[2] for p in pairs})
        lmap = {l: i for i, l in enumerate(lang_names)}
        lang_ids = np.fromiter((lmap[p[2]] for p in pairs), dtype=np.int32, count=len(pairs))
        cnt = np.bincount(lang_ids, minlength=len(lang_names))
        logger.info(f"[align] resample DEU {len(lang_names)} ngon ngu: ~{align_per_epoch // len(lang_names)} "
                    f"cap/ngon ngu/epoch (pool: min={int(cnt.min())}, max={int(cnt.max())} cap/ngon ngu).")
    plan = AlternatePlan(n_task, n_align, task_lengths, N, world_size, rank, args.seed, task_bs, align_bs,
                         lang_ids)

    # ------------------------------------------------------------------------------- optimizer
    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                   weight_decay=args.weight_decay,
                                   fused=True if use_fused else None, foreach=None if use_fused else True)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def _inv_sqrt(step: int) -> float:
        # paper: inverse square root schedule, warmup tuyen tinh -> lr_max, sau do lr_max * sqrt(warmup / step)
        step = step + 1
        return step / warmup_steps if step < warmup_steps else (warmup_steps / step) ** 0.5

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _inv_sqrt)

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

    readme_text = build_model_card(args, num_experts, top_k, align_layer_0based, num_layers,
                                    lora_layers, n_task, n_align, N,
                                    task_stats, world_size)

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
    try:
        for epoch in range(start_epoch, args.num_train_epochs):
            step_offset = start_step_in_epoch if epoch == start_epoch else 0
            task_batches, align_batches = plan.epoch_batches(epoch)

            pbar = tqdm(range(step_offset, steps_per_epoch), total=steps_per_epoch,
                        initial=step_offset, desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
                        disable=not is_main)
            for step_in_epoch in pbar:
                model.train()
                optimizer.zero_grad(set_to_none=True)

                # Alternate Training: step chan = task, step le = align
                # (steps_per_epoch chan nen step_in_epoch % 2 == global_step % 2)
                k = step_in_epoch // 2
                if step_in_epoch % 2 == 0:
                    step_type = "task"
                    r = compute_task_step(
                        task_batches[k].tolist(), pool, pad_id, model,
                        args.task_micro_batch_tokens, device,
                        router_logits_cache, num_experts, top_k, lb_loss_coef)
                    n_local = len(task_batches[k])
                    sums = reduce_metrics(
                        [r["lm"], r["lb"], r["tot"], r["correct"], r["ans"], 1.0, n_local],
                        device, is_distributed)
                    nr = sums[5]
                    log_kwargs = dict(lm_loss=sums[0] / nr, lb_loss=sums[1] / nr,
                                       task_total_loss=sums[2] / nr,
                                       token_acc=sums[3] / max(sums[4], 1.0),
                                       align_loss=None, n_samples=int(sums[6]))
                    postfix = {"type": "task", "L_LM": f"{log_kwargs['lm_loss']:.4f}",
                               "L_LB": f"{log_kwargs['lb_loss']:.4f}",
                               "acc": f"{log_kwargs['token_acc']:.3f}"}
                else:
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
                    log_kwargs = dict(align_loss=sums[0] / sums[1], n_samples=int(sums[2]))
                    postfix = {"type": "align", "L_align": f"{log_kwargs['align_loss']:.4f}"}

                if is_distributed:
                    sync_grads_across_ranks(trainable_params, world_size)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1
                last_done = (epoch, step_in_epoch)

                if is_main:
                    if global_step % args.log_every == 0:
                        pbar.set_postfix(postfix)
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