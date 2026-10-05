"""
MidAlign baseline cho mo hinh Mixture-of-Experts Qwen/Qwen1.5-MoE-A2.7B
(kien truc Qwen2MoeForCausalLM) — phien ban TASK STEP = LM loss tren SQuAD + SNLI + MMLU.

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
        Ba tap duoc GOP THANH 1 POOL DUY NHAT, moi task step lay 1 batch ngau nhien tu pool nay
        (tron lan 3 task trong cung 1 batch).
  - step LE (1, 3, 5, ...) = ALIGN (contrastive) step:
        L_align = symmetric InfoNCE (in-batch negatives) giua mean-pooled hidden state cua cau
        tieng Anh va cau target (cap english-other), tai DUNG 1 layer (--align_layer, mac dinh
        TU DONG = num_hidden_layers // 2 cua backbone dang load).

RANG BUOC "DUNG HET DU LIEU" va cach tinh so step
-------------------------------------------------
Vi step chan/le xen ke deu nhau, trong 1 epoch co N task step va N align step (tong 2N step,
steps_per_epoch luon CHAN nen epoch nao cung bat dau bang task step). De moi epoch:
    - N task step  tieu thu DUNG HET  n_task  sample (SQuAD + SNLI + MMLU sau khi loc),
    - N align step tieu thu DUNG HET  n_align cap bitext,
moi sample duoc dung dung 1 lan / epoch (khong lap, khong bo), ta chon N roi CHIA DEU
n_task va n_align cho N step (step k nhan floor hoac ceil cua n/N sample), roi chia tiep cho
cac rank. Co 2 cach chon N (chon bang co CLI, phia con lai duoc TINH RA):
    (mac dinh)  neo theo ALIGN: --align_batch_size (per-rank, mac dinh 128)
                N = round(n_align / (align_batch_size * world_size))
                -> batch size task per-rank trung binh = n_task / (N * world_size)
    (tuy chon)  neo theo TASK:  truyen --task_batch_size (per-rank)
                N = round(n_task / (task_batch_size * world_size))
                -> batch size align per-rank trung binh = n_align / (N * world_size)
Neu batch task per-rank lon thi moi rank tu dong chia thanh nhieu micro-batch (theo ngan sach
token --task_micro_batch_tokens, sort theo do dai de giam padding) va cong don gradient truoc
khi optimizer.step() — nen KHONG can OOM-split. Contrastive step khong chia micro-batch duoc
(can in-batch negatives), nen neu neo theo task ma align batch tinh ra qua lon thi giam
--task_batch_size de tang N.

Dong bo gradient: THAY DDP bang all-reduce gradient thu cong (giong cac file finetuning
english-task-only). Ly do: task step cong don gradient qua nhieu micro-batch (so micro-batch
co the khac nhau giua cac rank) va 2 loai step co do thi autograd khac nhau, nen tranh phai
phu thuoc vao hanh vi cua DDP Reducer / static_graph. Sau backward, MOI tham so trainable
(ke ca LoRA cua expert khong nhan token nao trong step do -> grad None) deu duoc dien 0 roi
all-reduce 1 lan duy nhat -> khong con nguy co NCCL watchdog / "marked ready twice".
(Do do cac co --find_unused_parameters / zero_grad_anchor / _set_static_graph da bo.)

Cac dieu kien giu nguyen tu ban MidAlign truoc:
  1. LoRA ap dung cho RANGE layer [L/3, 2L/3), tach bach voi layer tinh alignment loss.
     Attention / router / experts moi nhom 1 rank rieng (router 4, attn 16, experts 16).
  2. Cap ngon ngu english - other, doc tu du lieu multiway-parallel JSON (flores/ntrex/ted).
  3. Checkpoint chi giu ban moi nhat, push len HF Hub, resume tu checkpoint.

Vi du chay (8 GPU):
    torchrun --standalone --nproc_per_node=8 Qwen1.5-MoE-A2.7B.py \\
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \\
        --data_dir data/processed_alignment \\
        --squad_file data/english_task/squad/train.json \\
        --snli_file data/english_task/snli/train.json \\
        --mmlu_file data/english_task/mmlu/auxiliary_train.json \\
        --push_to_hub

Smoke test (1 GPU, du lieu nho):
    python Qwen1.5-MoE-A2.7B.py --max_samples 20000 --max_task_samples 5000 --no_push_to_hub

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
import random
import re
import shutil
import string
import time
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple

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
                    help="Nguon du lieu alignment: flores | ntrex | ted (1 hoac nhieu). Mac dinh flores ntrex ted, CUNG nguon voi code finetuning baseline.")
    p.add_argument("--data_files", type=str, nargs="+", default=None,
                    help="[Nang cao] Ghi de --alignment_data bang danh sach file JSON trong --data_dir.")
    p.add_argument("--eng_key", type=str, default="eng_Latn")
    p.add_argument("--max_lang_pairs_per_record", type=int, default=None,
                    help="Gioi han so ngon ngu ghep voi eng_key trong 1 record (None = dung het).")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Gioi han so cap bitext alignment (debug), None = dung het.")
    p.add_argument("--max_pairs_per_file", type=int, default=None,
                    help="Gioi han so cap bitext MOI FILE alignment (cat ngau nhien, can bang cac nguon). "
                         "Huu ich khi 1 file chiem da so cap (flores ~0.4M, ntrex ~0.25M; xem log "
                         "'<file>: +N cap bitext' de biet kich thuoc thuc te cua ted).")

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
                    help="Batch size align per-rank (so cap bitext / rank / align step). Mac dinh la "
                         "ANCHOR quyet dinh N = so task step = so align step moi epoch. Bi bo qua "
                         "(tinh lai) neu truyen --task_batch_size.")
    p.add_argument("--task_batch_size", type=int, default=None,
                    help="Neu truyen: dung lam ANCHOR (per-rank) thay cho --align_batch_size; batch "
                         "size align khi do se TU DONG TINH de dung het du lieu alignment.")
    p.add_argument("--task_micro_batch_tokens", type=int, default=16384,
                    help="Ngan sach token (so sample * do dai da padding) cho 1 micro-batch cua task "
                         "step. Giam neu OOM, tang neu con du VRAM.")
    p.add_argument("--max_length", type=int, default=256,
                    help="max_length cho cau alignment (contrastive step).")
    p.add_argument("--task_max_length", type=int, default=512,
                    help="max_length cho sample task (SQuAD can 512; sample SQuAD dai hon duoc "
                         "windowing, MMLU/SNLI dai hon bi bo).")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)

    # MidAlign: alignment objective
    p.add_argument("--align_layer", type=int, default=None,
                    help="Layer lay hidden state cho contrastive loss (hidden_states[align_layer]). "
                         "None = TU DONG num_hidden_layers // 2 cua backbone dang load.")
    p.add_argument("--align_temperature", type=float, default=1.5)
    p.add_argument("--align_global_negatives", dest="align_global_negatives",
                    action="store_true", default=True,
                    help="[Mac dinh BAT] all_gather embedding qua tat ca rank de contrastive loss dung "
                         "batch TOAN CUC (world_size * align_batch_size) thay vi chi batch local. "
                         "Vd 8 GPU x --align_batch_size 8 = batch contrastive 64 nhu paper.")
    p.add_argument("--no_align_global_negatives", dest="align_global_negatives", action="store_false")
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
    p.add_argument("--lora_r", type=int, default=16, help="Rank fallback (khong nen duoc dung).")
    p.add_argument("--lora_r_router", type=int, default=4)
    p.add_argument("--lora_r_attn", type=int, default=16)
    p.add_argument("--lora_r_expert", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

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
    p.add_argument("--log_every", type=int, default=10)
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
    for p in trainable_params:
        dist.broadcast(p.data, src=src)



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


# ============================================================================================
# Du lieu ALIGNMENT: bitext english-other tu JSON multiway-parallel
# ============================================================================================
def load_bitext_pairs(data_dir: str, data_files: Sequence[str], eng_key: str,
                       max_lang_pairs_per_record: Optional[int] = None,
                       seed: int = 42,
                       max_pairs_per_file: Optional[int] = None) -> List[Tuple[str, str, str]]:
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
        n_no_eng = 0
        for rec in records:
            if not isinstance(rec, dict):
                continue
            eng_text = rec.get(eng_key)
            if not isinstance(eng_text, str) or not eng_text.strip():
                n_no_eng += 1
                continue
            eng_text = eng_text.strip()

            other_keys = [k for k in rec.keys() if k not in ("id", eng_key)]
            if max_lang_pairs_per_record is not None and len(other_keys) > max_lang_pairs_per_record:
                other_keys = rng.sample(other_keys, max_lang_pairs_per_record)

            for k in other_keys:
                v = rec.get(k)
                if isinstance(v, str) and v.strip():
                    pairs.append((eng_text, v.strip(), k))

        if max_pairs_per_file is not None and len(pairs) - n_before > max_pairs_per_file:
            file_pairs = pairs[n_before:]
            rng.shuffle(file_pairs)
            pairs[n_before:] = file_pairs[:max_pairs_per_file]
            logger.info(f"{fname}: cat ngau nhien xuong {max_pairs_per_file} cap (--max_pairs_per_file).")
        if n_no_eng:
            ex = next((list(r.keys())[:6] for r in records if isinstance(r, dict)), [])
            logger.warning(f"{fname}: {n_no_eng}/{len(records)} record KHONG co khoa '{eng_key}' "
                           f"(hoac rong) -> bi bo qua. Vi du cac khoa cua 1 record: {ex}. Neu ca file "
                           f"bi bo qua, dung --eng_key de chi dinh dung ten khoa tieng Anh.")
        logger.info(f"{fname}: +{len(pairs) - n_before} cap bitext ({eng_key}-other), "
                    f"tong so record = {len(records)}")
    return pairs


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
    toi khi vua sat ngan sach max_context_tokens. Nho vay context van luon chua answer."""
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
        # Ngay ca cua so toi thieu (chi vua du cac tu cua answer) da vuot ngan sach -> tra ve
        # nguyen trang, ham goi se tu phat hien full_text van qua dai va skip sample nay.
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


def compute_lengths(tokenizer, texts: Sequence[str], chunk_size: int = 1000) -> List[int]:
    lengths: List[int] = []
    for i in tqdm(range(0, len(texts), chunk_size), desc="Tinh do dai token cho toan bo sample"):
        chunk = texts[i:i + chunk_size]
        enc = tokenizer(chunk, add_special_tokens=True)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return lengths


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
    naive_lengths = compute_lengths(tokenizer, [ex["full_text"] for ex in naive_examples])

    # Buoc 2: chi ap dung windowing (co the cham hon, goi tokenizer nhieu lan) cho phan THIEU SO
    # sample vuot qua max_length — da so sample SQuAD se vua trong 1 lan, khong can qua buoc nay.
    examples: List[Dict] = []
    lengths: List[int] = []
    n_windowed = 0
    n_dropped_too_long = 0
    for ex, naive_len in tqdm(list(zip(naive_examples, naive_lengths)),
                               desc="Kiem tra/loc do dai (windowing context qua dai neu can)"):
        if naive_len <= max_length:
            examples.append({"prompt": ex["prompt"], "full_text": ex["full_text"]})
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
        new_len = len(tokenizer(new_full_text, add_special_tokens=True)["input_ids"])
        if new_len > max_length:
            n_dropped_too_long += 1
            continue

        examples.append({"prompt": new_prompt, "full_text": new_full_text})
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

    lengths_all = compute_lengths(tokenizer, full_texts)

    examples: List[Dict] = []
    lengths: List[int] = []
    n_dropped_too_long = 0
    for prompt, full_text, length in zip(prompts, full_texts, lengths_all):
        if length > max_length:
            n_dropped_too_long += 1
            continue
        examples.append({"prompt": prompt, "full_text": full_text})
        lengths.append(length)

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


def categorize_lora_targets(target_modules: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    """Chia danh sach target_modules (da duoc build_lora_target_modules loc) thanh 3 nhom rieng
    biet — attention / experts / router — dung LAI CHINH XAC cung dieu kien nhu trong
    build_lora_target_modules(), de dam bao khop 1-1 voi target_modules truyen vao LoraConfig.
    Dung cho rank_pattern (dieu kien thay doi #2: moi nhom co the mang 1 rank LoRA rieng)."""
    attn_names, expert_names, router_names = [], [], []
    for name in target_modules:
        if is_router_leaf_name(name):
            router_names.append(name)
        elif ".experts." in name or ".expert." in name:
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
                            router_logits_cache: list, global_negatives: bool = True,
                            mask_false_negatives: bool = True):
    """Contrastive loss (symmetric InfoNCE, Eq.1 MidAlign) tai align_layer.

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
    pooled_eng = encode_layer_representation(eng_texts, tokenizer, model, align_layer,
                                              max_length, device, router_logits_cache)
    pooled_other = encode_layer_representation(other_texts, tokenizer, model, align_layer,
                                                max_length, device, router_logits_cache)

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
    sim_o2e = torch.matmul(b_loc, a_all.t()) / temperature   # [b, B]  query = target local
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
        sim_o2e = sim_o2e.masked_fill(dup, float("-inf"))

    loss_e2o = F.cross_entropy(sim_e2o, target)
    loss_o2e = F.cross_entropy(sim_o2e, target)
    return (loss_e2o + loss_o2e) / 2.0


# ============================================================================================
# Task pool: gop SQuAD + SNLI + MMLU thanh 1 danh sach example duy nhat {"prompt","full_text","src"}
# (doc/build prompt/windowing/loc do dai bang CHINH cac ham cua cac file finetuning goc)
# ============================================================================================
def build_task_pool(args, tokenizer) -> Tuple[List[Dict], List[int], Dict[str, Dict[str, int]]]:
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

    def _add(src: str, exs: List[Dict], lens: List[int], n_loaded: int):
        for ex, ln in zip(exs, lens):
            examples.append({"prompt": ex["prompt"], "full_text": ex["full_text"], "src": src})
            lengths.append(ln)
        stats[src] = {"loaded": n_loaded, "kept": len(exs), "dropped": n_loaded - len(exs)}

    if "squad" in args.task_datasets:
        recs = _cap(load_squad_records(args.squad_file))
        exs, lens = squad_build_examples(recs, tokenizer, eos, args.task_max_length)
        _add("squad", exs, lens, len(recs))
    if "snli" in args.task_datasets:
        recs = _cap(load_snli_records(args.snli_file))
        all_exs = snli_build_examples(recs, eos)
        all_lens = compute_lengths(tokenizer, [e["full_text"] for e in all_exs])
        keep = [(e, l) for e, l in zip(all_exs, all_lens) if l <= args.task_max_length]
        _add("snli", [k[0] for k in keep], [k[1] for k in keep], len(recs))
    if "mmlu" in args.task_datasets:
        recs = _cap(load_mmlu_records(args.mmlu_file))
        exs, lens = mmlu_build_examples(recs, tokenizer, eos, args.task_max_length)
        _add("mmlu", exs, lens, len(recs))

    for src, st in stats.items():
        logger.info(f"[task pool] {src}: doc {st['loaded']}, giu {st['kept']}, "
                    f"bo (qua dai khong the cat) {st['dropped']}")
    return examples, lengths, stats


# ============================================================================================
# Lich xen ke task/align + chia du lieu: moi epoch = N task step + N align step, dung HET data
# ============================================================================================
def solve_steps_per_side(n_task: int, n_align: int, world_size: int, align_batch_size: int,
                          task_batch_size: Optional[int]) -> Tuple[int, str]:
    """Tra ve (N, anchor). N = so task step = so align step moi epoch."""
    if task_batch_size is None:
        n = int(n_align / (align_batch_size * world_size) + 0.5)
        return max(1, n), "align"
    n = int(n_task / (task_batch_size * world_size) + 0.5)
    return max(1, n), "task"


def split_sizes(total: int, parts: int) -> List[int]:
    """Chia `total` thanh `parts` phan chenh nhau toi da 1 (phan dau nhan them 1 neu du)."""
    base, rem = divmod(total, parts)
    return [base + 1 if i < rem else base for i in range(parts)]


class AlternatePlan:
    """Moi epoch: shuffle (seed + epoch) pool task va pool align, cat thanh N lat theo
    split_sizes -> MOI sample xuat hien DUNG 1 LAN / epoch. Moi lat chia tiep cho cac rank
    (rank r lay phan tu r, r+W, r+2W, ...). Lat task duoc sort theo do dai TRUOC khi chia rank
    de moi rank co phan phoi do dai nhu nhau (can bang tai) va batch cua rank da sap xep tang
    dan (thuan loi cho micro-batching theo token). Xac dinh hoan toan theo (seed, epoch) nen
    resume giua epoch cho ra dung cac batch nhu lan chay truoc."""

    def __init__(self, n_task, n_align, task_lengths, steps_per_side, world_size, rank, seed):
        self.n_task, self.n_align = n_task, n_align
        self.task_lengths = task_lengths
        self.N, self.W, self.rank, self.seed = steps_per_side, world_size, rank, seed

    def _perm(self, n: int, epoch: int, salt: int) -> List[int]:
        rng = random.Random(self.seed * 1_000_003 + epoch * 101 + salt)
        perm = list(range(n))
        rng.shuffle(perm)
        return perm

    def epoch_batches(self, epoch: int) -> Tuple[List[List[int]], List[List[int]]]:
        t_perm = self._perm(self.n_task, epoch, 1)
        a_perm = self._perm(self.n_align, epoch, 2)
        task_batches, align_batches = [], []
        pos = 0
        for s in split_sizes(self.n_task, self.N):
            chunk = t_perm[pos:pos + s]
            pos += s
            chunk.sort(key=lambda i: self.task_lengths[i])
            task_batches.append(chunk[self.rank::self.W])
        # Align: moi lat PHAI chia deu cho cac rank (all_gather can cung shape tren moi rank).
        # Gom thanh `units` khoi, moi khoi W sample; khoi cuoi duoc dem them < W sample lay tu dau
        # hoan vi (chi toi da W-1 sample/epoch bi dung 2 lan, nam o lat khac nhau), roi chia
        # units khoi cho N lat. Nho vay moi lat co size = (so khoi) * W, rank nao cung nhan bang nhau.
        units = -(-self.n_align // self.W)
        pad = units * self.W - self.n_align
        a_perm = a_perm + a_perm[:pad]
        pos = 0
        for u in split_sizes(units, self.N):
            s = u * self.W
            chunk = a_perm[pos:pos + s]
            pos += s
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
        if cur and (len(cur) + 1) * lengths[i] > token_budget:
            micro.append(cur)
            cur = []
        cur.append(i)
    if cur:
        micro.append(cur)
    return micro


def forward_backward_task_micro(sub_examples, tokenizer, model, max_length, device,
                                 router_logits_cache, num_experts, top_k, lb_loss_coef,
                                 loss_weight):
    """Tokenize + forward + backward cho 1 micro-batch. L_LM = cross-entropy CHI tren token
    cua phan dap an ("<answer|label|letter><eos>"); prompt + padding bi mask -100 (giong cac
    file finetuning goc). Cross-entropy chi tinh tren cac vi tri co nhan (tuong duong
    ignore_index=-100 nhung khong phai copy ca tensor [B, T, V] logits).
    Tra ve (lm, lb, total, n_correct_tokens, n_answer_tokens)."""
    full_texts = [ex["full_text"] for ex in sub_examples]
    prompts = [ex["prompt"] for ex in sub_examples]

    enc = tokenizer(full_texts, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt")
    input_ids = enc["input_ids"].to(device, non_blocking=True)
    attention_mask = enc["attention_mask"].to(device, non_blocking=True)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    seq_len = labels.shape[1]
    for i, p in enumerate(prompts):
        ids = tokenizer(p, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
        labels[i, :min(len(ids), seq_len)] = -100

    router_logits_cache.clear()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits

    shift_labels = labels[..., 1:]
    ans_mask = shift_labels != -100
    sel_logits = logits[..., :-1, :][ans_mask]          # [n_answer_tokens, V]
    sel_labels = shift_labels[ans_mask]
    if sel_labels.numel() > 0:
        lm_loss = F.cross_entropy(sel_logits, sel_labels)
    else:
        lm_loss = logits.sum() * 0.0

    if router_logits_cache and num_experts and top_k:
        lb_loss = compute_load_balancing_loss(router_logits_cache, attention_mask, num_experts, top_k)
        lb_loss = lb_loss.to(lm_loss.device)
    else:
        lb_loss = torch.zeros((), device=lm_loss.device)

    total_loss = lm_loss + lb_loss_coef * lb_loss
    (total_loss * loss_weight).backward()

    with torch.no_grad():
        n_ans = int(sel_labels.numel())
        n_correct = int((sel_logits.argmax(dim=-1) == sel_labels).sum().item()) if n_ans else 0
    router_logits_cache.clear()
    return lm_loss.item(), lb_loss.item(), total_loss.item(), n_correct, n_ans


def compute_task_step(batch_idx: List[int], task_examples: List[Dict], task_lengths: List[int],
                       tokenizer, model, max_length: int, token_budget: int, device,
                       router_logits_cache: list, num_experts, top_k, lb_loss_coef: float) -> dict:
    n = len(batch_idx)
    agg = {"lm": 0.0, "lb": 0.0, "tot": 0.0, "correct": 0, "ans": 0}
    for mb in make_micro_batches(batch_idx, task_lengths, token_budget):
        sub = [task_examples[i] for i in mb]
        lm, lb, tot, corr, cnt = forward_backward_task_micro(
            sub, tokenizer, model, max_length, device, router_logits_cache,
            num_experts, top_k, lb_loss_coef, loss_weight=len(mb) / n,
        )
        agg["lm"] += lm * len(mb)
        agg["lb"] += lb * len(mb)
        agg["tot"] += tot * len(mb)
        agg["correct"] += corr
        agg["ans"] += cnt
    return {"lm": agg["lm"] / n, "lb": agg["lb"] / n, "tot": agg["tot"] / n,
            "correct": agg["correct"], "ans": agg["ans"]}


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
def log_step_to_jsonl(jsonl_path, global_step, epoch, step_type, lm_loss=None, lb_loss=None,
                       task_total_loss=None, token_acc=None, align_loss=None, n_samples=None):
    rec = {
        "step": global_step, "epoch": epoch, "step_type": step_type,
        "lm_loss": lm_loss, "lb_loss": lb_loss, "task_total_loss": task_total_loss,
        "token_acc": token_acc, "align_loss": align_loss, "n_samples": n_samples,
        "timestamp": time.time(),
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


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
                      lora_layer_start, lora_layer_end, n_task, n_align, steps_per_side,
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
- **Align step (le)**: symmetric InfoNCE (in-batch negatives) giua mean-pooled hidden state cau
  tieng Anh va cau target tai layer {args.align_layer} (block 0-indexed = {align_layer_0based}
  / {num_layers} layer), temperature = {args.align_temperature}.
- Moi epoch: {steps_per_side} task step + {steps_per_side} align step; dung het {n_task} sample task
  va {n_align} cap bitext (nguon: {", ".join(args.alignment_data)}), moi sample 1 lan / epoch.
- Contrastive batch: global_negatives = {args.align_global_negatives} (batch toan cuc = {world_size} x align_batch_size), mask_false_negatives = {args.mask_false_negatives}.
- world_size = {world_size}, align_batch_size = {args.align_batch_size}, task_batch_size = {args.task_batch_size}
  (None = tinh tu align), task_micro_batch_tokens = {args.task_micro_batch_tokens}.

## LoRA
- Range layer `[{lora_layer_start}, {lora_layer_end})` (0-indexed), attention / router / experts.
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

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, trust_remote_code=args.trust_remote_code
    )
    base_model.to(device)

    num_layers = get_num_layers(base_model.config)
    if args.align_layer is None:
        args.align_layer = infer_middle_layer(num_layers)
        logger.info(f"--align_layer khong duoc chi dinh -> middle layer = {args.align_layer} "
                    f"(num_hidden_layers={num_layers}, num_layers // 2).")
    if not (1 <= args.align_layer <= num_layers):
        raise ValueError(f"--align_layer={args.align_layer} phai nam trong [1, {num_layers}].")
    align_layer_0based = args.align_layer - 1

    lora_layer_start = num_layers // 3
    lora_layer_end = (2 * num_layers) // 3  # exclusive
    layer_indices = set(range(lora_layer_start, lora_layer_end))
    logger.info(f"Tong so layer = {num_layers}. Align tai layer {args.align_layer} "
                f"(block 0-indexed {align_layer_0based}). LoRA tren range "
                f"[{lora_layer_start}, {lora_layer_end}) ({len(layer_indices)} layer).")
    if align_layer_0based not in layer_indices:
        logger.warning(f"[canh bao] align layer (block {align_layer_0based}) nam NGOAI range LoRA "
                       f"[{lora_layer_start}, {lora_layer_end}) -> contrastive loss khong day "
                       f"gradient vao tham so LoRA nao.")

    target_modules = build_lora_target_modules(base_model, layer_indices)
    if not target_modules:
        raise RuntimeError("Khong tim thay module attention/router/experts nao trong range layer. "
                           "Kiem tra regex trong build_lora_target_modules().")
    attn_names, expert_names, router_names = categorize_lora_targets(target_modules)
    rank_pattern = {}
    rank_pattern.update({n: args.lora_r_router for n in router_names})
    rank_pattern.update({n: args.lora_r_attn for n in attn_names})
    rank_pattern.update({n: args.lora_r_expert for n in expert_names})
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
    logger.info(f"Dang doc bitext {args.eng_key}-other tu {args.data_dir} ({args.data_files}) ...")
    pairs = load_bitext_pairs(args.data_dir, args.data_files, args.eng_key,
                               args.max_lang_pairs_per_record, args.seed,
                               max_pairs_per_file=args.max_pairs_per_file)
    if args.max_samples:
        random.Random(args.seed).shuffle(pairs)
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise RuntimeError("Khong doc duoc cap bitext nao — kiem tra --data_dir / --data_files / --eng_key.")

    logger.info(f"Dang doc + build pool task ({args.task_datasets}) ...")
    task_examples, task_lengths, task_stats = build_task_pool(args, tokenizer)
    if not task_examples:
        raise RuntimeError("Pool task rong — kiem tra --squad_file / --snli_file / --mmlu_file.")

    n_task, n_align = len(task_examples), len(pairs)
    N, anchor = solve_steps_per_side(n_task, n_align, world_size, args.align_batch_size,
                                      args.task_batch_size)
    if n_task // N < world_size:
        raise RuntimeError(f"Pool task ({n_task}) qua nho cho N={N} step x {world_size} rank "
                           f"(moi rank can >= 1 sample/step). Giam --task_batch_size / tang du lieu.")
    if n_align // N < 2 * world_size:
        raise RuntimeError(f"Pool align ({n_align}) qua nho cho N={N} step x {world_size} rank "
                           f"(moi rank can >= 2 cap/step cho contrastive).")
    steps_per_epoch = 2 * N                       # luon chan -> epoch nao cung bat dau bang task step
    total_steps = steps_per_epoch * args.num_train_epochs
    logger.info(
        f"[schedule] anchor={anchor} | n_task={n_task} (SQuAD/SNLI/MMLU) | n_align={n_align} | "
        f"N={N} task step + {N} align step / epoch (steps_per_epoch={steps_per_epoch}, "
        f"total_steps={total_steps}) | batch/rank trung binh: task={n_task / (N * world_size):.1f}, "
        f"align={n_align / (N * world_size):.1f} | moi sample dung 1 lan/epoch."
    )

    task_bs_avg = n_task / (N * world_size)
    if task_bs_avg < 8:
        logger.warning(
            f"[canh bao] Batch task trung binh chi ~{task_bs_avg:.1f} sample/rank/step (global "
            f"~{task_bs_avg * world_size:.0f}) vi pool align ({n_align}) LON HON NHIEU pool task ({n_task}) "
            f"nen N={N} qua lon. Moi step MoE+LoRA ton thoi gian gan nhu co dinh (nhieu kernel nho) "
            f"-> {total_steps} step se rat lau va gradient task rat nhieu. Giam pool align "
            f"(--max_pairs_per_file / --max_lang_pairs_per_record / bo bot nguon khoi --alignment_data / "
            f"--max_samples) hoac tang --align_batch_size / giam --num_train_epochs. Muc tieu: "
            f"n_align ~ n_task x (align_batch_toan_cuc / task_batch_toan_cuc).")

    plan = AlternatePlan(n_task, n_align, task_lengths, N, world_size, rank, args.seed)

    # ------------------------------------------------------------------------------- optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                   weight_decay=args.weight_decay, foreach=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    start_epoch, start_step_in_epoch, global_step = 0, 0, 0
    prev_checkpoint_dir = resume_dir
    if resume_dir:
        state_path = os.path.join(resume_dir, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu")
            if state.get("steps_per_epoch") not in (None, steps_per_epoch):
                raise RuntimeError(
                    f"Checkpoint co steps_per_epoch={state['steps_per_epoch']} nhung cau hinh hien "
                    f"tai cho {steps_per_epoch} (world_size/batch size/du lieu da doi?) -> khong the "
                    f"resume chinh xac.")
            optimizer.load_state_dict(state["optimizer"])
            for group in optimizer.param_groups:
                group["foreach"] = True
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
                                    lora_layer_start, lora_layer_end, n_task, n_align, N,
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
                        task_batches[k], task_examples, task_lengths, tokenizer, model,
                        args.task_max_length, args.task_micro_batch_tokens, device,
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
                    batch = [pairs[i] for i in align_batches[k]]
                    eng_texts = [b[0] for b in batch]
                    other_texts = [b[1] for b in batch]
                    align_loss = compute_alignment_step(
                        eng_texts, other_texts, tokenizer, model, args.align_layer,
                        args.max_length, device, args.align_temperature, router_logits_cache,
                        global_negatives=args.align_global_negatives and is_distributed,
                        mask_false_negatives=args.mask_false_negatives)
                    align_loss.backward()
                    router_logits_cache.clear()
                    sums = reduce_metrics([align_loss.item(), 1.0, len(batch)], device, is_distributed)
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
                    pbar.set_postfix(postfix)
                    log_step_to_jsonl(jsonl_path, global_step, epoch, step_type, **log_kwargs)
                    if global_step % args.log_every == 0:
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
        for h in hooks:
            h.remove()
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()