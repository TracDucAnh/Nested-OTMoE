"""

How to run:

torchrun --standalone --nproc_per_node=8 Qwen1.5-MoE-A2.7B-SQuAD.py \
  --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
  --data_file data/english_task/squad/train.json \
  --batch_size 64 \
  --save_steps 200 \
  --push_to_hub

Fine-tuning LoRA cho mo hinh Mixture-of-Experts Qwen/Qwen1.5-MoE-A2.7B tren task SQuAD
(Extractive Question Answering), theo huong "english-task-only" (khong dung du lieu alignment).

Loss = L_LM (cross-entropy chuan, CHI tinh tren token cua cau tra loi/answer, prompt bi mask -100)
       + lb_loss_coef * L_LB (load balancing loss chuan cua MoE, tinh tren cac router nam
       trong khoang layer duoc gan LoRA).

Cac tinh nang chinh (GIONG HET setting cua Qwen1.5-MoE-A2.7B-SNLI.py / code alignment):
  1. LoRA chi ap dung tren middle layers [L/3, 2L/3), chi len attention / router / experts,
     moi thanh phan mot rank rieng (mac dinh: router r=4, attention r=16, experts r=16),
     alpha=32, dropout=0.05.
  2. Checkpointing + resume tai bat ky epoch/step nao, luu moi ~200 step (--save_steps).
  3. Luu LoRA weight tai training/english-task-only/checkpoints/Qwen1.5-MoE-A2.7B-SQuAD-Task-Only/.
  4. Dynamic batching: batch_size mac dinh 64 (context SQuAD dai hon nhieu so voi SNLI nen batch
     nho hon), khi OOM thi chia doi de tri (dequy), clear memory sau moi lan chia, skip sample
     neu OOM ca khi batch_size = 1, tra ve batch_size goc ngay cho batch tiep theo.
  5. Ho tro DDP (torchrun) giong het code SNLI/alignment.
  6. 3 epoch, tqdm day du.
  7. Sau moi ~200 step (--save_steps) thi luu checkpoint local + push len HuggingFace Hub
     (--push_to_hub, mac dinh True) voi ten repo la --hub_model_id
     (mac dinh: Qwen1.5-MoE-A2.7B-SQuAD-Task-Only).
  8. lb_loss_coef (lambda load balancing) = 0.01, KHONG fallback ve config nhu code alignment.

DIEM KHAC BIET so voi Qwen1.5-MoE-A2.7B-SNLI.py (do ban chat task khac nhau):
  - SNLI: 3 lop phan loai co dinh (entailment/neutral/contradiction) -> nhan luon la 1 tu duy nhat.
  - SQuAD: day la task QA TRICH XUAT (extractive), cau tra loi la 1 doan text TU DO trich tu
    context. Moi sample (context, question, answer_text) duoc chuyen thanh 1 prompt dang
    instruction:

        Context: <context>
        Question: <question>
        Answer: <answer_text><eos>

    Loss LM chuan (cross-entropy) van chi tinh tren phan "<answer_text><eos>" (prompt Context/
    Question bi mask -100), giong het co che SFT cua SNLI.

  - Context SQuAD co the RAT DAI (vai tram token), khac han cau ngan cua SNLI. Neu de tokenizer
    truncate binh thuong (truncation_side mac dinh = "right"), phan BI CAT se la chinh phan
    "Answer: <answer_text>" nam o CUOI chuoi -> hong nhan hoan toan. Vi vay script nay dung
    mot buoc rieng goi la "context windowing":
      + Voi nhung sample co full_text vuot qua --max_length, KHONG truncate tho, ma tim 1
        "cua so" (window) cac TU trong context bao quanh vi tri cua answer (dua vao
        answer_start), roi mo rong dan sang trai/phai (giu nguyen tu, khong cat giua tu) cho
        toi khi vua du ngan sach token cho phep. Nho vay context van chua answer, chi mat bot
        cac cau/doan KHONG lien quan truc tiep, thay vi mat luon ca nhan.
      + Neu ngay ca voi window toi thieu (chi vua du cac tu cua answer) van vuot --max_length
        (truong hop hiem, cau tra loi qua dai), sample do bi bo qua (skip + log).
  - Cac cau hoi "khong co dap an" (SQuAD 2.0 style, answers.text rong) bi loc bo ngay khi doc
    du lieu, vi muc tieu la finetune cho zero-shot eval tren cac benchmark dang SQuAD 1.1 nhu
    XQuAD (toan bo cau hoi trong XQuAD deu CO dap an).
  - Diagnostics "label_acc" o day la ty le token DUNG (teacher forcing) tren TOAN BO cac token
    cua answer (khong chi token dau tien nhu SNLI), vi answer cua SQuAD thuong la CUM TU nhieu
    token chu khong phai 1 tu don nhu SNLI.

Kien truc Qwen2MoeForCausalLM (router/experts dat ten theo quy uoc chuan cua HF transformers)
van duoc TU DONG DO TIM module attention / router / experts bang regex (khong hard-code),
giong het co che cua code SNLI/alignment.

Vi du chay:
    python Qwen1.5-MoE-A2.7B-SQuAD.py \
        --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
        --data_file data/english_task/squad/train.json \
        --push_to_hub

Resume:
    python Qwen1.5-MoE-A2.7B-SQuAD.py --resume_from_checkpoint auto

Sau khi train xong, model (LoRA adapter) duoc push len HuggingFace Hub. De load lai va zero-shot
eval tren mot task dang SQuAD khac (vi du XQuAD), chi can:

    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, "ducanhdinh/Qwen1.5-MoE-A2.7B-SQuAD-Task-Only")
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
    # generate() tu prompt nay, cat lay phan sinh ra truoc eos token la cau tra loi du doan.
"""

import argparse
import datetime
import gc
import glob
import json
import logging
import os
import random
import re
import time
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence, Tuple

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
logger = logging.getLogger("qwen15_moe_squad_finetune")


# Cac ten bien moi truong pho bien cho HF token, thu theo thu tu nay
_HF_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")

# Regex tach "tu" (tu = 1 chuoi ky tu khong-whitespace lien tuc), dung de xac dinh bien tu/cau
# trong context khi can lam context windowing (xem select_context_window()).
WORD_SPAN_PATTERN = re.compile(r"\S+")


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
    p = argparse.ArgumentParser(description="LoRA finetuning cho MoE Qwen1.5-MoE-A2.7B tren task SQuAD")

    # Model / data / output
    p.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen1.5-MoE-A2.7B")
    p.add_argument("--data_file", type=str, default="data/english_task/squad/train.json",
                    help="Duong dan file JSON SQuAD (list cac object "
                         "{id, title, context, question, answers:{text:[...], answer_start:[...]}}).")
    p.add_argument("--output_dir", type=str,
                    default="training/english-task-only/checkpoints/Qwen1.5-MoE-A2.7B-SQuAD-Task-Only")
    p.add_argument("--max_samples", type=int, default=None,
                    help="Gioi han so sample (debug/smoke test), None = dung het du lieu")

    # Hugging Face Hub
    p.add_argument("--push_to_hub", action="store_true", default=True)
    p.add_argument("--no_push_to_hub", dest="push_to_hub", action="store_false")
    p.add_argument("--hub_model_id", type=str, default="ducanhdinh/Qwen1.5-MoE-A2.7B-SQuAD-Task-Only",
                    help="Doi lai namespace/username HF cua ban neu khac 'ducanhdinh'.")
    p.add_argument("--hub_private", action="store_true")
    p.add_argument("--env_file", type=str, default=".env",
                    help="Duong dan file .env chua HF_TOKEN, tu dong nap bang python-dotenv")
    p.add_argument("--hf_token", type=str, default=None,
                    help="Override HF token thu cong, uu tien cao hon .env/bien moi truong")

    # Training schedule (giong het code SNLI/alignment)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64,
                    help="Nho hon SNLI vi context SQuAD dai hon nhieu (tranh OOM).")
    p.add_argument("--min_batch_size", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512,
                    help="Context SQuAD dai hon SNLI rat nhieu -> mac dinh 512. Sample vuot qua "
                         "gioi han nay se duoc 'windowing' (xem select_context_window) thay vi "
                         "bi truncate tho.")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_clip_norm", type=float, default=1.0)

    # MoE loss
    p.add_argument("--lb_loss_coef", type=float, default=0.01,
                    help="He so (lambda) cho load-balancing loss. Mac dinh 0.01 theo yeu cau "
                         "(KHAC voi code alignment, code do fallback ve router_aux_loss_coef "
                         "cua config neu khong truyen).")
    p.add_argument("--num_local_experts", type=int, default=None,
                    help="Override so luong experts, None = tu doc trong config model")
    p.add_argument("--num_experts_per_tok", type=int, default=None,
                    help="Override top-k router, None = tu doc trong config model")

    # LoRA - rank rieng cho tung thanh phan kien truc (goi bang PEFT rank_pattern) — GIONG
    # HET gia tri mac dinh cua code SNLI/alignment.
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
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--device_map", type=str, default=None,
                    help="vi du 'auto' cho multi-GPU. Neu set thi bo qua --device")
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--diagnostics_dir", type=str, default=None,
                    help="None = <output_dir>/diagnostics")
    p.add_argument("--log_every", type=int, default=10, help="Cap nhat plot loss moi N step")
    p.add_argument("--smooth_window", type=int, default=50,
                    help="So step lien tiep duoc gom lai (tinh trung binh cong) cho moi diem "
                         "tren cac duong *_smoothed.png.")

    # Distributed (DDP qua torchrun: doc RANK / LOCAL_RANK / WORLD_SIZE tu bien moi truong,
    # khong can truyen tay). Chay 1 GPU binh thuong neu khong launch qua torchrun.
    p.add_argument("--nccl_timeout_minutes", type=int, default=30,
                    help="Timeout cho moi collective op cua NCCL (mac dinh PyTorch la 10 phut, "
                         "tang len 30 phut de chiu duoc checkpoint/push cham).")

    return p


# ============================================================================================
# Utils chung (giong het code SNLI/alignment)
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
# Distributed (DDP qua torchrun) — giong het code SNLI/alignment
# ============================================================================================
def setup_distributed(nccl_timeout_minutes: int):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if not is_distributed:
        return rank, local_rank, world_size, is_distributed, None

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training (DDP) yeu cau CUDA (backend nccl).")

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
    return model.module if hasattr(model, "module") else model


def sync_grads_across_ranks(trainable_params, world_size: int):
    """All-reduce (trung binh) gradient THU CONG, gop thanh 1 buffer lien tuc de giam so
    luong NCCL call (xem giai thich chi tiet trong code alignment finetuning)."""
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    grads_by_dtype = {}
    for p in trainable_params:
        if p.grad is not None:
            grads_by_dtype.setdefault(p.grad.dtype, []).append(p.grad)

    for dtype, grads in grads_by_dtype.items():
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world_size)
        for g, synced in zip(grads, _unflatten_dense_tensors(flat, grads)):
            g.copy_(synced)


def reduce_result_across_ranks(result: dict, device, world_size: int) -> dict:
    """Gop loss / accuracy / so sample cua tat ca rank lai de log/plot phan anh dung so lieu
    TOAN CUC, va de dong bo quyet dinh do_step giua cac rank."""
    loss_t = torch.tensor(
        [result["lm_loss"], result["lb_loss"], result["total_loss"], result["label_acc"]],
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
        "label_acc": loss_t[3].item(),
        "n_processed": int(count_t[0].item()),
        "n_skipped": int(count_t[1].item()),
    }


# ============================================================================================
# Du lieu SQuAD: doc JSON -> build prompt instruction + mask cau tra loi cho tung sample
# ============================================================================================
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


def build_prompt(context: str, question: str) -> str:
    """Prompt dang instruction cho extractive QA. Phan sau 'Answer:' la phan model phai sinh ra
    va la phan DUY NHAT duoc tinh loss (xem build_full_text + mask trong
    forward_backward_one_subbatch). Dung format nay de tuong thich truc tiep voi cach eval
    zero-shot pho bien cho SQuAD/XQuAD (prompt giong het khi generate o eval, chi bo phan dap an)."""
    return (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def build_full_text(context: str, question: str, answer_text: str, eos_token: str) -> Tuple[str, str]:
    prompt = build_prompt(context, question)
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


def build_examples(records: List[Dict], tokenizer, eos_token: str, max_length: int) -> Tuple[List[Dict], List[int]]:
    """Tien xu ly: moi record -> {"prompt", "full_text", "answer_text"}.
    Sample nao co full_text vuot qua max_length se duoc "windowing" lai context (xem
    select_context_window); neu van khong vua sau khi windowing (hiem) thi bi bo qua.
    Tra ve (examples, lengths) da loc, dong bo index voi nhau — dung truc tiep cho
    LengthGroupedBatchSampler, tranh phai tokenize lai toan bo lan nua."""
    # Buoc 1: build naive (chua windowing) cho toan bo, tinh do dai token 1 lan (batch, nhanh).
    naive_examples = []
    for rec in records:
        prompt, full_text = build_full_text(rec["context"], rec["question"], rec["answer_text"], eos_token)
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
        new_prompt, new_full_text = build_full_text(windowed_context, ex["question"], ex["answer_text"], eos_token)
        new_len = len(tokenizer(new_full_text, add_special_tokens=True)["input_ids"])
        if new_len > max_length:
            n_dropped_too_long += 1
            continue

        examples.append({"prompt": new_prompt, "full_text": new_full_text})
        lengths.append(new_len)
        n_windowed += 1

    logger.info(
        f"build_examples: {len(examples)} sample giu lai (trong do {n_windowed} sample da duoc "
        f"windowing context vi vuot max_length={max_length}), bo qua {n_dropped_too_long} sample "
        f"van qua dai ngay ca sau khi windowing."
    )
    return examples, lengths


class SQuADDataset(Dataset):
    """Moi sample la 1 dict {"prompt", "full_text"} (da build/windowing san), tokenize thuc su
    duoc lam sau trong vong lap training de ho tro chia nho batch khi OOM."""

    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class LengthGroupedBatchSampler(Sampler[List[int]]):
    """Giong het code SNLI/alignment: moi epoch shuffle toan bo index (TREN TOAN BO DATASET,
    CHUA chia rank) -> sort theo do dai token -> gom mega-batch -> shuffle thu tu mega-batch ->
    chia deu LIEN TIEP cho cac rank, giup moi rank luon nhan sample co do dai xap xi nhau (giam
    straggler effect khi chay DDP nhieu GPU)."""

    def __init__(self, lengths: List[int], batch_size: int, world_size: int = 1,
                 rank: int = 0, seed: int = 42):
        self.lengths = lengths
        self.batch_size = batch_size
        self.world_size = max(world_size, 1)
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _build_rank_batches(self) -> List[List[int]]:
        g = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        g.shuffle(indices)
        indices.sort(key=lambda i: self.lengths[i])

        mega_size = self.batch_size * self.world_size
        n_mega = len(indices) // mega_size
        indices = indices[: n_mega * mega_size]
        mega_batches = [indices[i:i + mega_size] for i in range(0, len(indices), mega_size)]
        g.shuffle(mega_batches)

        start = self.rank * self.batch_size
        end = start + self.batch_size
        return [mb[start:end] for mb in mega_batches]

    def __iter__(self):
        for b in self._build_rank_batches():
            yield b

    def __len__(self):
        mega_size = self.batch_size * self.world_size
        return len(self.lengths) // mega_size


# ============================================================================================
# Tu dong tim target module cho LoRA: attention / router / experts trong middle layers
# (giong het code SNLI/alignment)
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
    """Tra ve (targets, kinds): targets la list ten module Linear duoc chon lam LoRA target
    trong khoang layer_indices, kinds la dict ten module -> "router" / "attention" / "expert",
    dung de gan rank rieng cho tung thanh phan qua rank_pattern cua PEFT."""
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
# MoE loss chuan: LM loss (chi tren answer) + Load Balancing loss (Switch/Mixtral style)
# ============================================================================================
def compute_load_balancing_loss(router_logits_list: List[torch.Tensor], attention_mask: torch.Tensor,
                                 num_experts: int, top_k: int):
    """Giong het code SNLI/alignment: loai bo padding truoc khi tinh thong ke, noi (concat)
    token cua TAT CA router layer da hook lai thanh 1 tap DUY NHAT roi moi tinh f_i / P_i / loss
    (dung cach cua HF load_balancing_loss_func, KHONG lay mean loss rieng tung layer vi ve toan
    hoc khong tuong duong)."""
    mask_flat = attention_mask.reshape(-1).bool()

    valid_logits = []
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
        if logits.shape[0] > 0:
            valid_logits.append(logits)

    if not valid_logits:
        return torch.tensor(0.0, device=attention_mask.device)

    concatenated_logits = torch.cat(valid_logits, dim=0)
    routing_weights = F.softmax(concatenated_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = F.one_hot(selected_experts, num_experts).float()
    tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # f_i, da chia trung binh theo token
    avg_prob_per_expert = routing_weights.mean(dim=0)
    return num_experts * torch.sum(tokens_per_expert * avg_prob_per_expert)


def forward_backward_one_subbatch(sub_examples, tokenizer, model, max_length, device,
                                   router_logits_cache, num_experts, top_k, lb_loss_coef,
                                   loss_weight):
    """Tokenize + forward + backward cho 1 sub-batch (co the la toan bo batch hoac 1 mieng
    sau khi chia doi vi OOM). Moi sample trong sub_examples la dict {"prompt", "full_text"}
    (da windowing san neu can, xem build_examples).

    L_LM van la cross-entropy TIEU CHUAN cho next-token prediction, nhung CHI tinh tren phan
    "<answer_text><eos>" (phan prompt Context/Question + padding bi mask -100), giong cach lam
    SFT/instruction tuning tieu chuan: model duoc day toan bo context+question lam context
    (khong ton loss) roi chi hoc sinh ra dung cau tra loi.

    Tra ve (lm_loss_val, lb_loss_val, total_loss_val, token_acc, n_samples). token_acc la ty le
    token DUNG (teacher forcing) tren TOAN BO cac token cua answer — chi la diagnostics de theo
    doi qua trinh train, KHONG dung trong loss va KHONG phai la Exact Match/F1 chuan cua SQuAD."""
    full_texts = [ex["full_text"] for ex in sub_examples]
    prompts = [ex["prompt"] for ex in sub_examples]

    enc = tokenizer(full_texts, padding=True, truncation=True, max_length=max_length,
                     return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100  # mask padding

    seq_len = labels.shape[1]
    # Do dai token cua prompt (ke ca special token neu tokenizer tu them) -> dung de mask
    # toan bo phan prompt trong labels, chi giu lai phan "<answer_text><eos>" de tinh loss.
    prompt_lens = []
    for p in prompts:
        ids = tokenizer(p, add_special_tokens=True, truncation=True, max_length=max_length)["input_ids"]
        prompt_lens.append(min(len(ids), seq_len))
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100

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

    # Diagnostics: ty le token DUNG (xap xi, teacher forcing) tren TOAN BO token cua answer
    # (khac SNLI vi answer SQuAD thuong la cum nhieu token chu khong phai 1 tu don) — chi de
    # theo doi qua trinh train, KHONG dung trong loss.
    with torch.no_grad():
        answer_mask = shift_labels != -100
        if answer_mask.any():
            preds = shift_logits.argmax(dim=-1)
            token_acc = (preds[answer_mask] == shift_labels[answer_mask]).float().mean().item()
        else:
            token_acc = 0.0

    return lm_loss.item(), lb_loss.item(), total_loss.item(), token_acc, len(sub_examples)


def run_batch_with_dynamic_oom_handling(batch_examples, tokenizer, model, max_length, device,
                                         router_logits_cache, num_experts, top_k, lb_loss_coef,
                                         min_batch_size):
    """Chay 1 batch (list example dict). Neu OOM: clear memory, chia doi, de quy. Neu OOM ca
    khi size = 1 (hoac == min_batch_size) thi skip sample do. Luon quay ve batch_size goc cho
    batch tiep theo (giong het co che cua code SNLI/alignment, xem docstring goc ve ly do tach
    _attempt/_run de giai phong dung frame bi OOM truoc khi retry)."""
    original_size = len(batch_examples)
    agg = {"lm_loss": 0.0, "lb_loss": 0.0, "total_loss": 0.0, "label_acc": 0.0, "n_ok": 0, "n_skipped": 0}

    def _attempt(sub_examples) -> bool:
        try:
            lm, lb, tot, acc, n = forward_backward_one_subbatch(
                sub_examples, tokenizer, model, max_length, device,
                router_logits_cache, num_experts, top_k, lb_loss_coef,
                loss_weight=len(sub_examples) / max(original_size, 1),
            )
        except RuntimeError as e:
            if not is_oom_error(e):
                raise
            return False
        agg["lm_loss"] += lm * n
        agg["lb_loss"] += lb * n
        agg["total_loss"] += tot * n
        agg["label_acc"] += acc * n
        agg["n_ok"] += n
        return True

    def _run(sub_examples):
        if _attempt(sub_examples):
            return

        router_logits_cache.clear()
        clear_memory()

        if len(sub_examples) <= max(min_batch_size, 1):
            logger.warning(f"OOM ngay ca voi sub-batch size={len(sub_examples)} -> skip sample nay.")
            agg["n_skipped"] += len(sub_examples)
            return
        mid = len(sub_examples) // 2
        logger.warning(f"OOM voi sub-batch size={len(sub_examples)} -> chia doi thanh {mid} + {len(sub_examples) - mid}.")
        _run(sub_examples[:mid])
        _run(sub_examples[mid:])

    _run(batch_examples)
    n = max(agg["n_ok"], 1)
    return {
        "lm_loss": agg["lm_loss"] / n,
        "lb_loss": agg["lb_loss"] / n,
        "total_loss": agg["total_loss"] / n,
        "label_acc": agg["label_acc"] / n,
        "n_processed": agg["n_ok"],
        "n_skipped": agg["n_skipped"],
    }


# ============================================================================================
# Checkpoint / resume (giong het code SNLI/alignment)
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
        candidates = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if candidates:
            candidates.sort(key=lambda p: int(p.rsplit("-", 1)[-1]))
            return candidates[-1]
        return None
    return resume_arg if os.path.isdir(resume_arg) else None


# ============================================================================================
# Diagnostics: jsonl + plot (loss + accuracy)
# ============================================================================================
def log_step_to_jsonl(jsonl_path, global_step, epoch, result):
    rec = {
        "step": global_step,
        "epoch": epoch,
        "lm_loss": result["lm_loss"],
        "lb_loss": result["lb_loss"],
        "total_loss": result["total_loss"],
        "label_acc": result["label_acc"],
        "n_processed": result["n_processed"],
        "n_skipped": result["n_skipped"],
        "timestamp": time.time(),
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_jsonl(jsonl_path):
    steps, lm, lb, total, acc = [], [], [], [], []
    if not os.path.exists(jsonl_path):
        return steps, lm, lb, total, acc
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
            acc.append(rec.get("label_acc", 0.0))
    return steps, lm, lb, total, acc


def plot_losses(jsonl_path, out_png):
    steps, lm, lb, total, _ = _read_jsonl(jsonl_path)
    if not steps:
        return
    plt.figure(figsize=(10, 6))
    plt.plot(steps, lm, label="L_LM (chi tren answer)")
    plt.plot(steps, lb, label="L_LB")
    plt.plot(steps, total, label="L_Total")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Qwen1.5-MoE-A2.7B-SQuAD-Task-Only LoRA finetuning loss (tho, tung step)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_losses_smoothed(jsonl_path, out_png, window: int):
    steps, lm, lb, total, _ = _read_jsonl(jsonl_path)
    if not steps or window <= 1:
        return

    def bin_mean(values):
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
    plt.title(f"Qwen1.5-MoE-A2.7B-SQuAD-Task-Only LoRA finetuning loss - trung binh moi {window} step")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_accuracy(jsonl_path, out_png, window: int):
    """Bieu do rieng cho label_acc (xap xi, teacher forcing tren TOAN BO token cua answer), da
    lam min theo `window` step lien tiep giong plot_losses_smoothed, vi ban chat acc theo tung
    step rat nhieu nhieu (batch nho / lech nhau)."""
    steps, _, _, _, acc = _read_jsonl(jsonl_path)
    if not steps:
        return
    window = max(window, 1)
    xs, ys = [], []
    for i in range(0, len(acc), window):
        chunk = acc[i:i + window]
        xs.append(steps[i + len(chunk) - 1])
        ys.append(sum(chunk) / len(chunk))
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, label="Answer token accuracy (train, xap xi)", color="green", marker="o", markersize=3)
    plt.xlabel(f"Training step (moi diem = trung binh cong cua {window} step lien tiep)")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("Qwen1.5-MoE-A2.7B-SQuAD-Task-Only — ty le token dung cua answer tren train batch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_all(jsonl_path, plot_path, plot_path_smoothed, plot_path_acc, smooth_window):
    plot_losses(jsonl_path, plot_path)
    plot_losses_smoothed(jsonl_path, plot_path_smoothed, smooth_window)
    plot_accuracy(jsonl_path, plot_path_acc, smooth_window)


# ============================================================================================
# Hugging Face Hub push
# ============================================================================================
def build_model_card(args, num_experts, top_k, layer_start, layer_end, num_layers) -> str:
    return f"""---
license: apache-2.0
base_model: {args.model_name_or_path}
datasets:
- squad
tags:
- lora
- peft
- moe
- mixture-of-experts
- question-answering
- squad
- fine-tuned
---

# Qwen1.5-MoE-A2.7B-SQuAD-Task-Only

Day la LoRA adapter finetune tu [`{args.model_name_or_path}`]\
(https://huggingface.co/{args.model_name_or_path}), mot mo hinh Mixture-of-Experts, tren task
SQuAD (extractive Question Answering) — sinh cau tra loi dang free text trich tu doan van
(context) de tra loi cau hoi.

## Prompt format
```
Context: <context>
Question: <question>
Answer: <answer_text>
```
Phan sau `Answer:` la phan model sinh ra. Prompt nay duoc thiet ke de dung TRUC TIEP cho
zero-shot eval tren cac benchmark QA extractive khac cung dinh dang, vi du XQuAD: chi can
generate() tu prompt (khong co phan dap an) roi cat lay text truoc token eos.

## Cau hinh LoRA
- Layer duoc finetune: `[{layer_start}, {layer_end})` trong tong so `{num_layers}` layer
  (tuong ung khoang 1L/3 -> 2L/3), giong het setting cua code SNLI/alignment finetuning.
- Module duoc gan LoRA: **attention**, **router**, **experts** trong khoang layer tren, moi
  thanh phan mot rank rieng qua `rank_pattern` cua PEFT:
  - attention: r = {args.lora_r_attention}
  - router: r = {args.lora_r_router}
  - experts: r = {args.lora_r_experts}
- alpha = {args.lora_alpha}, dropout = {args.lora_dropout}

## Loss
Loss MoE tieu chuan:

`L_total = L_LM + lb_loss_coef * L_LB`

- `L_LM`: cross-entropy chuan tren token tiep theo, CHI tinh tren phan `<answer_text><eos>`
  (prompt Context/Question bi mask, giong instruction-tuning/SFT tieu chuan).
- `L_LB`: load balancing loss chuan cua MoE (Switch/Mixtral style), tinh tren cac router nam
  trong khoang layer duoc finetune.
- `lb_loss_coef` (lambda) = {args.lb_loss_coef}
- `num_experts` = {num_experts}, `top_k` = {top_k}

## Du lieu
SQuAD, doc tu `{args.data_file}` (list object
`{{"id", "title", "context", "question", "answers": {{"text", "answer_start"}}}}`). Cac cau hoi
khong co dap an (SQuAD 2.0 style) bi loc bo. Context qua dai (vuot `--max_length`) duoc "windowing"
quanh vi tri answer thay vi bi truncate tho, de dam bao answer luon nam trong phan context duoc
giu lai (xem `select_context_window` trong script training).

## Diagnostics
Xem `diagnostics/loss_log.jsonl` (log theo tung step), `diagnostics/loss_curve.png`,
`diagnostics/loss_curve_smoothed.png` va `diagnostics/accuracy_curve.png` (ty le token dung xap
xi tren TOAN BO token cua answer — teacher forcing, tinh tren batch train, da lam min moi
`{args.smooth_window}` step; day KHONG phai Exact Match/F1 chuan cua SQuAD, chi la diagnostics
theo doi qua trinh train).
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
                "Khong dung --device_map cung luc voi distributed training (torchrun)."
            )
        args.device = ddp_device
        if rank != 0:
            logger.setLevel(logging.WARNING)

    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    diagnostics_dir = args.diagnostics_dir or os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = os.path.join(diagnostics_dir, "loss_log.jsonl")
    plot_path = os.path.join(diagnostics_dir, "loss_curve.png")
    plot_path_smoothed = os.path.join(diagnostics_dir, "loss_curve_smoothed.png")
    plot_path_acc = os.path.join(diagnostics_dir, "accuracy_curve.png")

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
    logger.info(f"lb_loss_coef (lambda load balancing) = {lb_loss_coef}")

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

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # ------------------------------------------------------------------------------------ data
    logger.info(f"Dang doc du lieu SQuAD tu {args.data_file} ...")
    records = load_squad_records(args.data_file)
    if args.max_samples:
        random.Random(args.seed).shuffle(records)
        records = records[: args.max_samples]
    logger.info(f"Tong so sample SQuAD su dung: {len(records)}")
    if len(records) == 0:
        raise RuntimeError("Khong doc duoc sample nao — kiem tra lai --data_file.")

    examples, lengths = build_examples(records, tokenizer, tokenizer.eos_token, args.max_length)
    if len(examples) == 0:
        raise RuntimeError(
            "Khong con sample nao sau buoc loc do dai — thu tang --max_length."
        )

    mega_size = args.batch_size * world_size
    if len(examples) < mega_size:
        raise RuntimeError(
            f"Chi co {len(examples)} sample, khong du de tao 1 step voi batch_size="
            f"{args.batch_size} x world_size={world_size} (can it nhat {mega_size} sample)."
        )
    n_dropped = len(examples) % mega_size
    if n_dropped and is_main_process(rank):
        logger.info(f"Moi epoch se bo {n_dropped} sample cuoi (trong tong {len(examples)}) de "
                    f"chia deu {args.batch_size} sample/rank x {world_size} rank cho MOI step.")

    dataset = SQuADDataset(examples)
    batch_sampler = LengthGroupedBatchSampler(lengths, batch_size=args.batch_size,
                                               world_size=world_size, rank=rank, seed=args.seed)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=lambda b: b)

    # ------------------------------------------------------------------------------- optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate,
                                weight_decay=args.weight_decay, foreach=True)

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

    readme_text = build_model_card(args, num_experts, top_k, layer_start, layer_end, num_layers)

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
            for step_in_epoch, batch_examples in pbar:
                if step_in_epoch < step_offset:
                    continue

                model.train()
                optimizer.zero_grad(set_to_none=not is_distributed)

                with no_sync_ctx():
                    result = run_batch_with_dynamic_oom_handling(
                        batch_examples=batch_examples,
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
                    "tok_acc": f"{result['label_acc']:.3f}",
                    "skipped": result["n_skipped"],
                })

                if is_main_process(rank):
                    if result["n_processed"] > 0:
                        log_step_to_jsonl(jsonl_path, global_step, epoch, result)

                    if global_step % args.log_every == 0:
                        plot_all(jsonl_path, plot_path, plot_path_smoothed, plot_path_acc, args.smooth_window)

                    if global_step % args.save_steps == 0:
                        ckpt_dir = save_checkpoint(args.output_dir, unwrap_model(model), optimizer,
                                                    scheduler, epoch, step_in_epoch, global_step)
                        plot_all(jsonl_path, plot_path, plot_path_smoothed, plot_path_acc, args.smooth_window)
                        logger.info(f"Da luu checkpoint local: {ckpt_dir}")
                        if args.push_to_hub:
                            push_to_hub(ckpt_dir, diagnostics_dir, args.hub_model_id,
                                        args.hub_private, readme_text)
                            logger.info(f"Da push checkpoint len hub: {args.hub_model_id}")

                if is_distributed and global_step % args.save_steps == 0:
                    dist.barrier()

            start_step_in_epoch = 0

        if is_main_process(rank):
            final_ckpt = save_checkpoint(args.output_dir, unwrap_model(model), optimizer, scheduler,
                                          args.num_train_epochs - 1, steps_per_epoch - 1, global_step)
            plot_all(jsonl_path, plot_path, plot_path_smoothed, plot_path_acc, args.smooth_window)
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
            plot_all(jsonl_path, plot_path, plot_path_smoothed, plot_path_acc, args.smooth_window)
        raise
    finally:
        for h in hooks:
            h.remove()
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()