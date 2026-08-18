"""
Fine-tuning LoRA cho mo hinh Mixture-of-Experts ATH-MaaS/Marco-Nano-Instruct.

Loss = L_LM (cross-entropy chuan) + lb_loss_coef * L_LB (load balancing loss chuan cua MoE,
tinh tren cac router nam trong khoang layer duoc gan LoRA).

Cac tinh nang chinh:
  1. LoRA chi ap dung tren middle layers [L/3, 2L/3), chi len attention / router / experts.
  2. Checkpointing + resume tai bat ky epoch/step nao, luu moi 1000 step.
  3. Luu LoRA weight tai training/finetuning/checkpoints/Macro-Nano-Instruct/.
  4. Dynamic batching: batch_size mac dinh 512, khi OOM thi chia doi de tri (dequy),
     clear memory sau moi lan chia, skip sample neu OOM ca khi batch_size = 1,
     tra ve batch_size goc ngay cho batch tiep theo.
  5. Dataset/DataLoader gom sample tu ca 3 file flores/bible/ntrex, shuffle roi sort theo do dai.
  6. 3 epoch, tqdm day du.
  7. argparse day du de tuy bien.

Vi kien truc chi tiet cua Marco-Nano-Instruct khong duoc cung cap truoc, script nay
TU DONG DO TIM cac module attention / router / experts bang ten (regex) thay vi hard-code,
va cho phep override qua CLI neu can.

Vi du chay:
    python Macro-Nano-Instruct.py \
        --model_name_or_path ATH-MaaS/Marco-Nano-Instruct \
        --data_dir data/processed_alignment \
        --push_to_hub

Resume:
    python Macro-Nano-Instruct.py --resume_from_checkpoint auto
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
import time
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
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
logger = logging.getLogger("marco_nano_finetune")


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
    p = argparse.ArgumentParser(description="LoRA finetuning cho MoE Marco-Nano-Instruct")

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
                         "fallback 0.01 (trong so nhu finetune binh thuong)")
    p.add_argument("--num_local_experts", type=int, default=None,
                    help="Override so luong experts, None = tu doc trong config model")
    p.add_argument("--num_experts_per_tok", type=int, default=None,
                    help="Override top-k router, None = tu doc trong config model")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_layer_start_ratio", type=float, default=1.0 / 3.0)
    p.add_argument("--lora_layer_end_ratio", type=float, default=2.0 / 3.0)

    # Checkpoint / resume
    p.add_argument("--save_steps", type=int, default=1000)
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

    losses = []
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
        if logits.shape[0] == 0:
            continue
        routing_weights = F.softmax(logits, dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # [tokens, top_k]
        expert_mask = F.one_hot(selected_experts, num_experts).float()  # [tokens, top_k, num_experts]
        # LUU Y: .mean(dim=0) da chia trung binh theo so token roi (cho ra f_i dung chuan,
        # trong khoang [0,1]) -> KHONG duoc chia them cho logits.shape[0] mot lan nua (bug
        # cu chia 2 lan lam L_LB nho gia tao ~N lan, N = so token trong sub-batch dang tinh).
        tokens_per_expert = expert_mask.sum(dim=1).mean(dim=0)  # [num_experts], = f_i
        avg_prob_per_expert = routing_weights.mean(dim=0)  # [num_experts]
        loss = num_experts * torch.sum(tokens_per_expert * avg_prob_per_expert)
        losses.append(loss)
    if not losses:
        return torch.tensor(0.0, device=attention_mask.device)
    return torch.stack(losses).mean()


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
    plt.title("Marco-Nano-Instruct LoRA finetuning loss")
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

# Macro-Nano-Instruct-Finetuning

Day la LoRA adapter finetune tu [`{args.model_name_or_path}`]\
(https://huggingface.co/{args.model_name_or_path}), mot mo hinh Mixture-of-Experts.

## Cau hinh LoRA
- Layer duoc finetune: `[{layer_start}, {layer_end})` trong tong so `{num_layers}` layer
  (tuong ung khoang 1L/3 -> 2L/3).
- Module duoc gan LoRA: **attention**, **router**, **experts** trong khoang layer tren.
- r = {args.lora_r}, alpha = {args.lora_alpha}, dropout = {args.lora_dropout}

## Loss
Loss MoE tieu chuan:

`L_total = L_LM + lb_loss_coef * L_LB`

- `L_LM`: cross-entropy chuan tren token tiep theo.
- `L_LB`: load balancing loss chuan cua MoE (Switch/Mixtral style), tinh tren cac router
  nam trong khoang layer duoc finetune.
- `lb_loss_coef` = {args.lb_loss_coef}
- `num_experts` = {num_experts}, `top_k` = {top_k}

## Du lieu
Cau don ngu duoc gom tu 3 bo du lieu alignment: `flores.json`, `bible.json`, `ntrex.json`
(moi field ngon ngu trong 1 record duoc coi la 1 sample), shuffle va sort theo do dai token
truoc khi gom batch.

## Diagnostics
Xem `diagnostics/loss_log.jsonl` (log theo tung step) va `diagnostics/loss_curve.png`
(bieu do L_LM / L_LB / L_Total theo step).
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

    os.makedirs(args.output_dir, exist_ok=True)
    diagnostics_dir = args.diagnostics_dir or os.path.join(args.output_dir, "diagnostics")
    os.makedirs(diagnostics_dir, exist_ok=True)
    jsonl_path = os.path.join(diagnostics_dir, "loss_log.jsonl")
    plot_path = os.path.join(diagnostics_dir, "loss_curve.png")

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

    target_modules = build_lora_target_modules(base_model, layer_indices)
    if not target_modules:
        raise RuntimeError(
            "Khong tim thay module (attention/router/experts) nao trong khoang layer da chon. "
            "Kien truc model co the dat ten khac quy uoc — kiem tra lai regex trong build_lora_target_modules()."
        )
    router_target_names = [n for n in target_modules if is_router_leaf_name(n)]
    logger.info(f"Tim thay {len(target_modules)} target module cho LoRA "
                f"({len(router_target_names)} router). Vi du: {target_modules[:8]}")

    num_experts, top_k = infer_moe_dims(base_model.config, args)

    lb_loss_coef = args.lb_loss_coef
    if lb_loss_coef is None:
        lb_loss_coef = float(getattr(base_model.config, "router_aux_loss_coef", 0.01))
    logger.info(f"lb_loss_coef (trong so load-balancing, nhu finetune binh thuong) = {lb_loss_coef}")

    # ---------------------------------------------------------------------------- resume / LoRA
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
    model.print_trainable_parameters()
    if not args.device_map:
        model.to(args.device)

    router_logits_cache: list = []
    hooks = register_router_hooks(model, router_target_names, router_logits_cache)

    # ------------------------------------------------------------------------------------ data
    logger.info(f"Dang doc du lieu tu {args.data_dir} ({args.data_files}) ...")
    texts = load_all_sentences(args.data_dir, args.data_files)
    if args.max_samples:
        random.Random(args.seed).shuffle(texts)
        texts = texts[: args.max_samples]
    logger.info(f"Tong so sample (cau) sau khi gom ca 3 bo du lieu: {len(texts)}")
    if len(texts) == 0:
        raise RuntimeError("Khong doc duoc sample nao — kiem tra lai --data_dir / --data_files.")

    lengths = compute_lengths(tokenizer, texts)
    dataset = SentenceDataset(texts)
    batch_sampler = LengthGroupedBatchSampler(lengths, batch_size=args.batch_size, seed=args.seed)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=lambda b: b)

    # ------------------------------------------------------------------------------- optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

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

            pbar = tqdm(
                enumerate(dataloader),
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{args.num_train_epochs}",
            )
            for step_in_epoch, batch_texts in pbar:
                if step_in_epoch < step_offset:
                    # dang resume: bo qua nhanh cac batch da xu ly o lan chay truoc
                    continue

                model.train()
                optimizer.zero_grad(set_to_none=True)

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

                if result["n_processed"] > 0:
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

                if result["n_processed"] > 0:
                    log_step_to_jsonl(jsonl_path, global_step, epoch, result)

                if global_step % args.log_every == 0:
                    plot_losses(jsonl_path, plot_path)

                if global_step % args.save_steps == 0:
                    ckpt_dir = save_checkpoint(args.output_dir, model, optimizer, scheduler,
                                                epoch, step_in_epoch, global_step)
                    plot_losses(jsonl_path, plot_path)
                    logger.info(f"Da luu checkpoint local: {ckpt_dir}")
                    if args.push_to_hub:
                        push_to_hub(ckpt_dir, diagnostics_dir, args.hub_model_id,
                                    args.hub_private, readme_text)
                        logger.info(f"Da push checkpoint len hub: {args.hub_model_id}")

            start_step_in_epoch = 0  # tu epoch tiep theo tro di, khong can offset resume nua

        # checkpoint cuoi cung sau khi hoan thanh training
        final_ckpt = save_checkpoint(args.output_dir, model, optimizer, scheduler,
                                      args.num_train_epochs - 1, steps_per_epoch - 1, global_step)
        plot_losses(jsonl_path, plot_path)
        if args.push_to_hub:
            push_to_hub(final_ckpt, diagnostics_dir, args.hub_model_id, args.hub_private, readme_text)
        logger.info("Training hoan tat.")

    except KeyboardInterrupt:
        logger.warning("Nhan KeyboardInterrupt — luu checkpoint khan cap truoc khi thoat ...")
        save_checkpoint(args.output_dir, model, optimizer, scheduler, epoch, step_in_epoch, global_step)
        plot_losses(jsonl_path, plot_path)
        raise
    finally:
        for h in hooks:
            h.remove()


if __name__ == "__main__":
    main()