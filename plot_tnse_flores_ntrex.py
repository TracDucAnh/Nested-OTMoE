"""
plot_tnse_flores_ntrex.py

Đọc dữ liệu FLORES + NTREX (đã chuẩn hoá trong data/processed_alignment),
gộp thành một tập observation data duy nhất, chỉ giữ lại top-K ngôn ngữ phổ
biến nhất (bỏ các trường ngôn ngữ còn lại trong từng mẫu, không bỏ cả mẫu),
encode các câu bằng Qwen/Qwen1.5-MoE-A2.7B theo batch, trích xuất hidden state
ở các layer chỉ định, rồi vẽ t-SNE (mỗi layer 1 subplot trên cùng 1 canvas),
tô màu theo từng cụm ngôn ngữ.

Cách chạy:
    python plot_tnse_flores_ntrex.py --top_k 10 --layers 0 12 23
    python plot_tnse_flores_ntrex.py --top_k none --layers 0 6 12 18 23
    python plot_tnse_flores_ntrex.py --top_k 10 --batch_size 32 --device cuda

Yêu cầu thư viện: torch, transformers, scikit-learn, matplotlib, numpy, tqdm
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import orjson as _json_backend

    def _load_json(path: Path):
        with open(path, "rb") as f:
            return _json_backend.loads(f.read())
except ImportError:
    import json as _json_backend

    def _load_json(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return _json_backend.load(f)


from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------------------------------
# Cấu hình mặc định
# ----------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data" / "processed_alignment"
DATASETS = ["flores", "ntrex"]

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEFAULT_LAYERS = [0, 12, 23]
DEFAULT_TOP_K = 10

IGNORE_KEYS = {"id"}
SOURCE_KEY = "__source__"  # key nội bộ, không trùng với mã ngôn ngữ nào


# ----------------------------------------------------------------------------
# Đọc & gộp dữ liệu
# ----------------------------------------------------------------------------
def load_dataset(name: str) -> list[dict]:
    path = DATA_DIR / f"{name}.json"
    data = _load_json(path)
    for record in data:
        record[SOURCE_KEY] = name
    return data


def load_observations() -> list[dict]:
    """Gộp flores + ntrex thành một observation data duy nhất."""
    observations: list[dict] = []
    for name in DATASETS:
        observations.extend(load_dataset(name))
    return observations


def build_lang_index(record: dict) -> dict[str, str]:
    """
    Map mã ngôn ngữ 3 ký tự (ISO 639-3, lấy từ prefix trước dấu '_') -> key gốc
    trong record. Ví dụ: 'jpn' -> 'jpn_Hira' hoặc 'jpn_Jpan' tuỳ dataset.
    Nếu 1 record có nhiều key trùng mã 3 ký tự, lấy key gặp đầu tiên.
    """
    index: dict[str, str] = {}
    for key in record:
        if key in IGNORE_KEYS or key == SOURCE_KEY:
            continue
        prefix = key.split("_")[0][:3]
        if prefix not in index:
            index[prefix] = key
    return index


def has_valid_text(record: dict, key: Optional[str]) -> bool:
    if key is None:
        return False
    text = record.get(key)
    return isinstance(text, str) and text.strip() != ""


# ----------------------------------------------------------------------------
# Chọn top-K ngôn ngữ & lọc mẫu
# ----------------------------------------------------------------------------
def select_top_languages(observations: list[dict], top_k: Optional[int]) -> list[str]:
    """Đếm số mẫu chứa mỗi ngôn ngữ (mã 3 ký tự) trên toàn bộ observation data."""
    freq: Counter = Counter()
    for record in tqdm(observations, desc="Đếm tần suất ngôn ngữ"):
        lang_index = build_lang_index(record)
        for code, key in lang_index.items():
            if has_valid_text(record, key):
                freq[code] += 1

    ranked = [code for code, _ in freq.most_common()]
    if top_k is None:
        return ranked
    return ranked[:top_k]


def resolve_languages(
    observations: list[dict], top_k: Optional[int], languages: Optional[list[str]]
) -> list[str]:
    """
    Nếu --languages được chỉ định: dùng đúng danh sách đó (chuẩn hoá về chữ thường,
    3 ký tự), cảnh báo mã nào không tồn tại trong dữ liệu.
    Ngược lại: chọn theo top_k như cũ.
    """
    if languages is not None:
        normalized = [code.strip().lower()[:3] for code in languages if code.strip()]
        available = set()
        for record in observations:
            available.update(build_lang_index(record).keys())

        missing = [code for code in normalized if code not in available]
        if missing:
            print(
                f"[Cảnh báo] Các mã ngôn ngữ sau không tìm thấy trong dữ liệu và sẽ bị bỏ qua: {missing}"
            )
        resolved = [code for code in normalized if code in available]
        if not resolved:
            raise ValueError(
                "Không có mã ngôn ngữ hợp lệ nào trong --languages khớp với dữ liệu."
            )
        return resolved

    return select_top_languages(observations, top_k)


def filter_samples_to_top_languages(
    observations: list[dict], top_langs: list[str]
) -> list[dict]:
    """
    Với mỗi mẫu, chỉ giữ các trường ngôn ngữ nằm trong top_langs, bỏ các ngôn ngữ
    còn lại của MẪU ĐÓ (không bỏ cả mẫu). Mẫu không còn ngôn ngữ nào thuộc
    top_langs (hiếm khi xảy ra) sẽ bị loại hoàn toàn.
    """
    top_set = set(top_langs)
    filtered: list[dict] = []
    for record in tqdm(observations, desc="Lọc mẫu theo top ngôn ngữ"):
        lang_index = build_lang_index(record)
        kept = {
            code: record[key]
            for code, key in lang_index.items()
            if code in top_set and has_valid_text(record, key)
        }
        if not kept:
            continue
        filtered.append(
            {
                "id": record.get("id"),
                "source": record.get(SOURCE_KEY),
                "langs": kept,  # {lang_code: text}
            }
        )
    return filtered


def flatten_texts(filtered: list[dict]) -> tuple[list[str], list[str], list[str]]:
    """Trải phẳng thành 3 list song song: texts, lang_labels, sample_ids."""
    texts: list[str] = []
    lang_labels: list[str] = []
    sample_ids: list[str] = []
    for sample in filtered:
        for lang_code, text in sample["langs"].items():
            texts.append(text)
            lang_labels.append(lang_code)
            sample_ids.append(sample["id"])
    return texts, lang_labels, sample_ids


# ----------------------------------------------------------------------------
# Encode bằng Qwen1.5-MoE-A2.7B
# ----------------------------------------------------------------------------
@torch.no_grad()
def encode_texts(
    texts: list[str],
    layers: list[int],
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    max_length: int,
) -> dict[int, np.ndarray]:
    """
    Encode toàn bộ texts theo batch bằng Qwen1.5-MoE-A2.7B, trả về
    dict {layer_idx: embedding_matrix [N, hidden_size]} (mean-pooling theo
    attention_mask trên mỗi layer được chỉ định).
    """
    print(f"Đang tải tokenizer & model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModel.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    num_hidden_layers = model.config.num_hidden_layers
    for layer in layers:
        if layer < 0 or layer > num_hidden_layers:
            raise ValueError(
                f"Layer {layer} không hợp lệ. Model có {num_hidden_layers} layer "
                f"(hidden_states index hợp lệ: 0..{num_hidden_layers}, "
                f"0 = embedding output)."
            )

    layer_embeddings: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_texts = texts[start : start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple, len = num_hidden_layers + 1

        mask = inputs["attention_mask"].unsqueeze(-1).to(dtype)  # [B, T, 1]
        valid_counts = mask.sum(dim=1).clamp(min=1)  # [B, 1]

        for layer in layers:
            hs = hidden_states[layer].to(dtype)  # [B, T, H]
            pooled = (hs * mask).sum(dim=1) / valid_counts  # mean pooling
            layer_embeddings[layer].append(pooled.float().cpu().numpy())

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        layer: np.concatenate(chunks, axis=0) for layer, chunks in layer_embeddings.items()
    }


# ----------------------------------------------------------------------------
# t-SNE & vẽ plot
# ----------------------------------------------------------------------------
def run_tsne(embeddings: np.ndarray, seed: int) -> np.ndarray:
    n_samples = embeddings.shape[0]
    perplexity = min(30, max(5, n_samples // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=seed,
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def plot_layers(
    layer_embeddings: dict[int, np.ndarray],
    lang_labels: list[str],
    layers: list[int],
    output_path: Path,
    seed: int,
):
    unique_langs = sorted(set(lang_labels))
    cmap = plt.get_cmap("tab20", max(len(unique_langs), 1))
    color_map = {lang: cmap(i) for i, lang in enumerate(unique_langs)}
    lang_labels_arr = np.array(lang_labels)

    fig, axes = plt.subplots(1, len(layers), figsize=(6 * len(layers), 6), squeeze=False)
    axes = axes[0]

    for ax, layer in tqdm(list(zip(axes, layers)), desc="Chạy t-SNE theo layer"):
        coords = run_tsne(layer_embeddings[layer], seed=seed)
        for lang in unique_langs:
            idx = np.where(lang_labels_arr == lang)[0]
            ax.scatter(
                coords[idx, 0],
                coords[idx, 1],
                s=12,
                alpha=0.7,
                color=color_map[lang],
                label=lang,
            )
        ax.set_title(f"Layer {layer}")
        ax.set_xticks([])
        ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(unique_langs), 10) or 1,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        "t-SNE hidden states — Qwen1.5-MoE-A2.7B — FLORES + NTREX", fontsize=14
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Đã lưu hình tại: {output_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="t-SNE hidden states của Qwen1.5-MoE-A2.7B trên FLORES + NTREX"
    )
    parser.add_argument(
        "--top_k",
        type=str,
        default=str(DEFAULT_TOP_K),
        help=(
            "Số ngôn ngữ phổ biến nhất cần giữ lại. Truyền 'none' để lấy tất cả. "
            "Bị bỏ qua nếu --languages được chỉ định."
        ),
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Danh sách mã ngôn ngữ 3 ký tự cần lấy (ví dụ: eng vie jpn zho). "
            "Nếu được chỉ định, sẽ dùng đúng danh sách này thay vì chọn theo top_k."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=DEFAULT_LAYERS,
        help="Danh sách layer index cần trích hidden state (0 = embedding output).",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128, help="Max token length khi tokenize.")
    parser.add_argument(
        "--max_samples_per_lang",
        type=int,
        default=None,
        help="Giới hạn số câu tối đa mỗi ngôn ngữ để encode (None = không giới hạn).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "analysis" / "tsne_flores_ntrex.png"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    top_k = None if args.top_k.strip().lower() == "none" else int(args.top_k)
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("Đang đọc dữ liệu FLORES + NTREX ...")
    observations = load_observations()
    print(f"Tổng số mẫu gộp: {len(observations)}")

    top_langs = resolve_languages(observations, top_k, args.languages)
    if args.languages is not None:
        print(f"Ngôn ngữ được chỉ định thủ công ({len(top_langs)}): {top_langs}")
    else:
        print(f"Top ngôn ngữ được chọn ({len(top_langs)}): {top_langs}")

    filtered = filter_samples_to_top_languages(observations, top_langs)
    print(f"Số mẫu còn lại sau khi lọc: {len(filtered)}")

    texts, lang_labels, sample_ids = flatten_texts(filtered)

    if args.max_samples_per_lang is not None:
        per_lang_count: dict = defaultdict(int)
        keep_idx = []
        for i, lang in tqdm(list(enumerate(lang_labels)), desc="Giới hạn số câu/ngôn ngữ"):
            if per_lang_count[lang] < args.max_samples_per_lang:
                keep_idx.append(i)
                per_lang_count[lang] += 1
        texts = [texts[i] for i in keep_idx]
        lang_labels = [lang_labels[i] for i in keep_idx]
        sample_ids = [sample_ids[i] for i in keep_idx]

    print(f"Tổng số câu sẽ encode: {len(texts)}")
    for lang in sorted(set(lang_labels)):
        print(f"  {lang}: {lang_labels.count(lang)} câu")

    layer_embeddings = encode_texts(
        texts=texts,
        layers=args.layers,
        batch_size=args.batch_size,
        device=args.device,
        dtype=dtype,
        max_length=args.max_length,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_layers(layer_embeddings, lang_labels, args.layers, output_path, seed=args.seed)


if __name__ == "__main__":
    main()