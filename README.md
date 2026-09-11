# OT-MoE

## Yêu cầu

- Docker Desktop hoặc Docker Engine
- GPU NVIDIA và NVIDIA Container Toolkit nếu chạy fine-tuning
- Hugging Face token có quyền truy cập các model cần thiết

## 1. Tạo Hugging Face token

Tại thư mục gốc của dự án, ví dụ:

```bash
cd /Users/dinhtracducanh/Downloads/OT-MoE
printf '%s\n' 'HF_TOKEN=your_hf_token' > .env
```

Không commit file `.env` lên Git. Đảm bảo `.gitignore` có:

```gitignore
.env
```

## 2. Build Docker image

```bash
docker build -t ot-moe:latest .
```

## 3. Khởi chạy container

### Có GPU NVIDIA

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --env-file .env \
  -v "$PWD:/app" \
  -w /app \
  ot-moe:latest \
  bash
```

### Chỉ chạy CPU

```bash
docker run --rm -it \
  --env-file .env \
  -v "$PWD:/app" \
  -w /app \
  ot-moe:latest \
  bash
```

Các lệnh tiếp theo được chạy bên trong container.

Kiểm tra môi trường:

```bash
python --version
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 4. Download và xử lý dữ liệu

```bash
python download_data.py
python download_mt.py
python process_data.py
```

## 5. Fine-tuning LoRA

Các tham số `--nproc_per_node` và `--batch_size` phụ thuộc vào số lượng và dung lượng GPU.

### Qwen1.5-MoE-A2.7B

Với 4 GPU:

```bash
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  training/finetuning/Qwen1.5-MoE-A2.7B.py \
  --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
  --data_dir /app/data/processed_alignment/ \
  --num_train_epochs 3 \
  --batch_size 64 \
  --resume_from_checkpoint auto \
  --output_dir /app/training/finetuning/checkpoints/Qwen1.5-MoE-A2.7B/ \
  --nccl_timeout_minutes 30 \
  --lora_r_router 4 \
  --lora_r_attention 16 \
  --lora_r_experts 16
```

### Marco-Nano-Instruct

Với 4 GPU:

```bash
torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  training/finetuning/Macro-Nano-Instruct.py \
  --model_name_or_path ATH-MaaS/Marco-Nano-Instruct \
  --data_dir /app/data/processed_alignment/ \
  --num_train_epochs 3 \
  --batch_size 64 \
  --resume_from_checkpoint auto \
  --output_dir /app/training/finetuning/checkpoints/Marco-Nano-Instruct/ \
  --nccl_timeout_minutes 30 \
  --lora_r_router 4 \
  --lora_r_attention 16 \
  --lora_r_experts 16
```

> Các lệnh `torchrun` trong README được chạy trong terminal, không thêm ký tự `!`.

## 6. Dừng và xóa container

Container được tự động xóa sau khi thoát vì sử dụng tùy chọn `--rm`:

```bash
exit
```

Dữ liệu, checkpoint và kết quả vẫn được lưu trong thư mục dự án nhờ tùy chọn:

```bash
-v "$PWD:/app"
```