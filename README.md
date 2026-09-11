# Instruction to run the code after bulding Docker

## 1. Build environment and put HF_TOKEN to .env

```bash
printf '%s\n' 'HF_TOKEN=your_hf_token' > /.env
```

## 2. Download and process data

```bash
python download_data.py
python download_mt.py
python process_data.py
```

## 3 LoRA finetuning baseline

### For Qwen MoE finetuning

```bash
!torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    /training/finetuning/Qwen1.5-MoE-A2.7B.py \
    --model_name_or_path Qwen/Qwen1.5-MoE-A2.7B \
    --data_dir /data/processed_alignment/ \
    --num_train_epochs 3 \
    --batch_size 64 \
    --resume_from_checkpoint auto \
    --output_dir /training/finetuning/checkpoints/Qwen1.5-MoE-A2.7B/ \
    --nccl_timeout_minutes 30 \
    --lora_r_router 4 --lora_r_attention 16 --lora_r_experts 16
```

### For Macro-Nano MoE finetuning

```bash
!torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    /training/finetuning/Macro-Nano-Instruct.py \
    --model_name_or_path ATH-MaaS/Marco-Nano-Instruct \
    --data_dir /data/processed_alignment/ \
    --num_train_epochs 3 \
    --batch_size 64 \
    --resume_from_checkpoint auto \
    --output_dir /training/finetuning/checkpoints/Marco-Nano-Instruct/ \
    --nccl_timeout_minutes 30 \
    --lora_r_router 4 --lora_r_attention 16 --lora_r_experts 16
```

Arguments such as --nproc_per_node, --batch_size depend on GPU infrastructure.