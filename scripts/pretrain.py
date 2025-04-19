from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    GPT2Config
)
from datasets import load_dataset
import torch
from pathlib import Path
import os
import argparse
from huggingface_hub import HfFolder

def create_small_config(vocab_size: int, tokenizer = None) -> GPT2Config:
    """创建一个中等规模的模型配置
    
    Args:
        vocab_size: tokenizer的词表大小，必须与tokenizer的vocab_size完全匹配
        tokenizer: 用于获取特殊token ID的tokenizer对象
    """
    print(f"Creating model config with vocab_size: {vocab_size}")
    
    # 设置默认的特殊token ID
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    
    # 如果提供了tokenizer，使用tokenizer的特殊token ID
    if tokenizer is not None:
        if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None:
            bos_token_id = tokenizer.bos_token_id
        if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
            eos_token_id = tokenizer.eos_token_id
        if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
            pad_token_id = tokenizer.pad_token_id
    
    return GPT2Config(
        vocab_size=vocab_size,          # 词表大小，使用tokenizer的实际大小
        n_positions=512,                # 增加到512
        n_ctx=512,                      # 增加上下文窗口
        n_embd=768,                     # 增加嵌入维度到768
        n_layer=12,                     # 增加到12层
        n_head=12,                      # 增加注意力头数到12
        n_inner=3072,                   # 增加FFN维度
        activation_function='gelu_new',  
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )

def parse_args():
    parser = argparse.ArgumentParser(description="预训练语言模型")
    parser.add_argument("--output_dir", type=str, default="../outputs/pretrain_full", help="模型输出目录")
    parser.add_argument("--tokenizer_path", type=str, default="gpt2", 
                       help="tokenizer路径，可以是本地路径（如minimind_hf/tokenizer）或Hugging Face模型ID（如gpt2）")
    parser.add_argument("--data_file", type=str, default="data/minimind/pretrain_hq.jsonl", help="训练数据文件路径")
    parser.add_argument("--batch_size", type=int, default=2, help="每个设备的训练batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存的checkpoint数量限制")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志记录步数")
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--max_length", type=int, default=512, help="序列最大长度")
    parser.add_argument("--data_limit", type=int, default=None, help="限制训练数据量（用于测试）")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token，用于访问私有模型")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查是否在分布式环境中运行
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank != -1:  # 分布式训练模式
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        print(f"Initialized distributed training. Local rank: {local_rank}, World size: {world_size}")
        ddp_kwargs = {
            "local_rank": local_rank,
            "ddp_backend": "nccl",
            "ddp_find_unused_parameters": False
        }
    else:  # 单机单卡模式
        print("Running in single GPU mode")
        ddp_kwargs = {}

    # 检查可用设备
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        # CUDA 设备支持 fp16
        use_fp16 = args.fp16
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS device")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        # MPS 设备不支持 fp16，强制使用 fp32
        use_fp16 = False
        if args.fp16:
            print("Warning: MPS device does not support fp16, using fp32 instead")
    else:
        device = "cpu"
        print("Using CPU device")
        # CPU 设备不支持 fp16，强制使用 fp32
        use_fp16 = False
        if args.fp16:
            print("Warning: CPU device does not support fp16, using fp32 instead")

    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置Hugging Face token（如果需要）
    if args.hf_token:
        HfFolder.save_token(args.hf_token)
        print("Hugging Face token set successfully")

    # 1. 加载tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    try:
        # 尝试直接从路径加载（可以是本地路径或模型ID）
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        print(f"Successfully loaded tokenizer with vocab size: {len(tokenizer)}")
        
        # 确保tokenizer有必要的特殊token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.eos_token = '</s>'
                
        if tokenizer.bos_token is None:
            tokenizer.bos_token = '<s>'
            
        print(f"Special tokens: BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please make sure the tokenizer path is correct and accessible")
        print("You can use either:")
        print("  1. Local path (e.g., minimind_hf/tokenizer)")
        print("  2. Hugging Face model ID (e.g., gpt2, facebook/opt-125m)")
        raise

    # 2. 创建或加载模型
    checkpoint_path = os.path.join(args.output_dir, "checkpoint-latest")
    
    # 获取实际的词表大小
    vocab_size = len(tokenizer)
    print(f"Creating model with vocab_size: {vocab_size}")
    
    config = create_small_config(vocab_size=vocab_size, tokenizer=tokenizer)
    
    # 检查checkpoint是否完整
    has_valid_checkpoint = (
        os.path.exists(os.path.join(checkpoint_path, "config.json")) and
        os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin")) and
        os.path.exists(os.path.join(checkpoint_path, "trainer_state.json"))
    )
    
    if has_valid_checkpoint:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    else:
        print("Starting training from scratch")
        if os.path.exists(checkpoint_path):
            print("Found incomplete checkpoint, removing it...")
            import shutil
            shutil.rmtree(checkpoint_path, ignore_errors=True)
        model = AutoModelForCausalLM.from_config(config)
    
    # 3. 准备预训练数据集
    dataset = load_dataset("json", data_files=args.data_file)
    dataset = dataset["train"]
    
    if args.data_limit is not None:
        print(f"Using limited dataset with {args.data_limit} examples")
        dataset = dataset.select(range(args.data_limit))
    else:
        print(f"Using full dataset with {len(dataset)} examples")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
    
    # 使用多进程加速数据处理
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
        desc="Tokenizing datasets",
    )

    # 4. 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        save_strategy="epoch",
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        fp16=use_fp16,
        report_to="none",
        **ddp_kwargs  # 动态添加分布式训练参数
    )

    # 5. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 6. 开始训练
    try:
        trainer.train(resume_from_checkpoint=has_valid_checkpoint and checkpoint_path)
        
        # 7. 保存最终模型
        trainer.save_model(os.path.join(args.output_dir, "final"))
        print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        trainer.save_model(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    except Exception as e:
        print(f"\nTraining interrupted by error: {e}")
        trainer.save_model(checkpoint_path)
        print(f"Emergency checkpoint saved to {checkpoint_path}")
        raise

if __name__ == "__main__":
    main() 