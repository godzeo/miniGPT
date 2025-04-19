import os
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb
import re
from huggingface_hub import HfFolder

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA训练脚本")
    parser.add_argument("--model_path", type=str, default="../outputs/sft", help="基础模型路径，可以是本地路径或Hugging Face Hub上的模型ID")
    parser.add_argument("--tokenizer_path", type=str, default="gpt2", 
                       help="tokenizer路径，可以是本地路径（如mini_hf/tokenizer）或Hugging Face模型ID（如gpt2, facebook/opt-125m）")
    parser.add_argument("--output_dir", type=str, default="../outputs/lora", help="模型输出目录")
    parser.add_argument("--data_file", type=str, default="../data/lora_medical.jsonl", help="训练数据文件路径")
    parser.add_argument("--batch_size", type=int, default=4, help="每个设备的训练batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--max_length", type=int, default=512, help="序列最大长度")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--data_limit", type=int, default=None, help="限制训练数据量，用于测试")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="mini-lora", help="wandb项目名称")
    parser.add_argument("--target_modules", type=str, default=None, help="LoRA目标模块，多个模块用逗号分隔，如'q_proj,v_proj'")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token，用于访问私有模型")
    return parser.parse_args()

def find_target_modules(model):
    """查找模型中适合作为LoRA目标模块的模块名称"""
    target_modules = []
    
    # 常见的注意力模块名称模式
    patterns = [
        r"q_proj", r"k_proj", r"v_proj", r"o_proj",  # 标准注意力模块
        r"query", r"key", r"value", r"output",       # 另一种命名方式
        r"W_q", r"W_k", r"W_v", r"W_o",             # 另一种命名方式
        r"c_attn", r"c_proj",                        # GPT-2风格
        r"attention\.query", r"attention\.key", r"attention\.value", r"attention\.output",  # 嵌套结构
    ]
    
    # 遍历模型的所有命名模块
    for name, module in model.named_modules():
        # 检查模块是否是线性层（nn.Linear）
        if isinstance(module, torch.nn.Linear):
            # 检查模块名称是否匹配任何模式
            for pattern in patterns:
                if re.search(pattern, name):
                    target_modules.append(name)
                    break
    
    # 如果没有找到匹配的模块，尝试使用更通用的方法
    if not target_modules:
        # 查找所有线性层
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                target_modules.append(name)
    
    # 如果仍然没有找到，使用默认值
    if not target_modules:
        target_modules = ["q_proj", "v_proj"]
    
    return target_modules

def tokenize_function(examples, tokenizer, args):
    # 处理对话格式
    if "conversations" in examples:
        # 将对话转换为文本
        texts = []
        for conv in examples["conversations"]:
            text = ""
            for msg in conv:
                if msg["role"] == "user":
                    text += f"Human: {msg['content']}\n"
                else:
                    text += f"Assistant: {msg['content']}\n"
            texts.append(text)
        return tokenizer(
            texts,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
    # 处理单文本格式
    elif "text" in examples:
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
    elif "input" in examples:
        return tokenizer(
            examples["input"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
    elif "prompt" in examples:
        return tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
    else:
        raise ValueError("Dataset must contain 'conversations', 'text', 'input', or 'prompt' field")

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

    # 初始化wandb
    if args.use_wandb and local_rank <= 0:  # 只在主进程上初始化wandb
        wandb.init(project=args.wandb_project, name=f"lora-{os.path.basename(args.output_dir)}")

    # 1. 加载模型和tokenizer
    print("Loading model and tokenizer...")
    print(f"Loading base model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
    )
    
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    try:
        # 尝试直接从路径加载（可以是本地路径或模型ID）
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            trust_remote_code=True,
        )
        print(f"Successfully loaded tokenizer with vocab size: {len(tokenizer)}")
        
        # 确保tokenizer有必要的特殊token
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Using EOS token as PAD token")
            else:
                tokenizer.pad_token = tokenizer.eos_token = '</s>'
                print("Setting both EOS and PAD tokens to '</s>'")
        
        if tokenizer.bos_token is None:
            tokenizer.bos_token = '<s>'
            print("Setting BOS token to '<s>'")
            
        # 确保模型配置与tokenizer一致
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        
        print(f"Special tokens configuration:")
        print(f"- PAD token: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
        print(f"- BOS token: '{tokenizer.bos_token}' (id: {tokenizer.bos_token_id})")
        print(f"- EOS token: '{tokenizer.eos_token}' (id: {tokenizer.eos_token_id})")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please make sure the tokenizer path is correct and accessible")
        print("You can use either:")
        print("  1. Local path (e.g., mini_hf/tokenizer)")
        print("  2. Hugging Face model ID (e.g., gpt2, facebook/opt-125m)")
        raise
    
    # 2. 确定LoRA目标模块
    if args.target_modules:
        # 如果用户指定了目标模块，使用用户指定的
        target_modules = args.target_modules.split(',')
        print(f"Using user-specified target modules: {target_modules}")
    else:
        # 否则自动检测
        target_modules = find_target_modules(model)
        print(f"Automatically detected target modules: {target_modules}")
    
    # 3. 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules
    )
    
    # 4. 将模型转换为PEFT模型
    model = get_peft_model(model, lora_config)
    
    # 将模型移到正确的设备
    model = model.to(device)

    # 5. 准备数据集
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=args.data_file)

    if args.data_limit is not None:
        print(f"Using limited dataset with {args.data_limit} examples")
        dataset["train"] = dataset["train"].select(range(min(args.data_limit, len(dataset["train"]))))
    else:
        print(f"Using full dataset with {len(dataset['train'])} examples")

    # 使用多进程加速数据处理
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args),
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=os.cpu_count(),
        desc="Tokenizing datasets",
    )

    # 6. 设置训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=use_fp16,
        report_to="wandb" if args.use_wandb else "none",
        save_total_limit=3,  # 只保留最新的3个checkpoint
        push_to_hub=False,
        **ddp_kwargs  # 添加分布式训练参数
    )

    # 7. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # 8. 开始训练
    try:
        trainer.train()
        
        # 9. 保存最终模型
        if local_rank <= 0:  # 只在主进程上保存模型
            trainer.save_model(os.path.join(args.output_dir, "final"))
            print("Training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        if local_rank <= 0:  # 只在主进程上保存checkpoint
            trainer.save_model(os.path.join(args.output_dir, "checkpoint-latest"))
            print(f"Model saved to {os.path.join(args.output_dir, 'checkpoint-latest')}")
    except Exception as e:
        print(f"\nTraining interrupted by error: {e}")
        if local_rank <= 0:  # 只在主进程上保存checkpoint
            trainer.save_model(os.path.join(args.output_dir, "checkpoint-latest"))
            print(f"Emergency checkpoint saved to {os.path.join(args.output_dir, 'checkpoint-latest')}")
        raise
    finally:
        if args.use_wandb and local_rank <= 0:  # 只在主进程上结束wandb
            wandb.finish()

if __name__ == "__main__":
    main() 