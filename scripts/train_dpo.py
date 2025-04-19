from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from trl.trainer.utils import DPODataCollatorWithPadding
from datasets import load_dataset
import torch
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="DPO训练脚本")
    parser.add_argument("--model_path", type=str, default="../outputs/sft", help="基础模型路径，可以是本地路径或Hugging Face Hub上的模型ID")
    parser.add_argument("--tokenizer_path", type=str, default="gpt2", 
                       help="tokenizer路径，可以是本地路径（如minimind_hf/tokenizer）或Hugging Face模型ID（如gpt2, facebook/opt-125m）")
    parser.add_argument("--output_dir", type=str, default="../outputs/dpo", help="模型输出目录")
    parser.add_argument("--data_file", type=str, default="../data/dpo.jsonl", help="训练数据文件路径")
    parser.add_argument("--batch_size", type=int, default=2, help="每个设备的训练batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--num_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="学习率")
    parser.add_argument("--save_steps", type=int, default=100, help="保存步数")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数")
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--max_length", type=int, default=512, help="序列最大长度")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="提示最大长度")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta参数")
    parser.add_argument("--data_limit", type=int, default=None, help="限制训练数据量，用于测试")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查可用设备
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS device")
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    else:
        device = "cpu"
        print("Using CPU device")

    # 1. 加载SFT模型作为初始模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
    )

    # 加载tokenizer
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
        print("  1. Local path (e.g., minimind_hf/tokenizer)")
        print("  2. Hugging Face model ID (e.g., gpt2, facebook/opt-125m)")
        raise

    # 2. 加载DPO数据集
    dataset = load_dataset("json", data_files=args.data_file)
    if args.data_limit:
        dataset["train"] = dataset["train"].select(range(min(args.data_limit, len(dataset["train"]))))
        print(f"Using limited dataset with {len(dataset['train'])} examples")
    else:
        print(f"Using full dataset with {len(dataset['train'])} examples")
    
    # 预处理数据集
    def preprocess_function(examples):
        # 使用tokenizer处理数据
        chosen_prompt = tokenizer.apply_chat_template(
            examples["chosen"], tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = tokenizer.apply_chat_template(
            examples["rejected"], tokenize=False, add_generation_prompt=False
        )
        
        # 构建新的数据格式
        return {
            "prompt": chosen_prompt,
            "chosen": chosen_prompt,
            "rejected": rejected_prompt
        }
    
    # 应用预处理
    dataset["train"] = dataset["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # 3. 设置训练参数
    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        fp16=args.fp16,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        report_to=[],  # 禁用所有报告工具，包括 wandb
    )

    # 4. 创建DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=DPODataCollatorWithPadding(),
        processing_class=tokenizer,
    )

    # 5. 开始训练
    dpo_trainer.train()

if __name__ == "__main__":
    main() 