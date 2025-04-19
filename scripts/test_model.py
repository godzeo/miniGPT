from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import argparse
import os
from huggingface_hub import HfFolder

def parse_args():
    parser = argparse.ArgumentParser(description="测试模型生成效果")
    parser.add_argument("--model_path", type=str, default="../outputs/pretrain/final", help="基础模型路径，可以是本地路径或Hugging Face Hub上的模型ID")
    parser.add_argument("--tokenizer_path", type=str, default="gpt2", 
                       help="tokenizer路径，可以是本地路径（如minimind_hf/tokenizer）或Hugging Face模型ID（如gpt2, facebook/opt-125m）")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA模型路径，如果提供则使用LoRA模型")
    parser.add_argument("--merge_lora", action="store_true", help="是否将LoRA权重合并到基础模型中")
    parser.add_argument("--max_length", type=int, default=100, help="生成文本的最大长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.95, help="top-p采样参数")
    parser.add_argument("--top_k", type=int, default=50, help="top-k采样参数")
    parser.add_argument("--prompt", type=str, default=None, help="自定义提示词，如果不提供则使用默认测试提示词")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token，用于访问私有模型")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """加载模型和tokenizer，支持LoRA模型和远程模型"""
    print(f"Loading base model from {args.model_path}...")
    
    # 检查可用设备
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS device")
    else:
        device = "cpu"
        print("Using CPU device")
    
    # 设置Hugging Face token（如果需要）
    if args.hf_token:
        HfFolder.save_token(args.hf_token)
        print("Hugging Face token set successfully")
    
    # 加载tokenizer
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    try:
        # 尝试直接从路径加载（可以是本地路径或模型ID）
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            trust_remote_code=True
        )
        print("Successfully loaded tokenizer")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Please make sure the tokenizer path is correct and accessible")
        print("You can use either:")
        print("  1. Local path (e.g., minimind_hf/tokenizer)")
        print("  2. Hugging Face model ID (e.g., gpt2, facebook/opt-125m)")
        raise
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    print(f"Loading base model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    # 如果提供了LoRA路径，加载LoRA模型
    if args.lora_path:
        print(f"Loading LoRA model from {args.lora_path}...")
        
        if args.merge_lora:
            # 合并LoRA权重到基础模型中
            print("Merging LoRA weights into base model...")
            model = PeftModel.from_pretrained(model, args.lora_path)
            model = model.merge_and_unload()
            print("LoRA weights merged successfully!")
        else:
            # 直接加载LoRA模型
            print("Loading LoRA model without merging...")
            model = PeftModel.from_pretrained(model, args.lora_path)
            print("LoRA model loaded successfully!")
    
    # 将模型移到正确的设备
    model = model.to(device)
    
    return model, tokenizer, device

def generate_text(prompt: str, model, tokenizer, device, args):
    # 对输入进行编码
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 将输入移到正确的设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成文本
    outputs = model.generate(
        **inputs,
        max_length=args.max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # 解码输出
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    args = parse_args()
    
    # 加载模型和tokenizer
    try:
        model, tokenizer, device = load_model_and_tokenizer(args)
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        return
    
    # 准备测试prompt
    if args.prompt:
        test_prompts = [args.prompt]
    else:
        test_prompts = [
            "你知道长江吗？",
            "万有引力是谁提出的？",
            "海水为什么是咸的？",
            "世界上最高的山峰是什么？",
        ]
    
    print("\n=== 开始测试生成 ===")
    print(f"模型路径: {args.model_path}")
    if args.lora_path:
        print(f"LoRA模型路径: {args.lora_path}")
        print(f"LoRA合并模式: {'已合并' if args.merge_lora else '未合并'}")
    print(f"生成参数: max_length={args.max_length}, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    
    for prompt in test_prompts:
        try:
            output = generate_text(prompt, model, tokenizer, device, args)
            print(f"result: {output}")
        except Exception as e:
            print(f"生成过程中出错: {e}")
        print("-" * 50)

if __name__ == "__main__":
    main() 