from huggingface_hub import snapshot_download
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="下载模型到本地")
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face Hub上的模型ID")
    parser.add_argument("--output_dir", type=str, required=True, help="本地保存目录")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face API token")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"开始下载模型 {args.model_id} 到 {args.output_dir}...")
    
    # 下载模型
    snapshot_download(
        repo_id=args.model_id,
        local_dir=args.output_dir,
        token=args.hf_token,
        local_dir_use_symlinks=False
    )
    
    print(f"模型下载完成！保存在: {args.output_dir}")

if __name__ == "__main__":
    main() 