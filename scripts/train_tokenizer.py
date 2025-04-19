from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
from datasets import load_dataset
from pathlib import Path

def train_tokenizer(
    data_files: str = "/data/pretrain_mini.jsonl",
    vocab_size: int = 6400,
    min_frequency: int = 2,
    save_dir: str = "../outputs/tokenizer",
):
    """训练一个新的tokenizer"""
    # 1. 加载数据集
    dataset = load_dataset("json", data_files=data_files)
    
    # 2. 创建tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # 3. 设置预分词器
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    # 4. 设置训练参数
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"],
    )
    
    # 5. 准备训练数据
    def get_training_corpus():
        for i in range(0, len(dataset["train"])):
            yield dataset["train"][i]["text"]
    
    # 6. 训练tokenizer
    tokenizer.train_from_iterator(get_training_corpus(), trainer)
    
    # 7. 设置后处理
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.decoder = decoders.ByteLevel()
    
    # 8. 转换为Transformers格式
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2", tokenizer_object=tokenizer)
    
    # 9. 设置特殊token
    hf_tokenizer.pad_token = "<pad>"
    hf_tokenizer.eos_token = "</s>"
    hf_tokenizer.bos_token = "<s>"
    hf_tokenizer.unk_token = "<unk>"
    
    # 10. 保存tokenizer
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    hf_tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to {save_dir}")
    
    return hf_tokenizer

if __name__ == "__main__":
    train_tokenizer() 