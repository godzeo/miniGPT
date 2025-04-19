# MiniGPT

## 快速测试流程

如果你想快速测试整个训练流程，可以使用以下命令。这些命令使用较小的数据集（1000条数据）和较少的训练轮数（1轮）来验证流程是否正常工作。

### 1. 预训练（学知识）
```bash
python scripts/pretrain.py \
    --output_dir ../outputs/pretrain_test \
    --data_file ../data/pretrain_hq.jsonl \
    --data_limit 1000 \
    --batch_size 2 \
    --num_epochs 1 \
    --fp16 \
    --tokenizer_path ../mini_tokenizer
```

### 2. 监督微调（学对话方式）
```bash
python scripts/train_sft.py \
    --model_path ../outputs/pretrain_test \
    --output_dir ../outputs/sft_test \
    --data_file ../data/mini/sft_mini_512.jsonl \
    --data_limit 1000 \
    --batch_size 4 \
    --num_epochs 1 \
    --fp16
```

### 3. 人类反馈强化学习（RLHF）
```bash
python scripts/train_dpo.py \
    --model_path ../outputs/sft_test \
    --output_dir ../outputs/dpo_test \
    --data_file ../data/dpo.jsonl \
    --data_limit 1000 \
    --batch_size 2 \
    --num_epochs 1 \
    --beta 0.1 \
    --fp16
```

### 4. LoRA 微调（领域适配）
```bash
python scripts/train_lora.py \
    --model_path ../outputs/sft_test \
    --output_dir ../outputs/lora_test \
    --data_file ../data/lora_medical.jsonl \
    --data_limit 1000 \
    --batch_size 4 \
    --num_epochs 1 \
    --lora_r 8 \
    --lora_alpha 32 \
    --fp16
```

### 5. 测试模型
```bash
python scripts/test_model.py \
    --model_path ../outputs/sft_test \
    --tokenizer_path ../mini_tokenizer \
    --lora_path ../outputs/lora_test \
    --merge_lora \
    --max_length 100 \
    --temperature 0.7 \
    --top_p 0.95 \
    --top_k 50
```

参数说明：
- `--model_path`: 基础模型路径
- `--tokenizer_path`: tokenizer路径
- `--lora_path`: LoRA模型路径（可选）
- `--merge_lora`: 是否将LoRA权重合并到基础模型中
- `--max_length`: 生成文本的最大长度
- `--temperature`: 采样温度
- `--top_p`: top-p采样参数
- `--top_k`: top-k采样参数
- `--prompt`: 自定义提示词（可选）

## 完整训练流程

如果你想进行完整的训练，可以省略 `--data_limit` 参数，并调整其他参数如 `--num_epochs`、`--batch_size` 等。

### 参数说明

所有训练脚本都支持以下通用参数：
- `--model_path`: 输入模型路径
- `--output_dir`: 输出目录
- `--data_file`: 训练数据文件路径
- `--batch_size`: 每个设备的训练batch size
- `--gradient_accumulation_steps`: 梯度累积步数
- `--num_epochs`: 训练轮数
- `--learning_rate`: 学习率
- `--save_steps`: 保存步数
- `--logging_steps`: 日志记录步数
- `--fp16`: 是否使用混合精度训练
- `--max_length`: 序列最大长度
- `--data_limit`: 限制训练数据量（用于测试）

特定训练阶段的额外参数：
- LoRA训练：
  - `--lora_r`: LoRA rank
  - `--lora_alpha`: LoRA alpha
  - `--lora_dropout`: LoRA dropout
- DPO训练：
  - `--max_prompt_length`: 提示最大长度
  - `--beta`: DPO beta参数



### 注意事项

1. 测试流程使用较小的数据集和较少的训练轮数，主要用于验证流程是否正常工作
2. 完整训练时，建议根据你的硬件条件调整 batch_size 和其他参数
3. 使用 `--fp16` 可以显著减少显存使用，但可能会影响训练效果
4. 所有路径参数都支持相对路径和绝对路径
5. 训练流程必须按照顺序进行：预训练 -> SFT -> RLHF -> LoRA

## 项目结构

```
minigpt/
├── configs/           # 配置文件目录
├── data/             # 数据集目录
│   ├── mini/         # 基础数据集
│   └── lora/         # LoRA微调数据集
├── scripts/          # 训练脚本
│   ├── pretrain.py   # 预训练脚本
│   ├── train_sft.py  # SFT训练脚本
│   ├── train_dpo.py  # DPO训练脚本
│   ├── train_lora.py # LoRA训练脚本
│   └── train_tokenizer.py # Tokenizer训练脚本
└── requirements.txt   # 项目依赖
```

## 环境准备

1. 创建并激活虚拟环境（可选）
```bash
# 1. 创建虚拟环境
python3 -m venv minigpt_env

# 2. 激活虚拟环境
source minigpt_env/bin/activate  # 在Mac/Linux上
# 或
minigpt_env\Scripts\activate  # 在Windows上

# 3. 在虚拟环境中安装依赖
pip install -r requirements.txt
```

## 数据准备

1. 从ModelScope或HuggingFace下载数据集：
   - ModelScope: https://www.modelscope.cn/datasets/gongjy/mini_dataset/files
   - HuggingFace: https://huggingface.co/datasets/jingyaogong/mini_dataset/tree/main

2. 将下载的数据文件放入`data/`目录：
   - pretrain_hq.jsonl     # 预训练数据
   - sft_mini_512.jsonl    # SFT数据
   - dpo.jsonl            # DPO数据
   - lora_medical.jsonl   # 领域LoRA数据（可选）


## 分布式训练

使用accelerate进行分布式训练：

1. 首次配置
```bash
accelerate config
```

2. 启动训练
```bash
accelerate launch scripts/train_xxx.py
```

## 注意事项

1. 训练参数（batch_size、learning_rate等）需要根据实际硬件条件调整
2. 代码中使用了半精度训练(fp16)以节省显存，如果显存充足可以使用全精度
3. 可以根据需要更换基座模型
4. 建议使用wandb等工具监控训练过程

## 参考

- HuggingFace文档：[Transformers](https://huggingface.co/docs/transformers/index)
- PEFT文档：[PEFT](https://huggingface.co/docs/peft/index) 