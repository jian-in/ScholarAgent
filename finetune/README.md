# M5(可选实验):QLoRA 微调领域小模型

> 状态:**脚手架已就绪,尚未实际训练**。
> 训练要下载底座模型(约 3 GB)并占用显卡数小时,请确认后再执行。

## 为什么微调、微调什么

现在系统的所有环节都由一个通用大模型承担。微调实验的假设是:
**"计划拆解"和"反思质检"这类格式固定的轻任务,一个 1.5B 的领域小模型
就能胜任**——推理更快、完全本地、还能作为论文的对比实验:

| 环节 | 微调前(通用 8B) | 微调后(专用 1.5B) |
|------|----------------|-------------------|
| Planner 的计划拆解 | 基线 | 对比组 |
| Planner 的反思质检 | 基线 | 对比组 |

用 M6 的评测框架对比两组的任务完成质量与耗时,就是论文实验章节的
现成素材(结论好坏都有价值:小模型不行也是发现)。

## 硬件预算(RTX 4060 8GB)

- QLoRA 4bit + Qwen2.5-1.5B-Instruct:峰值显存约 5-6 GB ✅
- 3B 模型也可尝试(约 7 GB,需关掉其他占显存的程序)
- 7B 训练不可行(推理用 Ollama 的 int4 量化没问题)

## 步骤(确认后执行;以下命令全部在本项目根目录下运行)

```powershell
# 1. 把 LLaMA-Factory 克隆到项目根目录下并安装(不要 cd 进去)
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
pip install -e "LLaMA-Factory[torch,metrics]"

# 2. 生成训练数据
python finetune/prepare_data.py

# 3. 登记数据集:把数据复制进 LLaMA-Factory 的 data 目录,
#    并在 LLaMA-Factory/data/dataset_info.json 里手工加一条:
#    "scholar_sft": {"file_name": "scholar_sft.json"}
Copy-Item finetune/data/scholar_sft.json LLaMA-Factory/data/

# 4. 开训(yaml 里的 dataset_dir 已指向 LLaMA-Factory/data)
llamafactory-cli train finetune/qlora_qwen_1_5b.yaml

# 5. 训练完成后,把 LoRA 增量合并回底座模型
llamafactory-cli export finetune/qlora_qwen_1_5b_export.yaml
```

### 第 6 步:部署到 Ollama(合并之后还有两小步,如实说明)

导出的是 HuggingFace 格式,Ollama 要吃 GGUF,需要借 llama.cpp 转一次:

```powershell
# 转 GGUF(llama.cpp 仓库里的转换脚本)
python llama.cpp/convert_hf_to_gguf.py finetune/output/merged --outfile finetune/output/scholar-1.5b.gguf
```

然后写一个两行的 Modelfile(`FROM ./finetune/output/scholar-1.5b.gguf`),
执行 `ollama create scholar-1.5b -f Modelfile`,之后在 `.env` 里
`LLM_MODEL=scholar-1.5b` 即可切换到微调后的模型。

## 数据从哪来

`prepare_data.py` 会从三处收集并生成 Alpaca 格式的训练样本:

1. **计划拆解样本**:调研任务 → 标准步骤列表(内置种子样本,可人工扩充)
2. **反思质检样本**:步骤+结果 → {"ok": ...} 判定
3. **系统运行日志**:data/ 下真实的笔记与记忆(作为领域语料参考)

微调数据质量 >> 数量:先人工核对每一条,几百条高质量样本就够起步。
