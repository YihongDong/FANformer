# Reasoning is Periodicity? Improving Large Language Models Through Effective Periodicity Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2502.21309-b31b1b.svg)](https://arxiv.org/abs/2502.21309)

This repository contains the implementation and pre-trained models for FANformer, a novel architecture that enhances Large Language Models through effective periodicity modeling.

🎉 Our work has been accepted to NeurIPS'25.

## Overview
- **Revised Architecture**: Implemented in `olmo/model.py`
- **Model Scale**: 1B parameters pre-trained model

### Training
Launch distributed training with 8 GPUs:
```bash
torchrun --nproc_per_node=8 scripts/train.py configs/test/FANformer-1B-pretrain.yaml
```

#### Tips

We are pre-training an LLM over 10B parameters based on FANformer and have achieved promising results. During training, we encountered a few subtle issues related to distributed training frameworks and specific model configurations. We share some tips here:

1) **Explicit `input_layernorm` when using Megatron**. When using Megatron for distributed training, Megatron's Transformer calls a fused `column_parallel + layernorm` CUDA kernel from Transformer Engine, which merges the `input_layernorm` (outside the Attention block) with the QKV matrix multiplication into a single low-level operator. FanLayer does **not** use this fused kernel — it uses `column_parallel` for two separate matrix multiplications — and as a result, `input_layernorm` was silently missing. Since FANformer applies a FanLayer *after* `input_layernorm` (and therefore cannot use the standard fused kernel), `input_layernorm` must be applied **explicitly** before the FAN computation.

2) **Separate FANLayer for V in tall-narrow models**. FANformer was originally trained on OLMo with a standard aspect ratio. When training tall-narrow models (deep but smaller hidden size), you can remove W_v entirely and use a dedicated FANLayer for V, i.e., Q and K still share one FANLayer (unchanged), while V is produced by its own FANLayer with no linear projection. This reduces parameter count. See `FANformerSequentialBlock` in `olmo/model.py` for details. Additionally, you can tune the `p_ratio` of the Q/K FANLayer for further optimization.


### Evaluation
Run comprehensive evaluation using the OLMo benchmark:
```bash
olmes --model ${MODEL_PATH} --task main_suite::olmo1 --output-dir ${OUTPUT_DIR}
```


### Pre-trained Models
| Model          | Non-embedding Parameters | Training Tokens | Download |
|----------------|:------------:|:-----------------:|----------|
| FANformer-1B   | 1.1B       | 1T | [Huggingface](https://huggingface.co/dongyh/FANformer-1B) or [Google Drive](https://drive.google.com/drive/folders/1lwxxPskEwp5tA2CImITOhiaugGFEVJAs?usp=drive_link) |


## Wandb report on comparison

[![WandB](https://img.shields.io/badge/WandB-Report-orange.svg)](https://api.wandb.ai/links/dongyh/vh4jyqsx)

## Citation
```bibtex
@article{dong2025fanformer,
  title={FANformer: Improving Large Language Models Through Effective Periodicity Modeling},
  author={Dong, Yihong and Li, Ge and Jiang, Xue and Tao, Yongding and Zhang, Kechi and Zhu, Hao and Liu, Huanyu and Ding, Jiazheng and Li, Jia and Deng, Jinliang and Mei, Hong},
  journal={arXiv preprint arXiv:2502.21309},
  year={2025}
}

@article{dong2024fan,
  title={FAN: Fourier Analysis Networks},
  author={Yihong Dong and Ge Li and Yongding Tao and Xue Jiang and Kechi Zhang and Jia Li and Jing Su and Jun Zhang and Jingjing Xu},
  journal={arXiv preprint arXiv:2410.02675},
  year={2024}
}
```
