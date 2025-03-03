# FANformer: Enhancing LLMs through Effective Periodicity Modeling

[![arXiv](https://img.shields.io/badge/arXiv-2502.21309-b31b1b.svg)](https://arxiv.org/abs/2502.21309)

This repository contains the implementation and pre-trained models for FANformer, a novel architecture that enhances Large Language Models through effective periodicity modeling.

## Overview
- **Revised Architecture**: Implemented in `olmo/model.py`
- **Model Scale**: 1B parameters pre-trained model

### Training
Launch distributed training with 8 GPUs:
```bash
torchrun --nproc_per_node=8 scripts/train.py configs/test/FANformer-1B-pretrain.yaml
```

### Evaluation
Run comprehensive evaluation using the OLMo benchmark:
```bash
olmes --model ${MODEL_PATH} --task main_suite::olmo1 --output-dir ${OUTPUT_DIR}
```

### Pre-trained Models
| Model          | Parameters | Download |
|----------------|------------|----------|
| FANformer-1B   | 1.1B       | [Available soon within a week](#) |


## Citation
```bibtex
@article{dong2024fanformer,
  title={FANformer: Improving Large Language Models Through Effective Periodicity Modeling},
  author={Dong, Yihong and Li, Ge and Jiang, Xue and Tao, Yongding and Zhang, Kechi and Zhu, Hao and Liu, Huanyu and Ding, Jiazheng and Li, Jia and Deng, Jinliang and Mei, Hong},
  journal={arXiv preprint arXiv:2502.21309},
  year={2024}
}