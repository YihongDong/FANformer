# FANformer

We provide the code and pretrained model of FANformer here.

### Running the Code

To run the training process, use the following command:

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/test/FANformer-1B-pretrain.yaml
```

### FANformer-1B

We provide the pretrained model of FANformer-1B at the following anonymous link:

[Anonymous Link of FANformer-1B](https://drive.google.com/drive/folders/1RXR1azoDL06IsScfp7HanWdiV6BOTsjR?usp=sharing)

### Evaluation

```bash
olmes --model ${model_path} --task main_suite::olmo1 --output-dir ${output_dir}
```


