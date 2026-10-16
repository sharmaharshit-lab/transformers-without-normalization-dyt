# Transformers without Normalization: Dynamic Tanh (DyT)

A reduced-scale PyTorch implementation inspired by **“Transformers without Normalization”** by Zhu et al. (CVPR 2025).

## Overview

This project investigates Dynamic Tanh (DyT), an alternative to conventional normalization layers in Transformer architectures. The experiments compare LayerNorm and DyT in two small models:

- **Vision Transformer (ViT):** CIFAR-10 image classification.
- **GPT:** Character-level language modeling on Tiny Shakespeare.

The implementation also studies how the initial value of DyT's learnable parameter, alpha, affects performance.

> This is a reduced-scale experimental study, not an exact reproduction of every result in the original paper.

## Method

Dynamic Tanh is defined as

\[
\operatorname{DyT}(x)=\gamma\odot\tanh(\alpha x)+\beta
\]

where alpha is a learnable scalar, while gamma and beta are learnable feature-wise scale and bias parameters. The element-wise product is denoted by \(\odot\).

In this implementation, DyT replaces LayerNorm at the Transformer block normalization locations and the final normalization layer.

## Datasets

### CIFAR-10

- 32 × 32 color images.
- 10 image classes.
- Training split: 45,000 images.
- Validation split: 5,000 images.
- Test split: 10,000 images.

### Tiny Shakespeare

- Approximately 1.12 million characters.
- Character-level vocabulary: 65 characters.
- Training split: 90%.
- Validation split: 10%.

## Model configurations

### Small ViT

- Patch size: 4 × 4.
- Embedding dimension: 128.
- Attention heads: 4.
- Transformer blocks: 2.
- MLP dimension: 256.
- Main comparison: 20 epochs.
- Optimizer: AdamW.
- Learning rate: 0.0003.
- Batch size: 128.

### Small GPT

- Character-level tokenization.
- Context length: 128.
- Embedding dimension: 128.
- Attention heads: 4.
- Transformer blocks: 2.
- MLP dimension: 256.
- Training: 1,000 steps.
- Optimizer: AdamW.
- Learning rate: 0.0003.
- Batch size: 64.

The GPT uses causal self-attention to prevent future characters from being visible when predicting the next character.

## Results

### CIFAR-10 ViT

| Model | Test loss | Test accuracy | Parameters |
|---|---:|---:|---:|
| LayerNorm | 1.0413 | 65.12% | 281,226 |
| DyT, alpha = 0.5 | 1.1312 | 58.81% | 281,231 |

In this experiment, LayerNorm achieved higher test accuracy. The parameter counts were nearly identical.

### Tiny Shakespeare GPT

| Model | Validation loss | Perplexity | Training time |
|---|---:|---:|---:|
| LayerNorm | 2.2218 | 9.22 | 18.06 s |
| DyT, alpha = 0.5 | 2.4111 | 11.15 | 16.79 s |

LayerNorm achieved lower validation loss and perplexity under the tested configuration. DyT took slightly less time, but this short run does not establish a general speed advantage.

### Alpha initialization ablation

For the ViT, the best five-epoch validation accuracy among the tested initial alpha values was 55.86% with alpha = 2.0.

For GPT, the best validation loss among the tested alpha values was 2.3928 with alpha = 1.0. The differences between GPT alpha settings were modest.

See the `results/` directory for complete numerical tables and the `figures/` directory for plots.

## Repository structure

```text
├── README.md
├── requirements.txt
├── notebooks/
├── results/
└── figures/
```

## How to run

1. Open the notebook in Google Colab or a compatible Python environment.
2. Install the dependencies listed in `requirements.txt`.
3. Run the notebook cells in order to download the datasets, define the models, train the variants, and evaluate the results.
4. The experiments can be reduced in duration by lowering the number of epochs or training steps.

A CUDA-capable GPU is recommended for faster training. The reported experiments were conducted on a Google Colab GPU environment.

## Limitations

- The models are deliberately small and use short training schedules.
- Results may differ from the original paper because of architecture, initialization, optimization, and training-duration differences.
- The alpha ablations use limited configurations and should not be interpreted as proving a universally optimal initialization.
- GPT performance is measured using validation loss and perplexity, while the ViT comparison reports test accuracy and loss.

## Reference

Zhu, J., Chen, Y., He, K., LeCun, Y., and Liu, Z., “Transformers without Normalization,” *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025.

- Paper: https://arxiv.org/abs/2503.10622
- Official implementation: https://github.com/jiachenzhu/DyT
