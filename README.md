# Transformers without Normalization: Dynamic Tanh (DyT)

A reduced-scale PyTorch implementation and experimental study of **Dynamic Tanh (DyT)** from the paper *“Transformers without Normalization”* by Zhu et al. (CVPR 2025).

This project was completed as an individual **Deep Learning Paper Implementation Project**. The main objective was to understand the core idea of Dynamic Tanh, implement it directly in PyTorch, replace LayerNorm in small Transformer models, and study the effect of the learnable parameter initialization through ablation experiments.

---

## 1. Project Overview

Layer Normalization (LayerNorm) is widely used in Transformer architectures. The paper *Transformers without Normalization* explores whether Transformers can work without conventional normalization layers by replacing them with a simple learnable nonlinear transformation called **Dynamic Tanh (DyT)**.

For this project, I implemented and compared two normalization approaches in two small Transformer-based models:

- **Small Vision Transformer (ViT)** on CIFAR-10.
- **Small GPT** on Tiny Shakespeare.

In addition to the main LayerNorm vs. DyT comparison, I performed an **alpha initialization ablation** to investigate how the initial value of DyT's learnable parameter affects training.

The experiments were intentionally kept small so that they could run on a free Google Colab Tesla T4 GPU.

> **Important:** This is a reduced-scale implementation of the main idea of the paper. It is not a full reproduction of all experiments and model sizes reported in the original paper.

---

## 2. Paper

**Zhu, J., Chen, Y., He, K., LeCun, Y., and Liu, Z.**

*Transformers without Normalization.*

IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

- Paper: https://arxiv.org/abs/2503.10622
- Official implementation: https://github.com/jiachenzhu/DyT

The paper was used as the basis for understanding and implementing the Dynamic Tanh idea.

---

## 3. What I Implemented

The main idea implemented in this project is **Dynamic Tanh (DyT)**.

Instead of applying LayerNorm, DyT applies the following transformation:

\[\operatorname{DyT}(x)=\gamma \odot \tanh(\alpha x) + \beta\]

where:

- \(x\) is the input tensor.
- \(\alpha\) is a **learnable scalar parameter**.
- \(\gamma\) is a learnable feature-wise scale parameter.
- \(\beta\) is a learnable feature-wise bias parameter.
- \(\odot\) denotes element-wise multiplication.

The important difference from a fixed activation function is that **alpha is learned during training**.

I implemented DyT directly using PyTorch and first tested the layer independently before using it inside the Transformer models.

The initial implementation check verified that:

- input and output shapes were preserved,
- alpha received gradients,
- gamma received gradients,
- beta received gradients.

Example from the implementation test:

```text
Input shape: torch.Size([2, 4, 8])
Output shape: torch.Size([2, 4, 8])
Initial alpha: 0.5
Parameters: 17
Alpha gradient: 0.5980923771858215
Gamma gradient exists: True
Beta gradient exists: True
```

---

# 4. Experimental Environment

The experiments were developed and run in Google Colab.

### Hardware

```text
GPU: NVIDIA Tesla T4
GPU memory: approximately 14.56 GB
CUDA available: True
```

### Software

```text
Python: 3.13.15
PyTorch: 2.11.0+cu130
Platform: Linux x86_64
Random seed: 42
```

The experiments use fixed random seeds where appropriate so that the LayerNorm and DyT models can be compared under similar initialization and data-order conditions.

---

# 5. CIFAR-10 Vision Transformer

## Dataset

The first experiment uses CIFAR-10 for image classification.

The dataset contains 10 classes:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

I used:

| Split | Images |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Test | 10,000 |

The original training set was divided into training and validation subsets.

---

## 6. Small ViT Architecture

I implemented a small Vision Transformer so that the experiment could be trained within the available Colab resources.

The main configuration was:

| Component | Configuration |
|---|---|
| Dataset | CIFAR-10 |
| Image size | 32 × 32 |
| Patch size | 4 × 4 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer blocks | 2 |
| MLP dimension | 256 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.05 |
| Main training | 20 epochs |
| Random seed | 42 |

The same basic architecture was used for both models:

1. **LayerNorm ViT**
2. **DyT ViT**

For the main DyT comparison, alpha was initialized to:

```text
alpha = 0.5
```

The LayerNorm and DyT models were trained with the same main hyperparameters.

---

# 7. ViT LayerNorm vs. DyT

The final 20-epoch test results were:

| Model | Test Loss | Test Accuracy | Parameters |
|---|---:|---:|---:|
| LayerNorm ViT | **1.0413** | **65.12%** | 281,226 |
| DyT ViT, alpha = 0.5 | 1.1312 | 58.81% | 281,231 |

The accuracy difference was:

\[
58.81 - 65.12 = -6.31
\]

Therefore, in this reduced-scale experiment, the DyT model was **6.31 percentage points below LayerNorm** in test accuracy.

The parameter counts were almost identical:

```text
LayerNorm: 281,226
DyT:       281,231
```

Only five additional parameters were introduced by replacing LayerNorm with the DyT implementation used in this project.

An important observation from this experiment is that DyT did not outperform LayerNorm under the particular architecture, training schedule, and reduced-scale setup used here.

---

# 8. Learned Alpha Values in the ViT

After the 20-epoch DyT experiment, the learned alpha values were:

| DyT Layer | Learned Alpha |
|---|---:|
| `blocks.0.norm1` | 0.708790 |
| `blocks.0.norm2` | 0.818662 |
| `blocks.1.norm1` | 0.757808 |
| `blocks.1.norm2` | 0.865841 |
| `final_norm` | 0.876798 |

The initial value was 0.5, but the learned values moved to approximately 0.71–0.88.

This shows that alpha was actively optimized during training rather than remaining fixed at its initialization.

---

# 9. ViT Alpha Initialization Ablation

To investigate the sensitivity of DyT to the initial value of alpha, I trained the small DyT ViT for five epochs using:

```text
alpha = 0.1
alpha = 0.5
alpha = 1.0
alpha = 2.0
```

The results were:

| Initial Alpha | Best Epoch | Best Validation Accuracy | Best Validation Loss | Mean Epoch Time |
|---:|---:|---:|---:|---:|
| 0.1 | 5 | 36.00% | 1.7303 | 12.69 s |
| 0.5 | 5 | 43.90% | 1.5215 | 12.66 s |
| 1.0 | 5 | 52.40% | 1.3229 | 12.69 s |
| 2.0 | 5 | **55.86%** | **1.2249** | 12.75 s |

Among the tested initializations, `alpha = 2.0` produced the best validation performance after five epochs.

This experiment suggests that alpha initialization can have a noticeable effect on the early optimization behavior of DyT.

However, I do not consider `alpha = 2.0` to be a universally optimal value because the ablation used only five epochs and four initialization values.

---

# 10. Tiny Shakespeare GPT

The second part of the project applies the same LayerNorm vs. DyT idea to a small GPT-style language model.

The Tiny Shakespeare dataset was downloaded and processed using **character-level tokenization**.

The dataset contained:

```text
Total characters: 1,115,394
Vocabulary size: 65
Training characters: 1,003,854
Validation characters: 111,540
```

The data was split into:

- 90% training
- 10% validation

A character-level model predicts the next character from a sequence of previous characters.

---

# 11. Small GPT Architecture

The GPT model uses causal self-attention so that a token cannot attend to future positions.

The training configuration was:

| Component | Configuration |
|---|---|
| Dataset | Tiny Shakespeare |
| Tokenization | Character-level |
| Vocabulary size | 65 |
| Context length | 128 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer blocks | 2 |
| MLP dimension | 256 |
| Batch size | 64 |
| Training steps | 1,000 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Random seed | 42 |

Two models were compared:

1. GPT with LayerNorm
2. GPT with DyT

For the main DyT experiment:

```text
alpha = 0.5
```

The GPT implementation also uses gradient clipping with a maximum norm of 1.0.

---

# 12. GPT LayerNorm vs. DyT

After 1,000 training steps, the results were:

| Model | Validation Loss | Perplexity | Training Time |
|---|---:|---:|---:|
| LayerNorm GPT | **2.2218** | **9.22** | 18.06 s |
| DyT GPT, alpha = 0.5 | 2.4111 | 11.15 | **16.79 s** |

For language modeling, lower validation loss and lower perplexity indicate better performance.

Therefore, LayerNorm performed better than DyT in this experiment.

The DyT model took slightly less training time in this particular run, but the difference was small:

```text
LayerNorm: 18.06 s
DyT:       16.79 s
```

I therefore do not treat this short timing difference as evidence that DyT is generally faster. A stronger speed comparison would require multiple runs and a more carefully controlled benchmarking setup.

---

# 13. GPT Alpha Initialization Ablation

For the GPT experiment, I tested:

```text
alpha = 0.5
alpha = 1.0
alpha = 2.0
```

The results were:

| Initial Alpha | Validation Loss | Perplexity | Training Time |
|---:|---:|---:|---:|
| 0.5 | 2.4111 | 11.1463 | 16.79 s |
| 1.0 | **2.3928** | **10.9438** | 18.39 s |
| 2.0 | 2.4139 | 11.1772 | 16.42 s |

Among the tested values, `alpha = 1.0` produced the lowest validation loss and perplexity.

The differences were relatively small compared with the ViT alpha ablation.

---

# 14. Main Experimental Observations

The experiments led to several observations.

### Observation 1 — DyT adds very few parameters

The parameter counts were:

| Model | LayerNorm | DyT |
|---|---:|---:|
| Small ViT | 281,226 | 281,231 |
| Small GPT | 298,305 | 298,310 |

The difference was only five parameters in each implementation.

This is one of the attractive properties of the DyT formulation: replacing the normalization layer does not require a large increase in model size.

### Observation 2 — LayerNorm performed better in my reduced-scale experiments

For both the ViT and GPT main comparisons, LayerNorm achieved better evaluation metrics than DyT with the configurations used in this project.

This does **not** mean that DyT does not work.

The experiments are much smaller than the experiments in the original paper, and the training settings, model sizes, and datasets are different.

Therefore, the result should be interpreted as:

> Under my reduced-scale experimental setup, LayerNorm performed better than DyT.

### Observation 3 — Alpha initialization affects training

The ViT experiment showed a relatively large difference between the tested alpha initializations.

The GPT experiment showed a smaller difference, with alpha = 1.0 producing the best result among the tested values.

### Observation 4 — Alpha is learned during training

The learned alpha values changed from their initial values during training.

For example, in the 20-epoch ViT experiment, alpha started at 0.5 and the learned values reached approximately 0.71–0.88 across the DyT layers.

---

# 15. Figures

The following figures were generated directly from the experiment histories:

### ViT accuracy

`figures/vit_accuracy.png`

Shows the training and validation accuracy of the LayerNorm and DyT ViT models over 20 epochs.

### ViT loss

`figures/vit_loss.png`

Shows the training and validation loss of the LayerNorm and DyT ViT models.

### GPT loss

`figures/gpt_loss.png`

Shows the training and validation loss during GPT training.

### GPT alpha ablation

`figures/gpt_alpha_ablation.png`

Shows the effect of different alpha initialization values on GPT validation performance.

---

# 16. Results Files

The `results/` directory contains the numerical experiment results:

```text
results/
├── experiment_config.json
├── summary.json
├── vit_comparison.csv
├── vit_alpha_ablation.csv
├── gpt_comparison.csv
└── gpt_alpha_ablation.csv
```

These files make it easier to inspect and reuse the experimental results without manually copying values from the notebook.

---

# 17. Repository Structure

```text
transformers-without-normalization-dyt/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── dyt_experiments.ipynb
│
├── figures/
│   ├── vit_accuracy.png
│   ├── vit_loss.png
│   ├── gpt_loss.png
│   └── gpt_alpha_ablation.png
│
└── results/
    ├── experiment_config.json
    ├── summary.json
    ├── vit_comparison.csv
    ├── vit_alpha_ablation.csv
    ├── gpt_comparison.csv
    └── gpt_alpha_ablation.csv
```

---

# 18. How to Run

The project was developed using Google Colab.

### Step 1 — Open the notebook

Open:

```text
notebooks/dyt_experiments.ipynb
```

in Google Colab.

### Step 2 — Select a GPU

A CUDA-capable GPU is recommended.

The reported experiments were performed using an NVIDIA Tesla T4.

### Step 3 — Install dependencies

The main libraries used are:

```text
torch
torchvision
numpy
pandas
matplotlib
```

The required packages are listed in `requirements.txt`.

### Step 4 — Run the notebook

Run the notebook cells in order.

The notebook covers:

1. Environment setup and random seed.
2. Dynamic Tanh implementation.
3. DyT implementation test.
4. CIFAR-10 loading.
5. Small ViT definition.
6. LayerNorm vs. DyT ViT training.
7. 20-epoch ViT comparison.
8. ViT alpha ablation.
9. Tiny Shakespeare loading.
10. Small GPT definition.
11. LayerNorm vs. DyT GPT training.
12. GPT alpha ablation.
13. Result generation.
14. Plot generation.
15. Saving results to CSV/JSON files.

---

# 19. Reproducibility

The main environment used for the experiments was:

```text
Python       : 3.13.15
PyTorch      : 2.11.0+cu130
GPU          : NVIDIA Tesla T4
GPU memory   : approximately 14.56 GB
Random seed  : 42
```

For the main ViT comparison, the LayerNorm and DyT models were initialized using the same random seed and trained using loaders created with the same shuffle seed.

The alpha ablation experiments were also reinitialized with controlled seeds so that the configurations could be compared more consistently.

Small numerical differences may occur when running the notebook on different hardware, CUDA versions, or software versions.

---

# 20. Limitations

This project has several limitations:

- The models are much smaller than the models used in the original paper.
- The ViT and GPT experiments use relatively short training schedules.
- The experiments were performed on a single GPU environment.
- The alpha ablations cover only a small number of initialization values.
- The experiments do not reproduce every dataset, model, and experiment from the original paper.
- The reported experiments are individual runs rather than repeated runs with different random seeds.
- The timing measurements are from short training runs and should not be treated as comprehensive hardware benchmarks.

Because of these limitations, the results should be viewed as a **reduced-scale experimental investigation of DyT**, rather than a complete reproduction of the original paper.

---

# 21. What I Learned

The main thing I wanted to understand through this project was whether replacing a standard Transformer normalization layer with a relatively simple learnable nonlinear function could work in practice.

Implementing DyT myself helped me understand the role of the learnable alpha, gamma, and beta parameters rather than treating the method as just another layer from a library.

The experiments also showed me that reproducing a paper at a smaller scale can give different results from the original work. In my experiments, LayerNorm performed better than DyT for both the small ViT and small GPT configurations.

The alpha ablation was particularly useful because it showed that the initialization of a seemingly simple parameter can noticeably affect early training behavior.

Overall, the project helped me understand not only the DyT equation itself, but also how model architecture, initialization, optimization, training duration, and evaluation choices affect the outcome of a paper implementation.

---

# 22. Acknowledgement

This project was completed as part of an individual Deep Learning paper implementation assignment.

The original paper was studied to understand the motivation and formulation of Dynamic Tanh. The core experimental implementation in this repository was written in PyTorch for this project.

If any external implementation or repository was consulted during development, it should be cited appropriately in the final report and repository.

---

## Citation

If you use or refer to this project, please cite the original paper:

```bibtex
@inproceedings{zhu2025transformers,
  title={Transformers without Normalization},
  author={Zhu, Jiachen and Chen, Yifei and He, Kaiming and LeCun, Yann and Liu, Zhuang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
