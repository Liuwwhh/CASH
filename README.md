# CASH: Capacity-Aware Selective Hashing for Continual Cross-Modal Retrieval

Official PyTorch implementation of **“Do More with Less: Capacity-Aware Selective Hashing for Continual Cross-Modal Retrieval”** (TPAMI submission/manuscript).

## Overview

**Continual cross-modal hashing** updates a hashing model as new tasks/categories arrive, while keeping **historical binary codes usable and comparable** under a *fixed bit budget*. Under this setting, semantics accumulate across tasks and the limited Hamming space becomes progressively saturated, increasing collisions and degrading neighborhood consistency.

We propose **Capacity-Aware Selective Hashing (CASH)** to explicitly improve bit utilization and stabilize long-term code compatibility under a fixed code length.

## Method Summary (CASH)

CASH is designed around two key ideas:

- **Task-adaptive prompt learning for continual update**  
  We freeze the underlying feature representations and introduce **lightweight task prompts**. A **Multi-Prompt Cross-Attention** module injects task-specific context into multimodal features for incremental adaptation.

- **Capacity-aware selective hashing under fixed bits**  
  A **Coarse-Fine Dual-Branch Hashing Network** produces two complementary code candidates (coarse/global vs. fine-grained). A **prompt-conditioned mask** performs **bit-wise selection** between branches, allocating each bit to the branch that is more discriminative, mitigating collision growth and cross-task interference.

> Figures:  
> - `Figure1` (Intro/Abstract figure) illustrates the motivation and the core components.  
> - `Figure2` (Framework figure) details the two-stage pipeline and module-level design.  
> Place your exported images under `assets/` and update the paths below:
>
> ![Figure1: Overview](assets/figure1.png)
> ![Figure2: Framework](assets/figure2.png)

## Repository Structure

```text
.
├── main.py        # entry point: sequential task training + evaluation
├── model.py       # CASH model: dual-branch hashing + prompts + mask + attention
├── train.py       # per-task training loop, validation, and continual evaluation
├── load_data.py   # dataloader + task-wise .mat feature loading
└── utils.py       # metrics (mAP@K), losses, logging, CSV dumping, seeding

```

## Datasets & Pre-trained CMH Models
1. Download datasets MSCOCO and NUSWIDE

```
MSCOCO
url: https://pan.baidu.com/s/1uJ5DgDIJIBRownazZXOWnA?pwd=2025
code: 2025

NUSWIDE
url: https://pan.baidu.com/s/17Rn92JwYELzV4YNQ2bndmg?pwd=2025
code: 2025

MSCOCO-Imbalance
url: https://pan.baidu.com/s/1gzUoMh3P-hH2iNysMxWSBA?pwd=2025
code: 2025

NUSWIDE-Imbalance
url: https://pan.baidu.com/s/1njmBa0j0EfeD_CzT0V4ZgA?pwd=2025
code: 2025
```

2. Change the value of `data_path` in file `main.py` to `/path/to/data`.

## Python Environment
``` bash
conda create -n CASH python=3.8
conda activate CASH
pip install -r requirements.txt
```

## Training
``` python
python main.py
```
