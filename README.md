# ICONS

**ICONS: Influence Consensus for Vision-Language Data Selection**

Under construction 🚧

[[paper](https://arxiv.org/abs/2501.00654)][[website](https://princetonvisualai.github.io/icons/)][[dataset](https://huggingface.co/datasets/xindiw/LLAVA-ICONS-133K)]

Authors: [Xindi Wu](https://xindiwu.github.io/), [Mengzhou Xia](https://xiamengzhou.github.io/), [Rulin Shao](https://rulinshao.github.io/), [Zhiwei Deng](https://lucas2012.github.io/), [Pang Wei Koh](https://koh.pw/), [Olga Russakovsky](https://www.cs.princeton.edu/~olgarus/)

We propose ICONS, a method for selecting vision-language data that optimizes training efficiency by identifying and prioritizing data samples that are consistently valuable across multiple tasks.

## News 🔥
- [01/25] We have released the LLAVA-ICONS-133K dataset on [Hugging Face](https://huggingface.co/datasets/xindiw/LLAVA-ICONS-133K) for public use.
- [12/24] We have released the paper [ICONS](https://arxiv.org/abs/2501.00654).

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)

## Installation

To set up the environment for ICONS, you can use the provided `environment.yml` file to create a Conda environment:

```bash
conda env create -f environment.yml
conda activate icons
```

## Usage

The ICONS pipeline consists of two main stages:

### Stage 1: Specialist (Computing Task-Specific Influence)

1. **Compute Training Data Gradients**
   ```bash
   # Submit SLURM jobs for processing training data chunks
   sbatch './scripts/0_slurm_train_grads.sh' 500  # or use other checkpoints, here we use ckpt=500 as an example
   ```

2. **Merge Gradient Files**
   ```bash
   bash ./scripts/1_merge_train_gradient.sh
   ```

3. **Process Validation Data**
   ```bash
   bash ./scripts/2_get_val_data_grads_all.sh
   ```

4. **Compute Influence Matrices**
   ```bash
   bash ./scripts/3_specialist.sh
   ```

### Stage 2: Generalist (Influence Consensus)

5. **Generate Consensus**
   ```bash
   bash ./scripts/4_generalist.sh
   ```


## Citation
If you find this repository useful for your research, please cite with the following BibTeX entry:
```
@article{wu2024icons,
  title={ICONS: Influence Consensus for Vision-Language Data Selection},
  author={Wu, Xindi and Xia, Mengzhou and Shao, Rulin and Deng, Zhiwei and Koh, Pang Wei and Russakovsky, Olga},
  journal={arXiv preprint arXiv:2501.00654},
  year={2024}
}
```


