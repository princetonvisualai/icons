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

First, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/princetonvisualai/icons.git
cd icons
```

To set up the environment for ICONS and LLaVA training (https://github.com/haotian-liu/LLaVA/), you can use the provided `environment.yml` file to create a Conda environment:

```bash
conda create -n icons python=3.10 -y
conda activate icons
pip install --upgrade pip
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```


## Dataset Download
LLaVA-665K dataset is available on [Download Link](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json).

Cambrian-7M dataset is available on [Download Link](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M/blob/main/jsons/Cambrian7M_withsystemprompt.jsonl).

Then follow the original repo to download the image data.

You can split the data into random chunks for parallel gradient computation using slurm scripts. For efficient processing, request as many CPUs as possible (e.g., 96 CPUs), as the splitting operation is CPU-intensive and can be parallelized. For example to split the 7M Cambrian dataset into 3000 chunks with 96 CPUs takes about 15-20 minutes.

```bash
# Split the LLaVA-665K dataset into chunks (request 32+ CPUs for faster processing)
python utils/split.py path/to/llava_v1_5_mix665k.json data/llava_665k_splits --num-splits 200

# Split the Cambrian-7M dataset into chunks (request 32+ CPUs for faster processing)
python utils/split.py path/to/Cambrian7M_withsystemprompt.jsonl data/cambrian_7m_splits --num-splits 2000
```

## Selection

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

## Training 
We follow the training pipeline from [LLaVA's official repository](https://github.com/haotian-liu/LLaVA/) and use the selected data for training. The training script for LLaVA-1.5-7B is located in `./llava_train_scripts/finetune_lora.py`.

Before training, download the required checkpoint files:

> ⚠️ **Note**: We use the LLaVA model checkpoints from before the visual instruction tuning stage (i.e., before training on the 665K instruction data). These checkpoints only contain the pretrained vision-language alignment weights.

<details>
<summary>7B Vicuna Model + Projector Checkpoint Download</summary>

```bash
# Download the mm_projector.bin file for LLaVA-1.5-7B training
mkdir -p checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5

wget https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5/resolve/main/mm_projector.bin -P checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5

# Download Vicuna-7B-v1.5 base model
git clone https://huggingface.co/lmsys/vicuna-7b-v1.5 checkpoints/vicuna-7b-v1.5
```
</details>

<details>
<summary>13B Vicuna Model + Projector Checkpoint Download</summary>

```bash
# Download the mm_projector.bin file for LLaVA-1.5-13B training
mkdir -p checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5

wget https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/resolve/main/mm_projector.bin -P checkpoints/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5

# Download Vicuna-13B-v1.5 base model
git clone https://huggingface.co/lmsys/vicuna-13b-v1.5 checkpoints/vicuna-13b-v1.5
```
</details>

<details>
<summary>8B Llama-3 Model + Projector Checkpoint Download</summary>

```bash
# Download the mm_projector.bin file for LLaVA-Llama-3-8B training
mkdir -p checkpoints/llava-llama-3-8b

# Download Llama-3-8B base model
git clone https://huggingface.co/xtuner/llava-llama-3-8b checkpoints/llava-llama-3-8b
```
</details>

To start training with the LLaVA-1.5-7B model:
```bash
sh llava_train_scripts/7b_finetune_lora.sh
```

Follow the instructions in the terminal to set the data_path and output_dir.


## Inference

For inference after training with selected data, you can choose one of the following two options:

1. Use the standard evaluation pipeline from [LLaVA's official evaluation script](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md).

2. Use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for comprehensive evaluation, which supports the evaluation on dozens of public datasets and allows new dataset onboarding.



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


