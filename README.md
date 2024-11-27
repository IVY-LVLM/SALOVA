<div align="center">

# **SALOVA: Segment-Augmented Long Video Assistant for Targeted Retrieval and Routing in Long-Form Video Analysis**

**Junho Kim\***, **Hyunjun Kim\***, Hosu Lee, Yong Man Ro  
Integrated Vision and Language Lab, KAIST  
\*Indicates Equal Contribution


[📄 **Paper**](https://arxiv.org/pdf/2411.16173)     [🤗 **Dataset (SceneWalk)**](https://huggingface.co/datasets/IVLLab/SceneWalk)

🏆 **Accepted at CVPR 2025**

<img src="assets/image.png" alt="SALOVA teaser" width="80%"/>

</div>

## 🛠 Environment Setup

We recommend using **CUDA 11.8** with the following core dependencies:

  * **Python:** ≥ 3.8
  * **PyTorch:** 2.1.2
  * **Torchvision:** 0.16.2
  * **Core Deps:** `deepspeed==0.14.2`, `transformers` (specific commit), `accelerate`, `timm`, `decord` / `av`, `wandb`.

### Installation

You can install the package in editable mode or via the requirements file:

```bash
# Option 1: Editable install
pip install -e .[train]

# Option 2: Requirements file
pip install -r requirements.txt
```

-----

## 📂 Directory Structure

  * `llava/`: Core source code including models, multimodal encoders/projectors/resamplers, trainer, and serving logic.
  * `scripts/train/`: Modular scripts for each training stage (`stage1_pretrain.sh`, `stage2_longv.sh`, `stage3_finetune.sh`).
  * `train.sh`: The entry point script that orchestrates the 3-stage pipeline.

-----

## 💾 Data Preparation

Please configure the following **Environment Variables** before running the scripts. The pipeline relies on these paths to locate your data.

| Variable | Description |
| :--- | :--- |
| `DATA_PATH` | Path to the training metadata JSON file. |
| `IMAGE_FOLDER` | Root directory for image datasets. |
| `VIDEO_FOLDER` | Root directory for video datasets. |
| `OUTPUT_DIR` | Directory to save checkpoints (defaults to script settings if unset). |

> **Note:** The default Vision/Text towers are set to `google/siglip-so400m-patch14-384`. You can override these by setting `VISION_TOWER` and `TEXT_TOWER` environment variables if necessary.

-----

## 🚀 Training Pipeline

The training process is divided into three distinct stages. You can execute them sequentially using `train.sh`, or customize the hyperparameters directly within the individual scripts.

### Stage 1: Multimodal Pretraining

Focuses on aligning the modalities using the Perceiver Resampler.

```bash
sh scripts/train/stage1_pretrain.sh <base-llm> <pretrained_mm_or_None> <batch> <run_name> <ver> <acc_step>
```

### Stage 1.5: Long Video Knowledge Injection

Adapts the model for long-context video understanding.

```bash
sh scripts/train/stage2_longv.sh <base-llm> <stage1_ckpt_or_delta> <batch> <run_name> <ver> <acc_step>
```


### Stage 2: Video Instruction Tuning

Final Supervised Fine-Tuning (SFT) for instruction following.

```bash
sh scripts/train/stage3_finetune.sh <base-llm> <stage2_ckpt_or_delta> <batch> <run_name> <ver> <acc_step>
```

-----

## 💻 Usage

To run the full pipeline, export your data paths and execute the main shell script.

```bash
# 1. Export Data Paths
export DATA_PATH=/path/to/your/llava_pretrain.json
export IMAGE_FOLDER=/path/to/your/images
export VIDEO_FOLDER=/path/to/your/videos
export OUTPUT_DIR=/path/to/save/checkpoints

# 2. Run the pipeline
bash train.sh
```


## 📚 Citation

If you find this repository helpful for your research, please consider citing our paper:

```bibtex
@inproceedings{kim2025salova,
  title={Salova: Segment-augmented long video assistant for targeted retrieval and routing in long-form video analysis},
  author={Kim, Junho and Kim, Hyunjun and Lee, Hosu and Ro, Yong Man},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={3352--3362},
  year={2025}
}
