# Towards Understanding Unsafe Video Generation

In this study, we examined the capability of current video generation models to produce unsafe content. We compiled a dataset of 2,112 unsafe videos using unsafe prompts. Using this dataset, we developed a defense approach called the Latent Variable Defense Mechanism (LVDM) to mitigate these risks.

This repository contains:
1. Introducing how to generate a training dataset
2. Code for training detection model.
3. Providing code for evaluating LVDM performance.

## Table of Contents

- [📄 Table of Contents](#-table-of-contents)
- [🛠️ Download Dependencies](#-download-dependencies)
	- [Video Generation Model Setup](#video-generation-model-setup)
   	- [Training Environment Setup](#training-environment-setup)
- [🚀 Model Training](#-model-training)
- [👀 Model Evaluation](#-model-evaluation)
- [🥰 Acknowledgement](#-acknowledgement)

## 🛠️ Download Dependencies

### Video Generation Model Setup

Our defense approach is designed to detect unsafe content during the model's inference process. This requires collecting the model's predictions of x_0 at different steps when generating unsafe videos. Thanks to [MagicTime](https://github.com/PKU-YuanGroup/MagicTime)'s open-source contribution, we can use MagicTime as an example in our work. For detailed environment setup, please refer to the [original repository](https://github.com/PKU-YuanGroup/MagicTime).

### Training Environment Setup

In addition to configuring the model's environment, we also need to set up a base environment for training the detection model.

```bash
pip install -r requirements.txt
```

### Model Training

Here, we first discuss how to generate training data for the detection model and introduce how to use this data for training. 

First, you need to prepare a `prompt.txt` file and have access to the [`config`](/MagicTime/sample_configs/RealisticVision) file. In this file, specify the path to your `prompt.txt` file and the desired path to save `pred_x0.pth`. The `pred_x0.pth` file contains the data required for the detection model.

After obtaining `pred_x0.pth`, you can use `build_x0.py` to generate a directory containing all predicted $x_0$ for each denoising step.









   
