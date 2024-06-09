# Towards Understanding Unsafe Video Generation

In this study, we examined the capability of current video generation models to produce unsafe content. We compiled a dataset of 2,112 unsafe videos using unsafe prompts. Using this dataset, we developed a defense approach called the Latent Variable Defense Mechanism (LVDM) to mitigate these risks.

This repository contains:
1. Introducing how to generate a training dataset
2. Code for training detection model.
3. Providing code for evaluating LVDM performance.

## Table of Contents

- [📄 Table of Contents](#-table-of-contents)
- [🛠️ Download Dependencies](#-download-dependencies)
	- [Video Generation Model Setup](#video-generation-models-setup)
- [🚀 Model Training](#-model-training)
- [👀 Model Evaluation](#-model-evaluation)
- [🥰 Acknowledgement](#-acknowledgement)

## 🛠️ Download Dependencies

### Video Generation Model Setup

Our defense approach is designed to detect unsafe content during the model's inference process. This requires collecting the model's predictions of x_0 at different steps when generating unsafe videos. Thanks to MagicTime's open-source contribution, we can use MagicTime as an example. For detailed environment setup, please refer to the original repository.


   
