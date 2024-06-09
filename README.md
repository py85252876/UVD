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

Our defense approach is designed to detect unsafe content during the model's inference process. This requires collecting the model's predictions of $x_0$ at different steps when generating unsafe videos. Thanks to [MagicTime](https://github.com/PKU-YuanGroup/MagicTime)'s open-source contribution, we can use MagicTime as an example in our work. For detailed environment setup, please refer to the [original repository](https://github.com/PKU-YuanGroup/MagicTime).

### Training Environment Setup

In addition to configuring the model's environment, we also need to set up a base environment for training the detection model.

```bash
pip install -r requirements.txt
```

### Model Training

Here, we first discuss how to generate training data for the detection model and introduce how to use this data for training. 

First, you need to prepare a `prompt.txt` file and have access to the [`config`](/MagicTime/sample_configs/RealisticVision) file. In this file, specify the path to your `prompt.txt` file and the desired path to save `pred_x0.pth`. The `pred_x0.pth` file contains the data required for the detection model.

After obtaining `pred_x0.pth`, you can use [`build_x0.py`](/build_x0.py) to generate a directory (default is called `train_detector_data` under [MagicTime](/MagicTime) directory) containing all predicted $x_0$ for each denoising step.

Run:

```bash
python build_x0.py --config build_x0.yaml \
--label_file_dir "Your label file directory"
```

> Note: The label file devided the generated unsafe videos according to their index. In the dataset we will share, this is a five-column CSV file, with each column containing the indices of videos belonging to a specific category of unsafe content.

Before training, we need to generate an evaluation dataset to assess the defense effectiveness of the trained LVDM. Run the `gen_eval_data.py` script to generate a `eval.pth` file. This file is saved in the [MagicTime](./MagicTime) directory by default.

```bash
python gen_eval_set.py
```

Then run [`build_eval_set.py`](./build_eval_set.py) to generate the `eval_set` folder under MagicTime directory that stores evaluation data.

```bash
python build_eval_set.py
```

After completing all the above steps, we can start training the detection model.

```bash
python train_mae.py --data_dir "Your training train_detector_data folder directory " \
--eval_dir "Your eval.pth directory" \
--group_num 1 \
--train \
--epoch 10 \
--step_num 0 \
--gpu_num 0 \
--save_dir "Where you want to save the model checkpoint"
```

> Note: Including `eval.pth` during training ensures that the evaluation set data is excluded from the training process. Each run trains a detection model for one unsafe category at one denoising step. Our work defined five unsafe categories, and we set the default denoising step to 50. Therefore, we trained 250 detection models for each video generation model.
> The training label unsafe video and class 1, you also need to generate the same number of harmful videos and use them in the training process. We used [InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid) captions to synthesize normal video in our work.

[👀 Model Evaluation](#-model-evaluation)

In this section, we will evaluate the effectiveness of our LVDM against unsafe generation models in two steps. 

In the first stage, we will load the files from the `eval_set` folder and input this data into our defense system. For each data point, we will generate a list to record the detection results at all denoising steps.

```bash
python embed_model.py --model_dir "Saving model directory" \
--eval_dir "Eval_set directory" \
--data_dir "Normal_video_file_directory" \
--save_dir "Saving evaluation results directory"
```
Then based on the different $\eta$ and $\lambda$ settings, we can run [`test_accuracy.py`](/test_accuracy.py) to see LVDM profermance.

```bash
python test_accuracy.py --data_dir "Your evaluation results directory" \
--lambda 0.6 \
--eta 20 
```


















   
