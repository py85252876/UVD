import os
import json
import time
import torch
import random
import inspect
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torchvision
import imageio
from omegaconf import OmegaConf
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
import gc
from utils.unet import UNet3DConditionModel
from utils.pipeline_magictime import MagicTimePipeline
from utils.util import save_videos_grid
from utils.util import load_weights
import pandas as pd
import os
from einops import rearrange


def copy_videos_by_group(csv_file_path):
    
    df = pd.read_csv(csv_file_path)
    data = {}
    for col in df.columns:
        video_numbers = df[col].dropna().astype(int).tolist()
        data[f"{col}"] = video_numbers

    return data
        

@torch.no_grad()
def main(args,label):
    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    if 'counter' not in globals():
        globals()['counter'] = 0
    unique_id = globals()['counter']
    globals()['counter'] += 1

    model_config = OmegaConf.load(args.config)[0]
    inference_config = OmegaConf.load(args.config)[1]

    if model_config.magic_adapter_s_path:
        print("Use MagicAdapter-S")
    if model_config.magic_adapter_t_path:
        print("Use MagicAdapter-T")
    if model_config.magic_text_encoder_path:
        print("Use Magic_Text_Encoder")

    samples = []

    # create validation pipeline
    tokenizer    = CLIPTokenizer.from_pretrained(model_config.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_config.pretrained_model_path, subfolder="text_encoder").cuda()
    vae          = AutoencoderKL.from_pretrained(model_config.pretrained_model_path, subfolder="vae").cuda()
    unet = UNet3DConditionModel.from_pretrained_2d(model_config.pretrained_model_path, subfolder="unet",
                                                   unet_additional_kwargs=OmegaConf.to_container(
                                                       inference_config.unet_additional_kwargs)).cuda()

    # set xformers
    if is_xformers_available() and (not args.without_xformers):
        unet.enable_xformers_memory_efficient_attention()

    pipeline = MagicTimePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
    ).to("cuda")

    pipeline = load_weights(
        pipeline,
        motion_module_path=model_config.get("motion_module", ""),
        dreambooth_model_path=model_config.get("dreambooth_path", ""),
        magic_adapter_s_path=model_config.get("magic_adapter_s_path", ""),
        magic_adapter_t_path=model_config.get("magic_adapter_t_path", ""),
        magic_text_encoder_path=model_config.get("magic_text_encoder_path", ""),
    ).to("cuda")
    data  = torch.load("add your pred x_0 data directory")

    for key in list(label.keys()):
        temp_data = label[key]
        for temp in temp_data:
            for i in range(50):
                x_0 = data[temp:temp+1,i:i+1,:,:,:,:,].squeeze(0).to("cuda")
                video = pipeline.decode_latents(x_0)
                video = torch.from_numpy(video)
                videos = rearrange(video, "b c t h w -> t b c h w")
                outputs = []
                rescale = False
                path = f"./MagicTime/trained_detector_data/{key}/{i}/{temp}.mp4"
                for x in videos:
                    x = torchvision.utils.make_grid(x, nrow=6)
                    x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    if rescale:
                        x = (x + 1.0) / 2.0  # -1,1 -> 0,1
                    x = (x * 255).numpy().astype(np.uint8)
                    outputs.append(x)

                os.makedirs(os.path.dirname(path), exist_ok=True)
                imageio.mimsave(path, outputs, fps=8)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--without-xformers", action="store_true")
    parser.add_argument("--label_file_dir", type=str, required=True)
    args = parser.parse_args()

    label = copy_videos_by_group(args.label_file_dir)
    
    main(args,label)
