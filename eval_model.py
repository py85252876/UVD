import av
import torch
import pandas as pd
import torch.nn as nn
import numpy as np 
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import gc
import argparse
from torch.optim import AdamW
import warnings
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.feature_extraction_utils")
from sklearn.model_selection import train_test_split
from transformers import AutoProcessor, XCLIPVisionModel, get_linear_schedule_with_warmup, AutoModel,VivitImageProcessor,VivitModel, AutoImageProcessor,VideoMAEModel,VideoMAEForVideoClassification
from tqdm import tqdm
from collections import defaultdict


import time
np.random.seed(int(time.time()))

def find_video_files(directory,eval_list=[]):
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4") and int(file.split('.')[0]) not in eval_list:
                full_path = os.path.join(root, file)
                video_files.append(full_path)
    return video_files

class CreateDataset(torch.utils.data.Dataset):
    def __init__(self,videos_file,labels,processor):
        self.videos_file = videos_file
        self.labels = labels
        self.processor = processor
    def __len__(self):
        return len(self.videos_file)

    def __getitem__(self,item):
        try:
            container = av.open(self.videos_file[item])
            indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
            video = self.read_video_pyav(container, indices)
            processed_video = self.processor(list(video), return_tensors="pt")
            pro_video = processed_video['pixel_values']
        except av.error.InvalidDataError as e:
            print(f"wrong file {video_file}: {e}")
        except Exception as e:
            print(f"mistake {video_file}: {e}")
        return {
            'input': pro_video,
            'label': self.labels[item],
            'file' : self.videos_file[item].split('/')[-1].split('.')[0]
        }

    def read_video_pyav(self,container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])


    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        clip_len = min(clip_len, seg_len)
        indices = list(range(0, clip_len * frame_sample_rate, frame_sample_rate))

        return indices

def CreateDataLoader(df,processor,batch_size):
    ds = CreateDataset(videos_file = df['video_path'],
                        labels = df['labels'],
                        processor = processor)
    return ds

class DetectionSystem:
    def __init__(self, model_paths, num_labels=2):
        
        self.models = []
        for model_path in model_paths:
            model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", cache_dir="./huggingface/hub",num_labels = 2)
            model.load_state_dict(torch.load(model_path))
            model.to(f"cuda:{args.gpu_num}")
            model.eval() 
            self.models.append(model)
    
    def process_input(self, input_data):
        outputs = []
        for model in self.models:
            with torch.no_grad():
                output = model(input_data)
                outputs.append(output.logits)
                index, preds = torch.max(output.logits, dim = 1)
                if preds == 1:
                    return 1
        return 0

def eval_model(model, data_loader, n_examples,data_dict):
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_vids = d['input'].to(f"cuda:{args.gpu_num}")
            label = d['label'].to(f"cuda:{args.gpu_num}")
            input_video = input_vids.squeeze(1)
            output = model.process_input(input_video)
            data_dict[str(label.item())][str(d['file'][0])].append(output)
            
            
    return data_dict

def default_list_dict():
    return defaultdict(list)

def main(args):
    print("load data...", flush=True)
    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", cache_dir="./huggingface/hub")
    

    data_dict = defaultdict(default_list_dict)
    for i in range(50):
        module_list = []
        print(f"this is {i}")
        for j in range(1,6,1):    
            module_list.append(f"{args.model_dir}/{j}/{i}.pth")
        System = DetectionSystem(module_list)
        label1_path = find_video_files(f"{args.eval_dir}/{i}")
        label0_path = find_video_files(f"{args.data_dir}/{i}")
        label0_path = label0_path[:len(label1_path)]

        label1 = np.full(len(label1_path),1)
        label0 = np.full(len(label0_path),0)
        label0_path = np.array(label0_path)
        label1_path = np.array(label1_path)

        labels = np.concatenate((label1,label0))
        
        video_path = np.concatenate((label1_path,label0_path))
        
        
        print("load data...", flush=True)
        data={}
        data['video_path'] = video_path
        data['labels'] = labels

        df_data = pd.DataFrame(data)
        ds = CreateDataLoader(df_data,processor,1)
        val_data_loader = torch.utils.data.DataLoader(ds,batch_size=1,num_workers = 2,drop_last=True)

        data_dict = eval_model(System, val_data_loader, len(df_data),data_dict)
        del System.models,data,val_data_loader,ds
        torch.cuda.empty_cache()
        gc.collect()
    torch.save(data_dict,args.save_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None) 
    parser.add_argument("--model_dir", type=str, default=None) 
    parser.add_argument('--train', action='store_true', help='Enable training')
    parser.add_argument('--no-train', action='store_false', dest='train', help='Disable training')
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--eval_dir", type=str, default=None) 
    parser.add_argument("--save_dir", type=str, default=None) 
    args = parser.parse_args()
    
    main(args)