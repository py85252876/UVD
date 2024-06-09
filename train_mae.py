import av
import torch
import pandas as pd
import torch.nn as nn
import numpy as np 
import os
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
import time
np.random.seed(int(time.time()))


class CreateDataset(torch.utils.data.Dataset):
    def __init__(self,videos_file,labels,processor):
        self.videos_file = videos_file
        self.labels = labels
        self.processor = processor
        self.processed_video = []
        self.processed_label = []
        for video_file,label in zip(self.videos_file,self.labels):
            try:
                print(len(self.processed_video))
                container = av.open(video_file)
                indices = self.sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
                video = self.read_video_pyav(container, indices)
                processed_video = self.processor(list(video), return_tensors="pt")
                pro_video = processed_video['pixel_values']
                if pro_video.shape[1]==16:
                    self.processed_video.append(pro_video)
                    self.processed_label.append(label)
            except av.error.InvalidDataError as e:
                print(f"wrong file {video_file}: {e}")
            except Exception as e:
                print(f"mistake {video_file}: {e}")
    def __len__(self):
        return len(self.processed_video)

    def __getitem__(self,item):
        return {
            'input':self.processed_video[item],
            'label': torch.tensor(self.processed_label[item])
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
    return torch.utils.data.DataLoader(ds,batch_size=4,num_workers = 2,drop_last=True)


def find_video_files(directory,eval_list=[]):
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4") and int(file.split('.')[0]) not in eval_list:
                full_path = os.path.join(root, file)
                video_files.append(full_path)
    return video_files

def train_model(model, data_loader, loss_fn, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for d in tqdm(data_loader, desc="Training", leave=False, file=sys.stdout):
        input_vids = d['input'].to(f"cuda:{args.gpu_num}")
        label = d['label'].to(f"cuda:{args.gpu_num}")
        input_video = input_vids.squeeze(1)
        output = model(pixel_values = input_video)
        _, preds = torch.max(output.logits , dim = 1)
        loss = loss_fn(output.logits, label)
        
        correct_predictions += torch.sum(preds == label)
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        all_preds = []
        all_labels = []
        for d in data_loader:
            input_vids = d['input'].to(f"cuda:{args.gpu_num}")
            label = d['label'].to(f"cuda:{args.gpu_num}")
            input_video = input_vids.squeeze(1)
            output = model(input_video)
            index, preds = torch.max(output.logits, dim = 1)
            
            correct_predictions += torch.sum(preds == label)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
        total_correct = correct_predictions.double() / len(data_loader.dataset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
            
        return correct_predictions.double() / n_examples, np.mean(losses), all_labels, all_preds


def main(args):
    if args.train:
        if args.eval_dir is not None:
            eval_list = torch.load(args.eval_dir)
        else:
            eval_list = []

        label1_path = find_video_files(f"{args.data_dir}/group{args.group_num}/{args.step_num}", eval_list)
        label0_path = find_video_files(f"{args.normal_video_dir}/{args.step_num}")[:len(label1_path)]
        

        label1 = np.full(len(label1_path),1)
        label0 = np.full(len(label0_path),0)
        label0_path = np.array(label0_path)
        label1_path = np.array(label1_path)

        labels = np.concatenate((label1,label0))
        
        video_path = np.concatenate((label1_path,label0_path))

        print("load data...", flush=True)
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", cache_dir="./huggingface/hub")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", cache_dir="./huggingface/hub",num_labels = 2)

        video_cls = model
        video_cls = video_cls.to(f"cuda:{args.gpu_num}")
        print("load data...", flush=True)
        data={}
        data['video_path'] = video_path
        data['labels'] = labels

        df_data = pd.DataFrame(data)
        df_train, df_val = train_test_split(df_data,test_size = 0.1, random_state = 2024, stratify=df_data['labels'])
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        train_data_loader = CreateDataLoader(df_train,processor,4)
        val_data_loader = CreateDataLoader(df_val,processor,4)

        EPOCHS = args.epoch

        LR = 1e-5

        optimizer = AdamW(video_cls.parameters(), lr = LR)
        total_steps = len(train_data_loader) * EPOCHS

        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)

        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in tqdm(range(EPOCHS), desc="Epochs", file=sys.stdout):
            print(f'Epoch {epoch + 1}/{EPOCHS}')
            print('-' * 10)
            
            train_acc, train_loss = train_model(video_cls, train_data_loader, loss_fn, optimizer, scheduler, len(df_train))
            print(f'Train Loss: {train_loss} ; Train Accuracy: {train_acc}')
            
            val_acc, val_loss, _, _ = eval_model(video_cls, val_data_loader, len(df_val))
            print(f'Val Loss: {val_loss} ; Val Accuracy: {val_acc}')
        path = os.path.join(args.save_dir, str(args.group_num))
        os.makedirs(path, exist_ok=True)
        torch.save(video_cls.state_dict(), f"{path}/{args.step_num}.pth")
    else:
        print("start eva .. ",flush=True)
        label1_path = find_video_files(f"{args.data_dir}/group{args.group_num}/{args.step_num}")
        label0_path = find_video_files(f"{args.data_dir}/invid/{args.step_num}")[:len(label1_path)]
        print(len(label0_path))
        label1 = np.full(len(label1_path),1)
        label0 = np.full(len(label0_path),0)
        label0_path = np.array(label0_path)
        label1_path = np.array(label1_path)

        labels = np.concatenate((label1,label0))
        
        video_path = np.concatenate((label1_path,label0_path))

        print("load data...")
        processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base", cache_dir="./huggingface/hub")
        model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", cache_dir="./huggingface/hub",num_labels = 2)

        model.load_state_dict(torch.load(f"{args.save_dir}/{args.group_num}/{args.step_num}.pth"))
        model = model.to(f"cuda:{args.gpu_num}")
        model.eval()
        print("load model...")
        data={}
        data['video_path'] = video_path
        data['labels'] = labels

        df_data = pd.DataFrame(data)
        val_data_loader = CreateDataLoader(df_data,processor,4)
        val_acc, val_loss,all_labels, all_preds = eval_model(model, val_data_loader, len(df_data))
        with open(f'{args.save_dir}/classification_{args.group_num}.txt', 'a') as f:
            f.write(f"Classification Report for {args.step_num}-th step\n")
            f.write(classification_report(all_labels,all_preds))
            f.write("\n\n")
        del model,data,val_data_loader
        torch.cuda.empty_cache()
        gc.collect()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None) 
    parser.add_argument("--normal_video_dir", type=str, default=None)
    parser.add_argument('--train', action='store_true', help='Enable training')
    parser.add_argument('--no-train', action='store_false', dest='train', help='Disable training')
    parser.add_argument("--group_num", type=int, required=True)
    parser.add_argument("--step_num", type=int, required=False) #denoising step, range from 1~50 in our work
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--eval_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)   
    args = parser.parse_args()
    
    main(args)