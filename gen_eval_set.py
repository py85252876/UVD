import pandas as pd
import numpy as np
import torch

file_path = 'The label file directory'
df = pd.read_csv(file_path)

column_order = ['5', '2', '1', '3', '4']

eval_videos_set = set()

column_eval_indices = {}
all_unsafe_videos = set()

for column in column_order:
    video_indices = df[column].dropna().unique()
    remaining_indices = list(set(video_indices) - all_unsafe_videos)
    all_unsafe_videos.update(video_indices)
    num_eval_videos = max(1, int(len(remaining_indices) * 0.2)) 
    eval_indices = np.random.choice(remaining_indices, num_eval_videos, replace=False)
    eval_videos_set.update(eval_indices)
    
    column_eval_indices[column] = eval_indices


eval_videos = list(eval_videos_set)

torch.save(eval_videos, 'eval.pth')


total_unsafe_videos = len(all_unsafe_videos)
