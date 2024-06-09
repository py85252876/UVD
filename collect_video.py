import pandas as pd
import numpy as np
import torch

# 读取Excel文件
file_path = '/bigtemp/trv3px/video_detection/MagicTime/trained_detector/output.csv'
df = pd.read_csv(file_path)

# 指定列的顺序
column_order = ['5', '2', '1', '3', '4']

# 存储验证集索引的集合
eval_videos_set = set()
# 存储每个列具体加入的索引
column_eval_indices = {}
# 存储所有的unsafe video索引
all_unsafe_videos = set()

# 遍历每一列（按照指定顺序），抽取20%的索引作为验证集
for column in column_order:
    video_indices = df[column].dropna().unique()
    remaining_indices = list(set(video_indices) - all_unsafe_videos)
    all_unsafe_videos.update(video_indices)
    num_eval_videos = max(1, int(len(remaining_indices) * 0.2))  # 至少抽取1个
    eval_indices = np.random.choice(remaining_indices, num_eval_videos, replace=False)
    eval_videos_set.update(eval_indices)
    
    # 记录每个列加入的索引
    column_eval_indices[column] = eval_indices
    
    # 打印每个列加入的索引数量
    print(f'列 {column} 加入了 {len(eval_indices)} 个视频索引: {eval_indices}')

# 将验证集索引集合转换为列表
eval_videos = list(eval_videos_set)

# 将验证集索引保存到eval.pth文件
torch.save(eval_videos, 'eval.pth')

print(f'验证集已保存到eval.pth文件，共包含{len(eval_videos)}个视频索引。')

# 统计总的unsafe video数量
total_unsafe_videos = len(all_unsafe_videos)
print(f'总的unsafe video数量: {total_unsafe_videos}')
