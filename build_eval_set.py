import os
import shutil
import torch

original_path = './MagicTime/train_detector_data'
new_path = './MagicTime/eval_set/'

if not os.path.exists(new_path):
    os.makedirs(new_path)

eval_list = torch.load("./MagicTime/eval.pth")  

for group in range(5):
    group_folder = os.path.join(original_path, f'group{group}')
    
    for i in range(50):
        original_subfolder = os.path.join(group_folder, str(i))
        new_subfolder = os.path.join(new_path, str(i))
        
        if not os.path.exists(new_subfolder):
            os.makedirs(new_subfolder)
        
        if os.path.exists(original_subfolder):
            for file_name in os.listdir(original_subfolder):
                file_index = int(file_name.split('.')[0])
                
                if file_index in eval_list:
                    src_file = os.path.join(original_subfolder, file_name)
                    dst_file = os.path.join(new_subfolder, file_name)
                    
                    shutil.copy(src_file, dst_file)
                    print(f"Copied {src_file} to {dst_file}")
