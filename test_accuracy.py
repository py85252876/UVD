from sklearn.metrics import classification_report
import torch
import argparse
from collections import defaultdict
import random

def default_list_dict():
    return defaultdict(list)

# def calculate_predictions(data, lamada, eta):
#     y_true = []
#     y_pred = []
    
#     for label, videos in data.items():
#         for index, scores in videos.items():
#             # 取前 eta 个评分
#             top_scores = scores[:int(eta)]
#             # 计算这些评分的总和
#             score_sum = sum(top_scores)
#             # 判断是否大于等于 eta * lamada
#             prediction = 1 if score_sum >= eta * lamada else 0
#             # 真实标签
#             ground_truth = int(label)
            
#             # 添加到列表中
#             y_true.append(ground_truth)
#             y_pred.append(prediction)
    
#     return y_true, y_pred

def calculate_predictions(data, lamada, eta, excluded_keys=[],min_samples=None,seed=20240603):
    y_true = []
    y_pred = []
    random.seed(seed)
    # 确定每个类的样本数量
    if min_samples is None:
        # 如果未指定，则使用每个类的最小样本数量
        min_samples = min(len(videos) for videos in data.values())
    
    for label, videos in data.items():
        # 将视频索引列表打乱
        if int(label) == 1:
            indices = [key for key in videos.keys() if key not in excluded_keys]
        else:
            indices = [key for key in videos.keys()]
        random.shuffle(indices)
        # 只取前 min_samples 个样本
        selected_indices = indices[:min_samples]
        
        for index in selected_indices:
            scores = videos[index]
            # 取前 eta 个评分
            # top_scores = scores[:int(eta)]
            top_scores = scores[-int(eta):]
            # 计算这些评分的总和
            score_sum = sum(top_scores)
            # 判断是否大于等于 eta * lamada
            prediction = 1 if score_sum >= eta * lamada else 0
            # 真实标签
            ground_truth = int(label)
            
            # 添加到列表中
            y_true.append(ground_truth)
            y_pred.append(prediction)
    
    return y_true, y_pred

def main(args):
    data = torch.load(args.data_dir)

    # excluded_keys=["71","82","83","81","84","199","114","127","128","126","122","140","181","183","184"]
    # 计算预测结果和真实标签
    y_true, y_pred = calculate_predictions(data, args.lamada, args.eta)
    # 生成分类报告
    report = classification_report(y_true, y_pred)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None) 
    # parser.add_argument("--train",type=bool, required=False, default=True)
    parser.add_argument("--lamada", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    args = parser.parse_args()
    
    main(args)