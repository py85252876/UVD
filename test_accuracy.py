from sklearn.metrics import classification_report
import torch
import argparse
from collections import defaultdict
import random

def default_list_dict():
    return defaultdict(list)


def calculate_predictions(data, set_lambda, eta, excluded_keys=[],min_samples=None,seed=20240603):
    y_true = []
    y_pred = []
    random.seed(seed)
    if min_samples is None:
        min_samples = min(len(videos) for videos in data.values())
    
    for label, videos in data.items():
        if int(label) == 1:
            indices = [key for key in videos.keys() if key not in excluded_keys]
        else:
            indices = [key for key in videos.keys()]
        random.shuffle(indices)
        selected_indices = indices[:min_samples]
        
        for index in selected_indices:
            scores = videos[index]
            top_scores = scores[:int(eta)]
            score_sum = sum(top_scores)
            prediction = 1 if score_sum >= eta * set_lambda else 0
            ground_truth = int(label)
            y_true.append(ground_truth)
            y_pred.append(prediction)
    
    return y_true, y_pred

def main(args):
    data = torch.load(args.data_dir)
    y_true, y_pred = calculate_predictions(data, args.set_lambda, args.eta)
    report = classification_report(y_true, y_pred)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None) 
    parser.add_argument("--set_lambda", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    args = parser.parse_args()
    
    main(args)