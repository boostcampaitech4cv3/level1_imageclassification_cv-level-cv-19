import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import argparse
import glob
import json
import multiprocessing
import os

import random
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset, LabelSplitDataset
from loss import create_criterion
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def load_model(saved_model, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls()

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

@torch.no_grad()
def get_result(data_dir, model_dir, args):
    seed_everything(args.seed)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    _ , val_set = dataset.split_dataset()

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, device).to(device)
    model.eval()

    
    print("Calculating inference results..")
    mask_preds = []
    gender_preds = []
    age_preds = []
    
    mask_label = []
    gender_label = []
    age_label = []
    with torch.no_grad():
        for val_batch in val_loader:
            images, mask_labels, gender_labels, age_labels = val_batch
            
            images = images.to(device)
            mask, gender, age = model(images) # shape: B * 3
            pred_mask = mask.argmax(dim=-1) # shape: B * 1
            pred_gender = gender.argmax(dim=-1) # shape: B * 1
            pred_age = age.argmax(dim=-1) # shape: B * 1

            mask_preds.extend(pred_mask.cpu().numpy())
            gender_preds.extend(pred_gender.cpu().numpy())
            age_preds.extend(pred_age.cpu().numpy())
            mask_label.extend(mask_labels.cpu().numpy())
            gender_label.extend(gender_labels.cpu().numpy())
            age_label.extend(age_labels.cpu().numpy())

    return mask_preds, mask_label, gender_preds, gender_label, age_preds, age_label


def visualize(target1, pred1, target2, pred2, target3, pred3, out_dir):
    cf_matrix1 = confusion_matrix(target1, pred1)
    cf_matrix2 = confusion_matrix(target2, pred2)
    cf_matrix3 = confusion_matrix(target3, pred3)

    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    sns.heatmap(cf_matrix1, linewidths=1, annot=True, ax=ax[0], fmt='g')
    sns.heatmap(cf_matrix2, linewidths=1, annot=True, ax=ax[1], fmt='g')
    sns.heatmap(cf_matrix3, linewidths=1, annot=True, ax=ax[2], fmt='g')
    
    for i, title in enumerate(['mask','gender','age']):
        # ax[i].axes.xaxis.set_ticks([])
        # ax[i].axes.yaxis.set_ticks([])
        ax[i].axes.set_title(title)
        ax[i].axes.set_xlabel('Predicted labels')
        ax[i].axes.set_ylabel('True labels')
    
    ax[0].axes.xaxis.set_ticklabels(['Wear', 'Incorrect', 'Not Wear'])
    ax[0].axes.yaxis.set_ticklabels(['Wear', 'Incorrect', 'Not Wear'])
    
    ax[1].axes.xaxis.set_ticklabels(['Male', 'Female'])
    ax[1].axes.yaxis.set_ticklabels(['Male', 'Female'])

    ax[2].axes.xaxis.set_ticklabels(['<30', '>=30 and < 60', '>=60'])
    ax[2].axes.yaxis.set_ticklabels(['<30', '>=30 and < 60', '>=60'])
 
    fig.savefig(out_dir+"/"+args.name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--dataset', type=str, default='LabelSplitDataset', help='dataset augmentation type (default: LabelSplitDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--name', default='Result', help='model save at output_dir/{name}')
    
    # Data and model checkpoints directories
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './vis_output'))

    args = parser.parse_args()
    print(args)
    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    mask_preds, mask_label, gender_preds, gender_label, age_preds, age_label = get_result(data_dir, model_dir, args)
    visualize(mask_preds, mask_label, gender_preds, gender_label, age_preds, age_label, output_dir)
