import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset


def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model_mask = model_cls(
        num_classes=3
    )
    model_gender = model_cls(
        num_classes=2
    )
    model_age = model_cls(
        num_classes=3
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_mask_path = os.path.join(saved_model, 'best_mask.pth')
    model_mask.load_state_dict(torch.load(model_mask_path, map_location=device))
    
    model_gender_path = os.path.join(saved_model, 'best_gender.pth')
    model_gender.load_state_dict(torch.load(model_gender_path, map_location=device))
    
    model_age_path = os.path.join(saved_model, 'best_age.pth')
    model_age.load_state_dict(torch.load(model_age_path, map_location=device))

    return model_mask, model_gender, model_age


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model_mask, model_gender, model_age = load_model(model_dir, num_classes, device)
    
    model_mask.to(device)
    model_gender.to(device)
    model_age.to(device)
    
    model_mask.eval()
    model_gender.eval()
    model_age.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred_mask = model_mask(images)
            pred_mask = pred_mask.argmax(dim=-1)
            
            pred_gender = model_gender(images)
            pred_gender = pred_gender.argmax(dim=-1)
            
            pred_age = model_age(images)
            pred_age = pred_age.argmax(dim=-1)
            
            pred = pred_mask *6 + pred_gender *2 + pred_age
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(128, 96), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='ResNet50', help='model type (default: BaseModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
