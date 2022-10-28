import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer_list):
    tmp = []
    for optimizer in optimizer_list:
        for param_group in optimizer.param_groups:
             tmp.append(param_group['lr'])
    return tmp


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

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
    dataset.set_transform(transform, transform, transform) # mask, gender, age model에 적용될 transform

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_list = torch.nn.ModuleList()
    for model_name, num_classes in zip(args.model,[3,2,3]):
        model = getattr(import_module("model"), model_name)(num_classes=num_classes).to(device)
        model = torch.nn.DataParallel(model)
        model_list.append(model)
        
    
    # -- loss & metric
    criterion_list = []
    for criterion_name in args.criterion:
        criterion_list.append(create_criterion(criterion_name))  # default: cross_entropy
        
    lr_list = [learning_rate for learning_rate in args.lr] # default: 1e-3
    
    optimizer_list = []
    for idx, optimizer_name in enumerate(args.optimizer): # default: Adam
        optimizer = getattr(import_module("torch.optim"), optimizer_name)(
            filter(lambda p: p.requires_grad, model_list[idx].parameters()),
            lr=lr_list[idx],
            weight_decay=5e-4
            )
        optimizer_list.append(optimizer)
    
    scheduler_list = []
    for lr_decay_step in args.lr_decay_step:
        if lr_decay_step == 0:
            scheduler_list.append(None)
        else:
            scheduler_list.append(StepLR(optimizer, args.lr_decay_step, gamma=0.5))
             

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_acc_mask = 0
    best_val_acc_gender = 0
    best_val_acc_age = 0
    
    for epoch in range(args.epochs):
        # train loop
        for i in range(3): model_list[i].train()
        loss_value = np.array([0.,0.,0.])
        matches = 0
        matches_list = np.array([0,0,0])
        preds = []
        
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs_list = list(inputs)
            for i in range(len(inputs_list)): inputs_list[i] = inputs_list[i].to(device)
            labels_list = list(labels)
            for i in range(len(labels_list)): labels_list[i] = labels_list[i].to(device)

            for i in range(3):
                optimizer_list[i].zero_grad()
                outs = model_list[i](inputs_list[i])
                preds.append(torch.argmax(outs,dim=-1))
                loss = criterion_list[i](outs,labels_list[i])
                
                loss.backward()
                optimizer_list[i].step()
                loss_value[i] += loss.item()
                matches_list[i] += (preds[i] == labels_list[i]).sum().item()
            
            matches += (preds[0]*6 + preds[1]*3 + preds[2] == labels_list[3]).sum().item()
            
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc_all = matches / args.batch_size / args.log_interval
                train_acc = matches_list / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer_list)
                if args.verbose == 1:
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss || mask:{train_loss[0]:4.4} || gender:{train_loss[1]:4.4} || age:{train_loss[2]:4.4}\n"
                        f" training accuracy || all:{train_acc_all:4.4%} || mask:{train_acc[0]:4.4%} || gender:{train_acc[1]:4.4%} || age:{train_acc[2]:4.4%} || lr {current_lr}"
                    )
                    print()
                else:
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || training loss {train_loss[0]:4.4} {train_loss[1]:4.4} {train_loss[2]:4.4} ||"
                        f" training accuracy {train_acc_all:4.4%} {train_acc[0]:4.4%} {train_acc[1]:4.4%} {train_acc[2]:4.4%}"
                    )
                    print()
                logger.add_scalar("Train/loss_mask", train_loss[0], epoch * len(train_loader) + idx)
                logger.add_scalar("Train/loss_gender", train_loss[1], epoch * len(train_loader) + idx)
                logger.add_scalar("Train/loss_age", train_loss[2], epoch * len(train_loader) + idx)
                
                logger.add_scalar("Train/accuracy_all", train_acc_all, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy_mask", train_acc[0], epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy_gender", train_acc[1], epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc[2], epoch * len(train_loader) + idx)
                
                loss_value = np.array([0.,0.,0.])
                matches = 0
                matches_list = np.array([0,0,0])
            preds= []
        
            
        
        
        for scheduler in scheduler_list:
            if scheduler == None:
                pass
            else:
                scheduler.step()
        
        
        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            for i in range(3): model_list[i].eval()
            preds = []
            loss_item = np.array([0.,0.,0.])
            acc_item = np.array([0.,0.,0.])
            acc_item_all = 0
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs_list = list(inputs)
                for idx in range(len(inputs_list)): inputs_list[idx] = inputs_list[idx].to(device)
                labels_list = list(labels)
                for idx in range(len(labels_list)): labels_list[idx] = labels_list[idx].to(device)

                for idx in range(3):
                    outs = model_list[idx](inputs_list[idx])
                    preds.append(torch.argmax(outs,dim=-1))
                    loss_item[idx] += criterion_list[idx](outs, labels_list[idx]).item()
                    acc_item[idx] += (preds[idx] == labels_list[idx]).sum().item()
                    
                acc_item_all += (preds[0]*6 + preds[1]*3 + preds[2] == labels_list[3]).sum().item()

                if figure is None:
                    inputs_np = torch.clone(inputs_list[0]).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels_list[3], preds[0]*6 + preds[1]*3 + preds[2], n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
                preds = []

            val_loss = loss_item / len(val_loader)
            val_acc = acc_item / len(val_set)
            val_acc_all = acc_item_all / len(val_set)
            
            if args.version == 1:
                if val_acc_all > best_val_acc:
                    print(f"New best model for val accuracy : ALL:{val_acc_all:4.2%} Mask: {val_acc[0]:4.2%} Gender: {val_acc[1]:4.2%} Age: {val_acc[2]:4.2%}! saving the best model..")
                    torch.save(model_list[0].module.state_dict(), f"{save_dir}/best_mask.pth")
                    torch.save(model_list[1].module.state_dict(), f"{save_dir}/best_gender.pth")
                    torch.save(model_list[2].module.state_dict(), f"{save_dir}/best_age.pth")
                    best_val_acc = val_acc_all
            else:
                if val_acc[0] > best_val_acc_mask:
                    print(f"New best mask model for val accuracy : ALL:{val_acc_all:4.2%} Mask: {val_acc[0]:4.2%}! saving the best mask model..")
                    torch.save(model_list[0].module.state_dict(), f"{save_dir}/best_mask.pth")
                    best_val_acc_mask = val_acc[0]
                    
                if val_acc[1] > best_val_acc_gender:
                    print(f"New best gender model for val accuracy : ALL:{val_acc_all:4.2%} Gender: {val_acc[1]:4.2%}! saving the best gender model..")
                    torch.save(model_list[1].module.state_dict(), f"{save_dir}/best_gender.pth")
                    best_val_acc_mask = val_acc[1]
                
                if val_acc[1] > best_val_acc_age:
                    print(f"New best age model for val accuracy : ALL:{val_acc_all:4.2%} Age: {val_acc[2]:4.2%}! saving the best age model..")
                    torch.save(model_list[2].module.state_dict(), f"{save_dir}/best_age.pth")
                    best_val_acc_mask = val_acc[2]
            
            
            if args.verbose == 1:
                print(
                    f"[Val] acc : ALL:{val_acc_all:4.2%} Mask: {val_acc[0]:4.2%} Gender: {val_acc[1]:4.2%} Age: {val_acc[2]:4.2%}||\n"
                    f"[Val] loss : Mask: {val_loss[0]:4.2} Gender: {val_loss[1]:4.2} Age: {val_loss[2]:4.2}\n"
                    f"best acc : {best_val_acc:4.2%}"
                )
            else:
                print(
                    f"[Val] acc : {val_acc_all:4.2%} {val_acc[0]:4.2%} {val_acc[1]:4.2%} {val_acc[2]:4.2%} ||"
                    f"[Val] loss : {val_loss[0]:4.2}  {val_loss[1]:4.2} {val_loss[2]:4.2} ||"
                    f"best acc : {best_val_acc:4.2%}"
                )
                
            logger.add_scalar("Val/loss_mask", val_loss[0], epoch)
            logger.add_scalar("Val/loss_gender", val_loss[1], epoch)
            logger.add_scalar("Val/loss_age", val_loss[2], epoch)
                
            logger.add_scalar("Val/accuracy_all", val_acc_all, epoch)
            logger.add_scalar("Val/accuracy_mask", val_acc[0], epoch)
            logger.add_scalar("Val/accuracy_gender", val_acc[1], epoch)
            logger.add_scalar("Val/accuracy", val_acc[2], epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='Three_Body_MaskBaseDataset', help='dataset augmentation type (default: Three_Body_MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', nargs="+", type=str, default=['ResNet50','ResNet50','ResNet50'], help='model type (default: ResNet50)')
    parser.add_argument('--optimizer', nargs="+", type=str, default=['Adam','Adam','Adam'], help='optimizer type (default: Adam)')
    parser.add_argument('--lr',nargs="+",type=float, default=[1e-3,1e-3,1e-3], help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', nargs="+",type=str, default=['cross_entropy','cross_entropy','cross_entropy'], help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', nargs="+", type=int, default=[0,0,0], help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--version', type = int, default = 1, help = '1 : related best, 2: respectively best (default = 1)')
    parser.add_argument('--verbose', type = int, default = 1, help ='1 : verbose, 2: one-line (default: 1)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)