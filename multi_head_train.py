import argparse
import glob
import json
import multiprocessing
import os
import random
import re
import numpy as np
import warnings

from importlib import import_module
from pathlib import Path

import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import ConfusionMatrix, F1Score
from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
import matplotlib.pyplot as plt

from scheduler import scheduler_module
from dataset import MaskBaseDataset, LabelSplitDataset
from loss import create_criterion

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 경고 off
warnings.filterwarnings(action='ignore')

# 재현성
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# tensorboard에 올리는 이미지 grid 생성
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

# 자동 경로 추가
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

def encode_multi_class(mask_label, gender_label, age_label) -> int:
    return mask_label * 6 + gender_label * 3 + age_label

# cutmix
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# confusion matrix
def plot_confusion_matrix(confusion_matrix,confusion_matrix_mask, confusion_matrix_gender, confusion_matrix_age, dir_path):
    fig_all, ax = plt.subplots(figsize=(15, 15))
    sns.heatmap(confusion_matrix, linewidths=1, annot=True, ax=ax, fmt='g', cmap= "Blues", cbar = False)
    ax.axes.set_xlabel('Predicted labels')
    ax.axes.set_ylabel('True labels')
    tmp = []
    for i in range(3):
        for j in range(2):
            for k in range(3):
                tmp.append(f'[{i},{j},{k}]')

    ax.axes.xaxis.set_ticklabels([*tmp])
    ax.axes.yaxis.set_ticklabels([*tmp])
    fig_all.savefig(dir_path+"/18_class_confusion_matrix.png")
    
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
    sns.heatmap(confusion_matrix_mask, linewidths=1, annot=True, ax=ax[0], fmt='g', cmap= "Blues", cbar = False)
    sns.heatmap(confusion_matrix_gender, linewidths=1, annot=True, ax=ax[1], fmt='g', cmap= "Blues", cbar = False)
    sns.heatmap(confusion_matrix_age, linewidths=1, annot=True, ax=ax[2], fmt='g', cmap= "Blues", cbar = False)
    
    for i, title in enumerate(['mask','gender','age']):
        ax[i].axes.set_title(title)
        ax[i].axes.set_xlabel('Predicted labels')
        ax[i].axes.set_ylabel('True labels')
    
    ax[0].axes.xaxis.set_ticklabels(['Wear', 'Incorrect', 'Not Wear'])
    ax[0].axes.yaxis.set_ticklabels(['Wear', 'Incorrect', 'Not Wear'])
    
    ax[1].axes.xaxis.set_ticklabels(['Male', 'Female'])
    ax[1].axes.yaxis.set_ticklabels(['Male', 'Female'])

    ax[2].axes.xaxis.set_ticklabels(['<30', '>=30 and < 60', '>=60'])
    ax[2].axes.yaxis.set_ticklabels(['<30', '>=30 and < 60', '>=60'])
    
    fig.savefig(dir_path+"/sep_class_confusion_matrix.png")  
    
    return fig_all, fig

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
    num_classes = dataset.num_classes
    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    if args.sampler == "None":
        sampler_flag = (True, None)
    else:
        sampler_module = getattr(import_module("sampler"), args.sampler)
        sampler_flag = (False, sampler_module(train_set, labels =  dataset.get_multi_labels())())

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=sampler_flag[0],
        pin_memory=use_cuda,
        drop_last=True,
        sampler= sampler_flag[1]
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
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module().to(device)
    model = torch.nn.DataParallel(model)

    # -- model freeze
    # model.requires_grad_(False)
    # for param, weight in model.named_parameters():
    #     # print(param)
    #     if param in ['module.backbone.head.weight', 'module.backbone.head.bias']:
    #         weight.requires_grad = True
        

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )

    # LR scheduler
    if args.scheduler != "None":
        scheduler = scheduler_module.get_scheduler(scheduler_module,args.scheduler, optimizer)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    best_f1_score = 0
    
    for epoch in range(args.epochs):
        # -- model freeze
        # if epoch > 30:
        #     model.requires_grad_(True)

        # train loop
        model.train()
        loss_value = 0
        matches = 0

        for idx, train_batch in enumerate(train_loader):
            inputs, mask_labels, gender_labels, age_labels = train_batch
            inputs = inputs.to(device) #(B, C, 320, 256)

            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)

            r = np.random.rand(1)

            # for CutMix
            if args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                
                mask_labels_a = mask_labels
                mask_labels_b = mask_labels[rand_index]
                
                gender_labels_a = gender_labels
                gender_labels_b = gender_labels[rand_index]

                age_labels_a = age_labels
                age_labels_b = age_labels[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                
                # compute output
                mask, gender, age = model(inputs)

                mask_loss = criterion(mask, mask_labels_a) * lam + criterion(mask, mask_labels_b) * (1. - lam)
                gender_loss = criterion(gender, gender_labels_a) * lam + criterion(gender, gender_labels_b) * (1. - lam)
                age_loss = criterion(age, age_labels_a) * lam + criterion(age, age_labels_b) * (1. - lam)

            else:
                mask, gender, age = model(inputs)
                mask_loss = criterion(mask, mask_labels)
                gender_loss = criterion(gender, gender_labels)
                age_loss = criterion(age, age_labels)

            optimizer.zero_grad()
            mask_loss.backward(retain_graph=True)
            gender_loss.backward(retain_graph=True)
            age_loss.backward()

            preds_mask = torch.argmax(mask, dim=-1).float()
            preds_gender = torch.argmax(gender, dim=-1).float()
            preds_age = torch.argmax(age, dim=-1).float()

            preds = encode_multi_class(preds_mask, preds_gender, preds_age)    
            labels = encode_multi_class(mask_labels, gender_labels, age_labels)
                        
            optimizer.step()
            loss_value += (mask_loss.item() + gender_loss.item() + age_loss.item()) / 3 
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        if int(args.lr_decay_step) == 0:
            pass
        else:
            scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None

            confusion_matrix  = torch.Tensor([[0]])
            confusion_matrix_mask = torch.Tensor([[0]])
            confusion_matrix_gender = torch.Tensor([[0]])
            confusion_matrix_age = torch.Tensor([[0]])
            
            preds_expand = torch.tensor([])
            labels_expand = torch.tensor([])
            
            for val_batch in val_loader:
                inputs, mask_labels, gender_labels, age_labels = val_batch
                
                inputs = inputs.to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)

                r = np.random.rand(1)
                mask, gender, age = model(inputs)
                mask_loss = criterion(mask, mask_labels)
                gender_loss = criterion(gender, gender_labels)
                age_loss = criterion(age, age_labels)
                
                preds_mask = torch.argmax(mask, dim=-1)
                preds_gender = torch.argmax(gender, dim=-1)
                preds_age = torch.argmax(age, dim=-1)
                
                preds = encode_multi_class(preds_mask, preds_gender, preds_age)    
                labels = encode_multi_class(mask_labels, gender_labels, age_labels)
                
                loss_item = (mask_loss.item() + gender_loss.item() + age_loss.item()) / 3
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                figure = grid_image(
                    inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                )

                confmat = ConfusionMatrix(num_classes = 18).to(device)
                confusion_matrix = confmat(preds,labels).detach().cpu() + confusion_matrix
                confmat = ConfusionMatrix(num_classes = 3).to(device)
                confusion_matrix_mask = confmat(preds_mask,mask_labels).detach().cpu() + confusion_matrix_mask
                confmat = ConfusionMatrix(num_classes = 2).to(device)
                confusion_matrix_gender = confmat(preds_gender,gender_labels).detach().cpu() + confusion_matrix_gender
                confmat = ConfusionMatrix(num_classes = 3).to(device)
                confusion_matrix_age = confmat(preds_age,age_labels).detach().cpu() + confusion_matrix_age
                
                preds_expand = torch.cat((preds_expand, preds.detach().cpu()),-1)
                labels_expand = torch.cat((labels_expand, labels.detach().cpu()),-1)
                
            confusion_all_fig, confusion_sep_fig = plot_confusion_matrix(confusion_matrix,confusion_matrix_mask, confusion_matrix_gender, confusion_matrix_age , save_dir)    
            
            f1 = F1Score(num_classes=num_classes)
            f1_score = f1(preds_expand.type(torch.LongTensor), labels_expand.type(torch.LongTensor)).item()

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)

            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2} || "
                f"best f1 score : {best_f1_score:4.2%}, f1 score: {f1_score:4.2}"
            )

            flag = True
            if val_acc > best_val_acc:
                print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best_acc.pth")
                best_val_acc = val_acc
                early_stopping = args.patient
                flag = False
                
            if f1_score > best_f1_score:
                print(f"New best model for f1 score : {f1_score:4.4}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best_f1.pth")
                best_f1_score = f1_score
                early_stopping = args.patient
                flag = False
                
            if flag:
                early_stopping = early_stopping -1
                print(f"patient_left: {early_stopping}")
                if early_stopping == 0:
                    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")                    
                    print("early_stopping, save last model as last.pth")
                    break

            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("early_stopping_count", early_stopping, epoch)
            logger.add_scalar("Val/f1_score", f1_score, epoch)
            logger.add_figure("results", figure, epoch)
            logger.add_figure("val_confusion_matrix_all",confusion_all_fig, epoch)
            logger.add_figure("val_confusion_matrix_sep",confusion_sep_fig, epoch)
        if args.scheduler != "None":
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss) # ReduceLROnPlateau는 추적할 metric을 넣어서 step을 수행한다
            else:
                scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patient', type=int, default = 15, help='early stopping patient(default: 15)')
    parser.add_argument('--cutmix_prob', type=float, default=0, help='cutmix probability')
    parser.add_argument('--beta', default=0, type=float, help='hyperparameter beta')
    parser.add_argument('--sampler', type=str, default='None', help='sampler for imblanced data (default:None), samplers in sampler.py')
    parser.add_argument('--scheduler', default='StepLR', type=str, help='scheduler(default:StepLR')
 # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)