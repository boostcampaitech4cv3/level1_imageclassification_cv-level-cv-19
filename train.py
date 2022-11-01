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

from scheduler import scheduler_module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset, MaskLabels, GenderLabels, AgeLabels
from loss import create_criterion
from torchmetrics import ConfusionMatrix, F1Score
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# 경고 off
import warnings
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
def grid_image(np_images, gts, preds, n=16, shuffle=False, fig_size = (12,20)):
    batch_size = np_images.shape[0]
    #assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=fig_size)  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    #plt.subplots_adjust(left = 0.1, right = 0.9, bottom = 0.3, top = 1.7)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)
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
def plot_confusion_matrix(confusion_matrix,confusion_matrix_mask, confusion_matrix_gender, confusion_matrix_age):
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
    
      
    
    return fig_all, fig

def wrong_fig_viz(inputs, labels, preds,
                  labels_mask, preds_mask,
                  labels_gender, preds_gender,
                  labels_age, preds_age,
                  inputs_np_wrong,
                  inputs_np_wrong_mask, inputs_np_wrong_gender, inputs_np_wrong_age,
                  labels_wrong, preds_wrong,
                  labels_wrong_mask, preds_wrong_mask,
                  labels_wrong_gender, preds_wrong_gender,
                  labels_wrong_age, preds_wrong_age
                  ):
    # wrong_list = (labels == preds).detach().cpu()
    # wrong_list = np.array([not boo for boo in wrong_list])
    # if inputs_np_wrong is None:
    #     inputs_np_wrong = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()[wrong_list]
    # else:
    #     inputs_np_wrong = np.concatenate((inputs_np_wrong,torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()[wrong_list]), axis = 0)
                    
    # labels_wrong = torch.cat((labels_wrong,labels[wrong_list].detach().cpu()), -1)
    # preds_wrong = torch.cat((preds_wrong,preds[wrong_list].detach().cpu()), -1)
    
    
    
    wrong_list_mask = (labels_mask == preds_mask).detach().cpu()
    wrong_list_mask = np.array([not boo for boo in wrong_list_mask])
    if inputs_np_wrong_mask is None:
        inputs_np_wrong_mask = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()[wrong_list_mask]
    else:
        inputs_np_wrong_mask = np.concatenate((inputs_np_wrong_mask,torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()[wrong_list_mask]), axis = 0)
                    
    labels_wrong_mask = torch.cat((labels_wrong_mask,labels[wrong_list_mask].detach().cpu()), -1)
    preds_wrong_mask = torch.cat((preds_wrong_mask,preds[wrong_list_mask].detach().cpu()), -1)
    
    
    wrong_list_gender = (labels_gender == preds_gender).detach().cpu()
    wrong_list_gender = np.array([not boo for boo in wrong_list_gender])
    if inputs_np_wrong_gender is None:
        inputs_np_wrong_gender = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()[wrong_list_gender]
    else:
        inputs_np_wrong_gender = np.concatenate((inputs_np_wrong_gender,torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()[wrong_list_gender]), axis = 0)
                    
    labels_wrong_gender = torch.cat((labels_wrong_gender,labels[wrong_list_gender].detach().cpu()), -1)
    preds_wrong_gender = torch.cat((preds_wrong_gender,preds[wrong_list_gender].detach().cpu()), -1)
    
    
    wrong_list_age = (labels_age == preds_age).detach().cpu()
    wrong_list_age = np.array([not boo for boo in wrong_list_age])
    if inputs_np_wrong_age is None:
        inputs_np_wrong_age = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()[wrong_list_age]
    else:
        inputs_np_wrong_age = np.concatenate((inputs_np_wrong_age,torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()[wrong_list_age]), axis = 0)
                    
    labels_wrong_age = torch.cat((labels_wrong_age,labels[wrong_list_age].detach().cpu()), -1)
    preds_wrong_age = torch.cat((preds_wrong_age,preds[wrong_list_age].detach().cpu()), -1)

    return (inputs_np_wrong, labels_wrong, preds_wrong), (inputs_np_wrong_mask, labels_wrong_mask, preds_wrong_mask),\
        (inputs_np_wrong_gender, labels_wrong_gender, preds_wrong_gender),(inputs_np_wrong_age, labels_wrong_age, preds_wrong_age)



# -- train loop
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
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader & sampler
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
        drop_last=False,
        sampler= sampler_flag[1],
    )


    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    
    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: ResNet50
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # # -- model freeze
    # model.requires_grad_(False)
    # for param, weight in model.named_parameters():
    #     # print(param)
    #     if param in ['module.backbone.head.weight', 'module.backbone.head.bias']:
    #         weight.requires_grad = True
            
    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    if args.optimizer == "AdamP":
        opt_module = getattr(import_module("adamp"), args.optimizer)
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        model.parameters(),
        lr=args.lr,
        weight_decay=5e-4
    )
    
    

    if args.scheduler != "None":
        scheduler = scheduler_module.get_scheduler(scheduler_module,args.scheduler, optimizer)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)
    layout = {
        "Train_Val": {
            "accuracy": ["Multiline", ["Train/accuracy_epcoh", "Val/accuracy"]], 
            "f1_score": ["Multiline", ["Val/f1_score"]], 
            "loss": ["Multiline", ['Train/loss_epoch', 'Val/loss']]
        }, 
        "3_Task": {
            "accuracy": ["Multiline", ['task/acc/mask', 'task/acc/gender', 'task/acc/age']], 
            "f1_score": ["Multiline", ['task/f1/mask', 'task/f1/gender', 'task/f1/age']]
        }, 
        "Each_Class": {
            "mask_f1_score":["Multiline", ["class/f1/MASK", 'class/f1/INCORRECT', 'class/f1/NORMAL']], 
            "gender_f1_score":["Multiline", ['class/f1/MALE', 'class/f1/FEMALE']],
            "age_f1_score":["Multiline", ['class/f1/YOUNG', 'class/f1/MIDDLE', 'class/f1/OLD']],
        }
    }
    logger.add_custom_scalars(layout)

    best_val_acc = 0
    best_val_loss = np.inf
    best_f1_score = 0
    
    early_stopping = args.patient
    
    for epoch in tqdm(range(args.epochs)):

        # train loop
        model.train()
        loss_value = 0
        matches = 0
        loss_value_sum = 0
        train_acc_sum = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            r = np.random.rand(1)

            # for CutMix
            if args.beta > 0 and r < args.cutmix_prob:
                # generate mixed sample
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(inputs.size()[0]).cuda()
                
                labels_a = labels
                labels_b = labels[rand_index]
                
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                
                # compute output
                outs = model(inputs)

                loss = criterion(outs, labels_a) * lam + criterion(outs, labels_b) * (1. - lam)
                
            else:
                outs = model(inputs)
                loss = criterion(outs, labels)
                
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

            loss_value += loss.item()

            preds = torch.argmax(outs, dim=-1)
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
                logger.add_scalar("lr", current_lr, epoch)

                loss_value_sum += train_loss
                train_acc_sum += train_acc
                loss_value = 0
                matches = 0
        
        loss_value_sum /= (len(train_loader)//args.log_interval)
        train_acc_sum /= (len(train_loader)//args.log_interval)
        logger.add_scalar("Train/loss_epoch", loss_value_sum, epoch)
        logger.add_scalar("Train/accuracy_epcoh", train_acc_sum, epoch)
        
        


        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            
            inputs_np_wrong = None;labels_wrong = torch.Tensor([]);preds_wrong = torch.Tensor([])
            inputs_np_wrong_mask = None;labels_wrong_mask = torch.Tensor([]);preds_wrong_mask = torch.Tensor([])
            inputs_np_wrong_gender = None;labels_wrong_gender = torch.Tensor([]);preds_wrong_gender = torch.Tensor([])
            inputs_np_wrong_age = None;labels_wrong_age = torch.Tensor([]);preds_wrong_age = torch.Tensor([])
            wrong_flag = epoch in args.wrong_fig
            
            confusion_matrix  = torch.Tensor([[0]])
            confusion_matrix_mask = torch.Tensor([[0]])
            confusion_matrix_gender = torch.Tensor([[0]])
            confusion_matrix_age = torch.Tensor([[0]])
            
            preds_expand = torch.tensor([])
            labels_expand = torch.tensor([])
           
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outs = model(inputs)
                loss = criterion(outs, labels)

                # -- calculate metrics(loss, acc, f1) & confusion matrix & visualize wrong figure
                preds = torch.argmax(outs, dim=-1)
                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None and epoch == 0:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )
                    logger.add_figure("results", figure, epoch)
                
                preds_mask, preds_gender, preds_age = MaskBaseDataset.decode_multi_class(preds)
                labels_mask, labels_gender, labels_age = MaskBaseDataset.decode_multi_class(labels)
                
                
                
                if wrong_flag:
                    wrong_all, wrong_mask, wrong_gender, wrong_age = wrong_fig_viz(
                        inputs, labels, preds,
                        labels_mask, preds_mask, labels_gender, preds_gender, labels_age, preds_age,
                        inputs_np_wrong, inputs_np_wrong_mask, inputs_np_wrong_gender, inputs_np_wrong_age,
                        labels_wrong, preds_wrong, labels_wrong_mask, preds_wrong_mask,labels_wrong_gender, preds_wrong_gender,labels_wrong_age, preds_wrong_age
                    )
                    
                    # inputs_np_wrong, labels_wrong, preds_wrong = wrong_all
                    inputs_np_wrong_mask, labels_wrong_mask, preds_wrong_mask = wrong_mask
                    inputs_np_wrong_gender, labels_wrong_gender, preds_wrong_gender = wrong_gender
                    inputs_np_wrong_age, labels_wrong_age, preds_wrong_age = wrong_age
                    
                    
                                  
                    
                    
                              
                confmat = ConfusionMatrix(num_classes = 18).to(device)
                confusion_matrix = confmat(preds,labels).detach().cpu() + confusion_matrix
                confmat = ConfusionMatrix(num_classes = 3).to(device)
                confusion_matrix_mask = confmat(preds_mask,labels_mask).detach().cpu() + confusion_matrix_mask
                confmat = ConfusionMatrix(num_classes = 2).to(device)
                confusion_matrix_gender = confmat(preds_gender,labels_gender).detach().cpu() + confusion_matrix_gender
                confmat = ConfusionMatrix(num_classes = 3).to(device)
                confusion_matrix_age = confmat(preds_age,labels_age).detach().cpu() + confusion_matrix_age
                
                preds_expand = torch.cat((preds_expand, preds.detach().cpu()),-1)
                labels_expand = torch.cat((labels_expand, labels.detach().cpu()),-1)
            
                                 
                        
            confusion_all_fig, confusion_sep_fig = plot_confusion_matrix(confusion_matrix,confusion_matrix_mask, confusion_matrix_gender, confusion_matrix_age)    
            # logger.add_figure("val_confusion_matrix_all",confusion_all_fig, epoch)
            # logger.add_figure("val_confusion_matrix_sep",confusion_sep_fig, epoch)
            
            preds_mask, preds_gender, preds_age = MaskBaseDataset.decode_multi_class(preds_expand)
            labels_mask, labels_gender, labels_age = MaskBaseDataset.decode_multi_class(labels_expand)
            
            # -- evaluation functions
            f1 = MulticlassF1Score(num_classes=num_classes)
            f1_mask = MulticlassF1Score(num_classes=3)
            f1_gender = MulticlassF1Score(num_classes=2)
            f1_age = MulticlassF1Score(num_classes=3)
            f1_mask_cl = MulticlassF1Score(num_classes=3, average=None)
            f1_gender_cl = MulticlassF1Score(num_classes=2, average=None)
            f1_age_cl = MulticlassF1Score(num_classes=3, average=None)
            
            # -- evaluation values
            f1_score = f1(preds_expand.type(torch.LongTensor), labels_expand.type(torch.LongTensor)).item()
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            f1_score_mask = f1_mask(preds_mask, labels_mask)
            f1_score_gender = f1_gender(preds_gender, labels_gender)
            f1_score_age = f1_age(preds_age, labels_age)
            f1_score_mask_cl = f1_mask_cl(preds_mask, labels_mask)
            f1_score_gender_cl = f1_gender_cl(preds_gender, labels_gender)
            f1_score_age_cl = f1_age_cl(preds_age, labels_age)
            val_acc_mask = (labels_mask == preds_mask).sum().item() / len(val_set)
            val_acc_gender = (labels_gender == preds_gender).sum().item() / len(val_set)
            val_acc_age = (labels_age == preds_age).sum().item() / len(val_set)
            
            
            
            flag = True
            # if val_acc > best_val_acc:
            #     print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
            #     torch.save(model.module.state_dict(), f"{save_dir}/{epoch}best_acc{val_acc}{}.pth")
            #     if args.model_save:
            #         torch.save(model, f"{save_dir}/{epoch}best_acc{val_acc}.pt")
            #     best_val_acc = val_acc
            #     flag = False
                
            if f1_score > best_f1_score:
                print(f"New best model for f1 score : {f1_score:4.4}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/{epoch}_best_f1_{f1_score:4.3}_{val_acc:4.3}.pth")
                if args.model_save:
                    torch.save(model, f"{save_dir}/{epoch}best_f1{f1_score:4.4}_{val_acc:4.4}.pth")
                best_f1_score = f1_score
                best_val_acc = val_acc
                flag = False
            
            if flag == False:
                print("saving confusion matrix")
                confusion_all_fig.savefig(save_dir+"/best_18_class_confusion_matrix.png")
                confusion_sep_fig.savefig(save_dir+"/best_sep_class_confusion_matrix.png")
                
            if wrong_flag:
                print("saving wrong images")
                # all wrong images
                # inputs_np_wrong = dataset_module.denormalize_image(inputs_np_wrong, dataset.mean, dataset.std)
                # figure_wrong = grid_image(inputs_np_wrong, labels_wrong, preds_wrong, n= len(labels_wrong), fig_size = (40,40))
                # figure_wrong.savefig(save_dir+"/wrong_image.png")
                    
                inputs_np_wrong_mask = dataset_module.denormalize_image(inputs_np_wrong_mask, dataset.mean, dataset.std)
                figure_wrong_mask = grid_image(inputs_np_wrong_mask, labels_wrong_mask, preds_wrong_mask, n= len(labels_wrong_mask), fig_size = (64,48))
                figure_wrong_mask.savefig(save_dir+"/wrong_mask_image.png")
                    
                inputs_np_wrong_gender = dataset_module.denormalize_image(inputs_np_wrong_gender, dataset.mean, dataset.std)
                figure_wrong_gender = grid_image(inputs_np_wrong_gender, labels_wrong_gender, preds_wrong_gender, n= len(labels_wrong_gender), fig_size = (64,48))
                figure_wrong_gender.savefig(save_dir+"/wrong_gender_image.png")
                    
                inputs_np_wrong_age = dataset_module.denormalize_image(inputs_np_wrong_age, dataset.mean, dataset.std)
                figure_wrong_age = grid_image(inputs_np_wrong_age, labels_wrong_age, preds_wrong_age, n= len(labels_wrong_age), fig_size = (64,48))
                figure_wrong_age.savefig(save_dir+"/wrong_age_image.png")
                    
            # --early stopping
            if val_loss < best_val_loss:
                early_stopping = args.patient
                best_val_loss = val_loss
            else:
                early_stopping = early_stopping -1
                print(f"patient_left: {early_stopping}")
                if early_stopping == 0:                
                    print("early_stopping, save last model as last.pth")
                    break         
                        
                                    
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1: {f1_score:4.4}|| "
                f"best loss: {best_val_loss:4.2}, best f1: {best_f1_score:4.4}, acc for best f1 : {best_val_acc:4.2%}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("early_stopping_count", early_stopping, epoch)
            logger.add_scalar("Val/f1_score", f1_score, epoch)
            logger.add_scalar('task/f1/mask', f1_score_mask, epoch)
            logger.add_scalar('task/f1/gender', f1_score_gender, epoch)
            logger.add_scalar('task/f1/age', f1_score_age, epoch)
            for i,f in enumerate(f1_score_mask_cl):
                logger.add_scalar(f'class/f1/{MaskLabels(i).name}', f, epoch)
            for i,f in enumerate(f1_score_gender_cl):
                logger.add_scalar(f'class/f1/{GenderLabels(i).name}', f, epoch)
            for i,f in enumerate(f1_score_age_cl):
                logger.add_scalar(f'class/f1/{AgeLabels(i).name}', f, epoch)
            logger.add_scalar('task/acc/mask', val_acc_mask, epoch)
            logger.add_scalar('task/acc/gender', val_acc_gender, epoch)
            logger.add_scalar('task/acc/age', val_acc_age, epoch)
            
        
        
        # --scheduler
        if args.scheduler != "None":
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_loss) # ReduceLROnPlateau는 추적할 metric을 넣어서 step을 수행한다
            elif scheduler.__class__.__name__ == "CosineLRScheduler":
                scheduler.step(epoch)
            else:
                scheduler.step()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='EfficientNet_B2', help='model type (default: ResNet50)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--patient', type=int, default = 15, help='early stopping patient(default: 15)')
    parser.add_argument('--cutmix_prob', type=float, default=0, help='cutmix probability')
    parser.add_argument('--beta', type=float, default=0, help='hyperparameter beta')
    parser.add_argument('--sampler', type=str, default='None', help='sampler for imblanced data (default:None), samplers in sampler.py')
    parser.add_argument('--scheduler', type=str, default='None', help='scheduler(default:None), scheduler list in scheduler.py')
    parser.add_argument('--model_save',type=bool, default=False, help='save model architecture with state_dict')
    parser.add_argument('--wrong_fig',nargs="+",type=int, default=[-1], help='visualize wrong figures when args.wrong_fig == epoch (default:[-1])')
    
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)