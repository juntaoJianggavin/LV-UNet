# ==== imports ====
import argparse
import os
import random          # NEW
import numpy as np     # NEW
from collections import OrderedDict
from glob import glob
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose  # removed OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize, Flip
import losses
from dataset import Dataset
from metrics import iou_score
import utils
from utils import AverageMeter, str2bool
from LV_UNet import LV_UNet
LOSS_NAMES = list(getattr(losses, '__all__', []))  # safer
LOSS_NAMES.append('BCEWithLogitsLoss')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=None, help='run name')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=16, type=int)

    # model
    parser.add_argument('--deep_training', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int)
    parser.add_argument('--input_w', default=256, type=int)
    parser.add_argument('--input_h', default=256, type=int)

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss', choices=LOSS_NAMES)

    # dataset
    parser.add_argument('--dataset', default='isic')
    parser.add_argument('--img_ext', default='.png')
    parser.add_argument('--mask_ext', default='.png')

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['AdamW', 'Adam', 'SGD'])
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float, dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--nesterov', default=False, type=str2bool)

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int)
    parser.add_argument('--cfg', type=str, metavar="FILE")
    parser.add_argument('--num_workers', default=4, type=int)
    return parser.parse_args()

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    # make dataloader workers deterministic
    worker_seed = 42
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()
        target = target.cuda()


        output = model(input)
        loss = criterion(output, target)
        iou,dice = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()
            output = model(input)
            loss = criterion(output, target)
            iou,dice = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg)])

def main():
    seed_everything(42)
    args = parse_args()
    config = vars(args)

    # simple, safe name
    if config['name'] is None:
        config['name'] = f"{config['dataset']}_LV_UNet"

    os.makedirs(f"models/{config['name']}", exist_ok=True)

    print('-' * 20)
    for k, v in config.items():
        print(f"{k}: {v}")
    print('-' * 20)
    with open(f"models/{config['name']}/config.yml", 'w') as f:
        yaml.dump(config, f)

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # loss
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().to(device)
    else:
        criterion = losses.__dict__[config['loss']]().to(device)

    # model
    model = LV_UNet().to(device)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])

    # scheduler
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'],
                                                   patience=config['patience'], verbose=True, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                        milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    else:
        scheduler = None

    # data
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, test_img_ids = train_test_split(img_ids, test_size=0.2, random_state=42)
    train_img_ids, val_img_ids  = train_test_split(train_img_ids, test_size=0.2, random_state=42)

    train_transform = Compose([RandomRotate90(), Flip(),
                               Resize(config['input_h'], config['input_w']),
                               transforms.Normalize(),])
    val_transform = Compose([Resize(config['input_h'], config['input_w']),
                             transforms.Normalize(),])

    train_dataset = Dataset(train_img_ids,
                            img_dir=os.path.join('inputs', config['dataset'], 'images'),
                            mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
                            img_ext=config['img_ext'], mask_ext=config['mask_ext'],
                            transform=train_transform)
    val_dataset = Dataset(val_img_ids,
                          img_dir=os.path.join('inputs', config['dataset'], 'images'),
                          mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
                          img_ext=config['img_ext'], mask_ext=config['mask_ext'],
                          transform=val_transform)

    g = torch.Generator()
    g.manual_seed(42)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=config['batch_size'], shuffle=True,
                        num_workers=config['num_workers'], worker_init_fn=seed_worker,
                        generator=g, drop_last=True, pin_memory=(device=='cuda'))
    val_loader = torch.utils.data.DataLoader(val_dataset,
                        batch_size=config['batch_size'], shuffle=False,
                        num_workers=config['num_workers'], worker_init_fn=seed_worker,
                        generator=g, drop_last=False, pin_memory=(device=='cuda'))

    log = OrderedDict([('epoch', []), ('lr', []), ('loss', []), ('iou', []),
                       ('val_loss', []), ('val_iou', []), ('val_dice', [])])

    best_iou = 0.0   # FIX
    trigger = 0

    import math
    for epoch in range(config['epochs']):
        print(f"Epoch [{epoch}/{config['epochs']}]")

        if config['deep_training']:
            act_learn = 1 - math.cos(math.pi/2 * epoch / config['epochs'])
            try:
                model.change_act(act_learn)
            except AttributeError:
                pass 

        # train / val
        train_log = train(config, train_loader, model, criterion, optimizer)
        val_log   = validate(config, val_loader, model, criterion)

        # step scheduler
        if scheduler is not None:
            if config['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_log['loss'])
            else:
                scheduler.step()

        cur_lr = optimizer.param_groups[0]['lr'] 
        print(f"lr {cur_lr:.6f} - loss {train_log['loss']:.4f} - iou {train_log['iou']:.4f} - "
              f"val_loss {val_log['loss']:.4f} - val_iou {val_log['iou']:.4f}")

        # log
        log['epoch'].append(epoch)
        log['lr'].append(cur_lr)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice'])
        pd.DataFrame(log).to_csv(f"models/{config['name']}/log.csv", index=False)

        trigger += 1
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), f"models/{config['name']}/best_model.pth")
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        torch.save(model.state_dict(), f"models/{config['name']}/last_model.pth")

        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
