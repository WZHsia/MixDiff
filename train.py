import argparse
from torch import optim
import os
import datetime
import utils
import pandas as pd
from collections import OrderedDict
from torchsummary import summary
from torchstat import stat
from model import *
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from utils import iou_score, BCEDiceLoss, ImageLoader
from tqdm import tqdm
import copy
from pathlib import Path
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser()

    # [(Unet, UCnet, UAnet, UDnet), (ULite, ULite_Mix), (ResUnet, ResUnet_Mix)
    # (Unet++, Unet++_Mix), (UNext, UNext_Mix), (UNext_S, UNext_S_Mix)]
    parser.add_argument('--name', default="UAnet",
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # model
    # [(Unet, UCnet, UAnet, UGnet), (ULite, ULite_Mix), (ResUnet, ResUnet_Mix)
    # (Unet++, Unet++_Mix), (UNext, UNext_Mix), (UNext_S, UNext_S_Mix)]
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UAnet')
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    parser.add_argument('--amp', default=True)

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss')

    # dataset
    parser.add_argument('--dataset', default='BUSI',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-3, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False,
                        help='nesterov')
    # scheduler
    parser.add_argument('--scheduler', default='ReduceLROnPlateau',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--mode', default='min')
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2 / 3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=4, type=int)

    config = parser.parse_args()

    return config


def train_net(
        net,
        device,
        data_path,
        optimizer,
        scheduler,
        los_path,
        criterion,
        dir_checkpoint,
        epochs=50,
        batch_size=5,
        save_checkpoint=True,
        amp: bool = True,
):
    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_iou', []),
        ('val_dice', []),
    ])
    dataset = ImageLoader(data_path)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_model, best_loss = 0, float('inf')
    for epoch in range(epochs):
        net.train()
        train_loss, train_iou, train_dice = 0, 0, 0
        train_num = len(train_loader)
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                # with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                pred = net(image)
                loss = criterion(pred, label)
                train_loss += loss.item()
                torch.autograd.set_detect_anomaly(True)
                loss.requires_grad_(True)
                grad_scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                # optimizer.step()
                iou, dice = iou_score(pred, label)
                train_iou += iou
                train_dice += dice
                pbar.update(image.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'iou (batch)': iou, 'dice (batch)': dice})

        with torch.no_grad():
            val_loss, val_iou, val_dice = evaluate(net, val_loader, n_val, device, amp)
        if best_loss > val_loss:
            best_model = copy.deepcopy(net.state_dict())
            best_loss = val_loss
        scheduler.step(loss)
        log['epoch'].append(epoch + 1)
        log['loss'].append(train_loss / train_num)
        log['iou'].append(train_iou / train_num)
        log['dice'].append(train_dice / train_num)
        log['val_loss'].append(val_loss)
        log['val_iou'].append(val_iou)
        log['val_dice'].append(val_dice)
        print('Validation loss: {}'.format(loss))
        torch.cuda.empty_cache()
        pd.DataFrame(log).to_csv('%s/log.csv' %
                                 los_path, index=False)
    if save_checkpoint:
        Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        torch.save(best_model, os.path.join(dir_checkpoint, 'checkpoint.pth'))
        print('Checkpoint saved!')


if __name__ == "__main__":
    config = vars(parse_args())
    now = str(datetime.datetime.now()).split()
    os.makedirs('logs/{}_{}_{}'.format(config['name'], now[0], now[1].split(':')[0]), exist_ok=True)
    log_path = 'logs/{}_{}_{}'.format(config['name'], now[0], now[1].split(':')[0])
    checkpoints_path = 'checkpoints/{}_{}_{}'.format(config['name'], now[0], now[1].split(':')[0])
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(os.path.join(log_path, 'config.yml'), 'w') as f:
        yaml.dump(config, f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        criterion = utils.loss.__dict__[config['loss']]()
    except:
        print("loss is not init in loss_file")

    if config['arch'] == "Unet":
        model = UNet(config['input_channels'], config['num_classes'])
    elif config['arch'] == "UCnet":
        model = UCNet(config['input_channels'], config['num_classes'], num=0.5)
    elif config['arch'] == "UAnet":
        model = UANet(config['input_channels'], config['num_classes'], num0=0.5, num1=0.7)
    elif config['arch'] == "ULite":
        model = ULite(config['input_channels'], config['num_classes'])
    elif config['arch'] == "ULite_Mix":
        model = ULite_Mixadd(config['input_channels'], config['num_classes'])
    elif config['arch'] == "ResUnet":
        model = Resnet34_Unet(config['input_channels'], config['num_classes'])
    elif config['arch'] == "ResUnet_Mix":
        model = Resnet34_Unet_Mix(config['input_channels'], config['num_classes'])
    elif config['arch'] == "Unet++":
        model = UnetPlusPlus(config['input_channels'], config['num_classes'])
    elif config['arch'] == "Unet++_Mix":
        model = UnetPlusPlus_Mix(config['input_channels'], config['num_classes'])
    elif config['arch'] == "UNext":
        model = UNext(config['num_classes'], config['input_channels'])
    elif config['arch'] == "UNext_Mix":
        model = UNext_Mix(config['num_classes'], config['input_channels'])
    elif config['arch'] == "UNext_S":
        model = UNext_S(config['num_classes'], config['input_channels'])
    elif config['arch'] == "UNext_S_Mix":
        model = UNext_S_Mix(config['num_classes'], config['input_channels'])
    elif config['arch'] == "UDnet":
        model = UDNet(config['input_channels'], config['num_classes'], num=0.5)

    model.to(device=device)
    data_path = os.path.join("./data", config['dataset'])

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(params, lr=config['lr'], weight_decay=config['weight_decay'], momentum=config['momentum'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config['mode'], factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    # params = list(model.parameters())
    # stat(model, (3, 256, 256), device)
    # summary(model, (3, 256, 256))
    train_net(net=model, device=device, data_path=data_path, dir_checkpoint=checkpoints_path,
              optimizer=optimizer, scheduler=scheduler, criterion=criterion, los_path=log_path,
              epochs=config['epochs'], batch_size=config['batch_size'],
              save_checkpoint=True, amp=config['amp'])




