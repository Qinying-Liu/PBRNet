import os
import numpy as np
import shutil
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from config import cfg, params
from tqdm import tqdm
from i3d import I3D
from network import PBRNet
from utils import *
from dataset import THUMOS
import time
from tensorboardX import SummaryWriter


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eval()


def main():
    train_annotation, _ = read_annotation(cfg.train_annotation_path)
    train_sequence, train_max_num = generate_sequence(cfg.train_data_path, train_annotation, cfg.train_overlap,
                                                      cfg.num_frame_clip, cfg.fps)
    print('The length of training sequence is {}'.format(len(train_sequence)))
    val_annotation, _ = read_annotation(cfg.val_annotation_path)
    val_sequence, val_max_num = generate_sequence(cfg.val_data_path, val_annotation, cfg.val_overlap,
                                                  cfg.num_frame_clip, cfg.fps)
    print('The length of validation sequence is {}'.format(len(val_sequence)))
    # max_num = max(train_max_num, val_max_num)
    train_dataset = THUMOS(train_sequence, cfg.train_data_path, cfg.num_frame_clip, cfg.crop_size, train_max_num)
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    val_dataset = THUMOS(val_sequence, cfg.val_data_path, cfg.num_frame_clip, cfg.crop_size, val_max_num, False)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False)

    if os.path.exists(cfg.save_path):
        shutil.rmtree(cfg.save_path)

    os.makedirs(cfg.save_path)

    writer = SummaryWriter(os.path.join(cfg.save_path, 'tensorboard'))

    anchors = anchors_all_layers(params.layer_shape, params.anchor_sizes, cfg.num_frame_clip)

    backbone = I3D(num_classes=400, modality='rgb')
    pretrained_dict = torch.load('model/model_rgb.pth')
    model_dict = backbone.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    backbone.load_state_dict(model_dict)
    model = PBRNet(backbone, anchors, cfg.num_classes)
    model = torch.nn.DataParallel(model).cuda()

    model = model.cuda()
    model.train()
    model.apply(freeze_bn)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    lr_scheduler = MultiStepLR(optimizer, params.milestones)
    num_iter = 0
    val_loader = iter(val_loader)
    for epoch in tqdm(range(cfg.epoch), ncols=70, leave=False, unit='b'):
        for clip_data, actions, labels, start_label, end_label in tqdm(train_loader, ncols=70, leave=False, unit='b'):
            loss, _, _, _, _, _ = model(
                clip_data.cuda(),
                actions.cuda(),
                labels.cuda(),
                start_label.cuda(),
                end_label.cuda())
            total_loss = 0
            for k in loss.keys():
                # print(k, 'is ', loss[k].item())
                total_loss = total_loss + loss[k] * params.weights[k]
            optimizer.zero_grad()
            total_loss.mean().backward()
            optimizer.step()
            lr_scheduler.step()

            if num_iter % cfg.sum_freq == 0:
                writer.add_scalars('total', {'train': total_loss.mean()}, num_iter)
                for loss_name in loss.keys():
                    writer.add_scalars(loss_name, {'train': loss[loss_name].mean()}, num_iter)
                clip_data, actions, labels, start_label, end_label = next(val_loader)
                model.eval()
                with torch.no_grad():
                    loss, _, _, _, _, _ = model(
                        clip_data.cuda(),
                        actions.cuda(),
                        labels.cuda(),
                        start_label.cuda(),
                        end_label.cuda())
                model.train()
                model.apply(freeze_bn)
                total_loss = 0
                for loss_name in loss.keys():
                    total_loss = total_loss + loss[loss_name] * params.weights[loss_name]
                writer.add_scalars('total', {'val': total_loss.mean()}, num_iter)
                for loss_name in loss.keys():
                    writer.add_scalars(loss_name, {'val': loss[loss_name].mean()}, num_iter)
                writer.flush()
            if num_iter % cfg.save_freq == 0:
                save_obj = {
                    'config': cfg,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict()
                }
                torch.save(save_obj, os.path.join(cfg.save_path, 'epoch-{}.pth'.format(epoch)))
            num_iter = num_iter + 1


if __name__ == '__main__':
    main()
