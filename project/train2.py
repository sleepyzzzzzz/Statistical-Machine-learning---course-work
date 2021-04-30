import argparse
import os
import time
from tqdm import tqdm
import shutil
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.nn as nn

import segmentation_models_pytorch as smp

from config import cfg
from utils import *
from dataset import AgriTrainDataset, AgriValDataset
from model.deeplab import DeepLab
from model.loss import ComposedLossWithLogits

torch.manual_seed(42)
np.random.seed(42)

def get_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )

    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
        dest='cfg'
    )

    return parser.parse_args()


def main():
	args = get_args()
	cfg.merge_from_file(args.cfg)
	cfg.DIR = os.path.join(cfg.DIR,
                           args.cfg.split('/')[-1].rstrip('.yaml') +
                           datetime.now().strftime('-%Y-%m-%d-%a-%H-%M-%S-%f'))
	os.makedirs(cfg.DIR, exist_ok=True)
	os.makedirs(os.path.join(cfg.DIR, 'weight'), exist_ok=True)
	os.makedirs(os.path.join(cfg.DIR, 'history'), exist_ok=True)
	shutil.copy(args.cfg, cfg.DIR)
	if os.path.exists(os.path.join(cfg.DIR, 'log.txt')):
		os.remove(os.path.join(cfg.DIR, 'log.txt'))
	args.world_size = 1
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	logger = setup_logger(distributed_rank=0,
                          filename=os.path.join(cfg.DIR, 'log.txt'))
	logger.info(f'Using device {device}')

	# model = smp.Unet(encoder_name='resnet101',classes=7)
	model = DeepLab(num_classes=cfg.DATASET.num_class,
	            backbone=cfg.MODEL.backbone,
	            output_stride=cfg.MODEL.os,
	            ibn_mode=cfg.MODEL.ibn_mode,
	            num_low_level_feat=cfg.MODEL.num_low_level_feat,
	            interpolate_before_lastconv=cfg.MODEL.interpolate_before_lastconv)

	convert_model(model, 4)

	model = model.cuda()

	loss_fn = ComposedLossWithLogits(dict(cfg.LOSS)).cuda()

	optimizer = torch.optim.SGD(model.parameters(),
	                                 lr=cfg.TRAIN.lr,
	                                 weight_decay=cfg.TRAIN.weight_decay,
	                                 momentum=cfg.TRAIN.beta1)

	train_dataset = AgriTrainDataset(
	    cfg.DATASET.root_dataset,
	    cfg.DATASET.list_train,
	    cfg.DATASET,
	    channels='rgbn')

	val_dataset = AgriValDataset(
	        cfg.DATASET.root_dataset,
	        cfg.DATASET.list_val,
	        cfg.DATASET,
	        channels='rgbn')

	loader_train = torch.utils.data.DataLoader(
	    train_dataset,
	    batch_size=cfg.TRAIN.batch_size_per_gpu,
	    shuffle=False,
	    num_workers=cfg.TRAIN.workers,
	    drop_last=True,
	    pin_memory=True,
	)

	loader_val = torch.utils.data.DataLoader(
	        val_dataset,
	        batch_size=cfg.VAL.batch_size_per_gpu,
	        shuffle=False,
	        num_workers=cfg.VAL.batch_size_per_gpu,
	        drop_last=True,
	        pin_memory=True,
	)

	cfg.TRAIN.epoch_iters = len(loader_train)
	cfg.VAL.epoch_iters = len(loader_val)

	cfg.TRAIN.running_lr = cfg.TRAIN.lr

	cfg.TRAIN.log_fmt = 'TRAIN >> Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, ' \
	                    'lr: {:.6f}, Loss: {:.6f}'

	cfg.VAL.log_fmt = 'Mean IoU: {:.4f}\nMean Loss: {:.6f}'

	logger.info("TRAIN:epoch_iters: {}".format(cfg.TRAIN.epoch_iters))
	logger.info("TRAIN.sum_bs: {}".format(cfg.TRAIN.batch_size_per_gpu))

	logger.info("VAL.epoch_iters: {}".format(cfg.VAL.epoch_iters))
	logger.info("VAL.sum_bs: {}".format(cfg.VAL.batch_size_per_gpu))

	logger.info("TRAIN.num_epoch: {}".format(cfg.TRAIN.num_epoch))

	history = init_history(cfg)
	for i in range(cfg.TRAIN.num_epoch):
		train(i + 1, loader_train, model, loss_fn, optimizer, history, args, logger)

		val(i + 1, loader_val, model, loss_fn, history, args, logger)

		checkpoint(model, history, cfg, i + 1, args, logger)

def adjust_learning_rate(optimizer, cur_iter, cfg):
    if cur_iter < cfg.TRAIN.iter_warmup:
        scale_running_lr = (cur_iter + 1) / cfg.TRAIN.iter_warmup
    elif cur_iter < (cfg.TRAIN.iter_warmup + cfg.TRAIN.iter_static):
        scale_running_lr = 1
    else:
        cur_iter -= cfg.TRAIN.iter_warmup + cfg.TRAIN.iter_static
        scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.iter_decay) ** cfg.TRAIN.lr_pow)

    cfg.TRAIN.running_lr = cfg.TRAIN.lr * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr


def init_history(cfg):
    from copy import deepcopy

    losses_dict = {name: [] for name, _ in dict(cfg.LOSS).items() if _ != 0}
    val_dict = {'epoch': [], 'sum_loss': [], 'losses': losses_dict, 'mean_iou': []}

    history = {'train': {'epoch': [], 'sum_loss': [], 'losses': losses_dict, 'lr': []},
               'val': {}}

    for channels in cfg.DATASET.val_channels:
        history['val'][channels] = deepcopy(val_dict)

    return history

def train(epoch, loader_train, model, loss_fn, optimizer, history, args, logger):
	iter_time = AverageMeter()
	data_time = AverageMeter()
	model.train()
	tic = time.time()

	for i, (img, mask, label) in enumerate(loader_train):
		img = img.cuda()
		mask = mask.cuda()
		label = label.cuda()
		label *= mask.unsqueeze(1)
		optimizer.zero_grad()
		cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
		#adjust_learning_rate(optimizer, cur_iter, cfg)
		data_time.update(time.time() - tic)
		pred = model(img)
		pred[~mask.unsqueeze(1).expand_as(pred).bool()] = -2 ** 15
		sum_loss, losses = loss_fn(pred, label)

		sum_loss.backward()
		optimizer.step()

		reduced_sum_loss = sum_loss.item()
		reduced_losses = losses.tolist()

		iter_time.update(time.time() - tic)
		tic = time.time()
		if i % cfg.TRAIN.disp_iter == 0:
			logger.info(cfg.TRAIN.log_fmt
                        .format(epoch, i, cfg.TRAIN.epoch_iters,
                                iter_time.average(), data_time.average(),
                                cfg.TRAIN.lr,
                                #cfg.TRAIN.running_lr,
                                reduced_sum_loss, *reduced_losses))

			fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters

			history['train']['epoch'].append(fractional_epoch)
			history['train']['sum_loss'].append(reduced_sum_loss)
			history['train']['lr'].append(cfg.TRAIN.running_lr)

def val(epoch, loader_val, model, loss_fn, history, args, logger):
	avg_sum_loss = AverageMeter()
	avg_losses = AverageMeter()
	intersection_meter = AverageMeter()
	union_meter = AverageMeter()

	model.eval()

	tic = time.time()

	channels = loader_val.dataset.channels
	loader_val = tqdm(loader_val, total=cfg.VAL.epoch_iters)

	with torch.no_grad():
		for img, mask, label in loader_val:
			img = img.cuda()
			mask = mask.cuda()
			label = label.cuda()
			label *= mask.unsqueeze(1)
			pred = model(img)
			pred[~mask.unsqueeze(1).expand_as(pred).bool()] = -2 ** 15
			sum_loss, losses = loss_fn(pred, label)

			reduced_sum_loss = sum_loss.item()
			reduced_losses = losses.data

			avg_sum_loss.update(reduced_sum_loss)
			avg_losses.update(reduced_losses)

			intersection, union = intersectionAndUnion(pred.data, label.data, 0)
			intersection_meter.update(intersection)
			union_meter.update(union)

	reduced_inter = intersection_meter.sum.cpu()
	reduced_union = union_meter.sum.cpu()
	iou = reduced_inter / (reduced_union + 1e-10)

	losses = avg_losses.average().tolist()

	for i, _iou in enumerate(iou):
 		logger.info('class [{}], IoU: {:.4f}'.format(i, _iou))

	logger.info('[Eval Summary][Channels: {}]:'.format(channels))
	logger.info(cfg.VAL.log_fmt.format(iou.mean(), avg_sum_loss.average(), *losses))

	history['val'][channels]['epoch'].append(epoch)
	history['val'][channels]['sum_loss'].append(avg_sum_loss.average())
	history['val'][channels]['mean_iou'].append(iou.mean().item())


def checkpoint(model, history, cfg, epoch, args, logger):
	logger.info("Saving checkpoints to '{}'".format(cfg.DIR))

	dict_model = model.state_dict()

	torch.save(history,'{}/history_epoch_{}.pth'.format(os.path.join(cfg.DIR, 'history'), epoch))
	torch.save(dict_model,'{}/weight_epoch_{}.pth'.format(os.path.join(cfg.DIR, 'weight'), epoch))


if __name__ == "__main__":
	main()
