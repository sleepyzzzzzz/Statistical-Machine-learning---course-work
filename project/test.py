import argparse
import os
import time
from tqdm import tqdm
import shutil
from datetime import datetime

import matplotlib.pyplot as plt

import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision.transforms import ToPILImage

import segmentation_models_pytorch as smp
import ttach as tta

from config import cfg
from utils import *
from dataset import AgriTestDataset
from model.deeplab import DeepLab


def save_result(info, pred):
    classes = pred.argmax(dim=1, keepdim=True).cpu()
    for i in range(classes.shape[0]):
        result_png = ToPILImage()(classes[i].float() / 255.)
        img_name = info[i]
        result_png.save(os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))


def test(loader_test, model, args, logger):
    model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode="mean")
    model.eval()
    loader_test = tqdm(loader_test, total=cfg.TEST.epoch_iters)

    with torch.no_grad():
        for img, mask, info in loader_test:
            img = img.cuda()
            mask = mask.cuda()

            pred = model(img)

            save_result(info, pred)


def get_args():
    parser = argparse.ArgumentParser(
        description="Test"
    )

    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    return parser.parse_args()


def main():
    args = get_args()

    args.world_size = 1

    cfg.merge_from_file(args.cfg)

    cfg.DIR = os.path.join(cfg.DIR,
                           args.cfg.split('/')[-1].split('.')[0] +
                           datetime.now().strftime('-%Y-%m-%d-%a-%H,%M,%S,%f'))


    os.makedirs(cfg.DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.DIR, 'weight'), exist_ok=True)
    os.makedirs(os.path.join(cfg.DIR, 'history'), exist_ok=True)
    shutil.copy(args.cfg, cfg.DIR)

    if os.path.exists(os.path.join(cfg.DIR, 'log.txt')):
        os.remove(os.path.join(cfg.DIR, 'log.txt'))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger(distributed_rank=0,
                          filename=os.path.join(cfg.DIR, 'log.txt'))
    logger.info(f'Using device {device}')

    model = DeepLab(num_classes=cfg.DATASET.num_class,
                   backbone=cfg.MODEL.backbone,                  # resnet101
                   output_stride=cfg.MODEL.os,
                   ibn_mode=cfg.MODEL.ibn_mode,
                   num_low_level_feat=cfg.MODEL.num_low_level_feat)

    convert_model(model, 4)
    from pytorch_model_summary import summary
    print(summary(model, torch.zeros((1, 4, 512, 512)), show_input=True))

    model = model.cuda()

    if cfg.TEST.checkpoint != "":
        logger.info("Loading weight from {}".format(
                cfg.TEST.checkpoint))
        model.load_state_dict(torch.load(cfg.TEST.checkpoint))

    dataset_test = AgriTestDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_test,
        cfg.DATASET)

    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size_per_gpu,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    cfg.TEST.epoch_iters = len(loader_test)

    logger.info("World Size: {}".format(args.world_size))
    logger.info("TEST.epoch_iters: {}".format(cfg.TEST.epoch_iters))
    logger.info("TEST.sum_bs: {}".format(cfg.TEST.batch_size_per_gpu *
                                         args.world_size))

    test(loader_test, model, args, logger)

if __name__ == '__main__':
    main()
