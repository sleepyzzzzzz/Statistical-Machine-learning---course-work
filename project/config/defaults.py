from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.DIR = "ckpt/ade20k-resnet50dilated-ppm_deepsup"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.root_dataset = "./data/"
_C.DATASET.list_train = ["./data/training.odgt"]
_C.DATASET.list_val = "./data/validation.odgt"
_C.DATASET.list_test = ""
_C.DATASET.num_class = 150
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.imgSizes = (300, 375, 450, 525, 600)
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1000
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation image
_C.DATASET.img_downsampling_rate = 1.0
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 1.0
# randomly horizontally flip images when train/test
_C.DATASET.random_flip = True
_C.DATASET.train_channels = 'rgbn'
_C.DATASET.val_channels = ['rgbn']

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.arch = ""
_C.MODEL.os = 16
_C.MODEL.backbone = "resnet101"
_C.MODEL.ibn_mode = 'none'
_C.MODEL.pred_downsampling_rate = 1.0
_C.MODEL.fc_dim = 2048
_C.MODEL.num_low_level_feat = 1
_C.MODEL.interpolate_before_lastconv = False

_C.LOSS = CN()
_C.LOSS.bce = 0.0
_C.LOSS.dice = 0.0
_C.LOSS.lovasz = 0.0
_C.LOSS.focal = 0.0
# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2
_C.TRAIN.num_epoch = 15
_C.TRAIN.iter_warmup = 1000
_C.TRAIN.iter_static = 7000
_C.TRAIN.iter_decay = 17000
_C.TRAIN.start_epoch = 0

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr = 0.02
_C.TRAIN.beta1 = 0.9
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.workers = 8

_C.TRAIN.disp_iter = 20
_C.TRAIN.seed = 304

_C.TRAIN.resume_checkpoint = ''

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
_C.VAL.batch_size_per_gpu = 2
_C.VAL.visualize = False
_C.VAL.visualized_label = ""
_C.VAL.visualized_pred = ""
_C.VAL.checkpoint = "epoch_20.pth"

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.batch_size_per_gpu = 1
_C.TEST.checkpoint = "epoch_20.pth"
_C.TEST.result = "./"
