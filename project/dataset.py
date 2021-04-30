import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

np.random.seed(42)

def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # down sampling rate of segm img
        self.img_downsampling_rate = opt.img_downsampling_rate
        # down sampling rate of segm label
        self.segm_downsampling_rate = opt.segm_downsampling_rate

        self.parse_input_list(odgt, **kwargs)

        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.nir_normalize = transforms.Normalize(
            mean=[0.485],
            std=[0.229]
        )

        self.classes = ['background',
                        'cloud_shadow',
                        'double_plant',
                        'planter_skip',
                        'standing_water',
                        'waterway',
                        'weed_cluster']

        self.img_down_size = lambda img: imresize(
            img,
            (int(img.size[0] / self.img_downsampling_rate), int(img.size[1] / self.img_downsampling_rate)),
            interp='bilinear')

        self.label_down_size = lambda label: imresize(
            label,
            (int(label.size[0] / self.segm_downsampling_rate), int(label.size[1] / self.segm_downsampling_rate)),
            interp='nearest')

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):        
        if isinstance(odgt, str):
            odgt = [odgt]

        self.list_sample = []
        for o in odgt:
            self.list_sample += [json.loads(x.rstrip()) for x in open(o, 'r')]

        self.list_sample = np.random.permutation(self.list_sample)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        # self.list_sample = self.list_sample * 5

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __len__(self):
        return self.num_sample

    def img_transform(self, rgb, nir):
        # 0-255 to 0-1
        rgb = np.float32(np.array(rgb)) / 255.
        nir = np.expand_dims(np.float32(np.array(nir)), axis=2) / 255.

        rgb = rgb.transpose((2, 0, 1))
        nir = nir.transpose((2, 0, 1))

        rgb = self.rgb_normalize(torch.from_numpy(rgb))
        nir = self.nir_normalize(torch.from_numpy(nir))

        if self.channels == 'rgbn':
            img = torch.cat([rgb, nir], axis=0)
        elif self.channels == 'rgb':
            img = rgb
        elif self.channels == 'nir3':
            img = torch.cat([nir, nir, nir], axis=0)
        elif self.channels == 'nir4':
            img = torch.cat([nir, nir, nir, nir], axis=0)
        elif self.channels == 'rgbr':
            img = torch.cat([rgb, rgb[0: 1]], axis=0)
        else:
            raise NotImplementedError

        return img

    def get_valid_mask(self, boundary, mask):
        boundary = np.array(boundary) / 255.
        mask = np.array(mask) / 255.

        boundary = torch.from_numpy(boundary).long()
        mask = torch.from_numpy(np.array(mask)).long()

        return boundary * mask

    def get_label(self, label_imgs):
        labels = [torch.from_numpy(np.array(img) / 255.).long() for img in label_imgs]
        labels = torch.stack(labels, dim=0)

        sumed = labels.sum(dim=0, keepdim=True)
        bg_channel = torch.zeros_like(sumed)
        bg_channel[sumed == 0] = 1

        return torch.cat((bg_channel, labels), dim=0).float()


class AgriTrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, channels='rgbn', **kwargs):
        super(AgriTrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

        self.channels = channels

    def __getitem__(self, index):
        sample_odgt = self.list_sample[index]

        rgb_path = os.path.join(self.root_dataset, sample_odgt['fpath_rgb'])
        nir_path = os.path.join(self.root_dataset, sample_odgt['fpath_nir'])

        rgb = Image.open(rgb_path).convert('RGB')
        nir = Image.open(nir_path).convert('L') #Greyscale

        rgb, nir = self.img_down_size(rgb), self.img_down_size(nir)

        boundary_path = os.path.join(self.root_dataset, sample_odgt['fpath_boundary'])
        mask_path = os.path.join(self.root_dataset, sample_odgt['fpath_mask'])

        boundary = Image.open(boundary_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        boundary, mask = self.label_down_size(boundary), self.label_down_size(mask)

        label_paths = sample_odgt['fpath_label']
        label_imgs = []
        for c in self.classes[1:]:
            mask_path = os.path.join(self.root_dataset, label_paths[c])
            label_imgs.append(Image.open(mask_path).convert('L'))

        rgb = np.array(rgb)
        nir = np.array(nir)
        boundary = np.array(boundary)
        mask = np.array(mask)
        label_imgs = [np.array(self.label_down_size(label_img)) for label_img in label_imgs]

        rgb, nir, boundary, mask, label_imgs = self.randomHorizontalFlip(
            rgb, nir, boundary, mask, label_imgs)

        rgb, nir, boundary, mask, label_imgs = self.randomVerticalFlip(
            rgb, nir, boundary, mask, label_imgs)

        rgb, nir, boundary, mask, label_imgs = self.randomRotate90(
            rgb, nir, boundary, mask, label_imgs)

        img = self.img_transform(rgb, nir)
        valid_mask = self.get_valid_mask(boundary, mask)
        label = self.get_label(label_imgs)

        label *= valid_mask.unsqueeze(dim=0)

        return img, valid_mask, label

    def randomHorizontalFlip(self, rgb, nir, boundary, mask, labels,
                             u=0.5):
        if np.random.random() < u:
            rgb = cv2.flip(rgb, 1)
            nir = cv2.flip(nir, 1)
            boundary = cv2.flip(boundary, 1)
            mask = cv2.flip(mask, 1)

            for i in range(len(labels)):
                labels[i] = cv2.flip(labels[i], 1)

        return rgb, nir, boundary, mask, labels

    def randomVerticalFlip(self, rgb, nir, boundary, mask, labels,
                             u=0.5):
        if np.random.random() < u:
            rgb = cv2.flip(rgb, 0)
            nir = cv2.flip(nir, 0)
            boundary = cv2.flip(boundary, 0)
            mask = cv2.flip(mask, 0)

            for i in range(len(labels)):
                labels[i] = cv2.flip(labels[i], 0)

        return rgb, nir, boundary, mask, labels

    def randomRotate90(self, rgb, nir, boundary, mask, labels,
                             u=0.5):
        if np.random.random() < u:
            rgb = np.rot90(rgb)
            nir = np.rot90(nir)
            boundary = np.rot90(boundary)
            mask = np.rot90(mask)

            for i in range(len(labels)):
                labels[i] = np.rot90(labels[i])

        return rgb, nir, boundary, mask, labels


class AgriValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, channels='rgbn', **kwargs):
        super(AgriValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

        # down sampling rate of segm img
        self.img_downsampling_rate = opt.img_downsampling_rate
        # down sampling rate of segm label
        self.segm_downsampling_rate = opt.segm_downsampling_rate

        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.nir_normalize = transforms.Normalize(
            mean=[0.485],
            std=[0.229]
        )

        self.channels = channels

    def __getitem__(self, index):
        sample_odgt = self.list_sample[index]

        rgb_path = os.path.join(self.root_dataset, sample_odgt['fpath_rgb'])
        nir_path = os.path.join(self.root_dataset, sample_odgt['fpath_nir'])

        rgb = Image.open(rgb_path).convert('RGB')
        nir = Image.open(nir_path).convert('L') #Greyscale

        rgb, nir = self.img_down_size(rgb), self.img_down_size(nir)

        # image transform, to torch float tensor 4xHxW
        img = self.img_transform(rgb, nir)

        boundary_path = os.path.join(self.root_dataset, sample_odgt['fpath_boundary'])
        mask_path = os.path.join(self.root_dataset, sample_odgt['fpath_mask'])

        boundary = Image.open(boundary_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        boundary, mask = self.label_down_size(boundary), self.label_down_size(mask)
        valid_mask = self.get_valid_mask(boundary, mask)

        label_paths = sample_odgt['fpath_label']
        label_imgs = []
        for c in self.classes[1:]:
            mask_path = os.path.join(self.root_dataset, label_paths[c])
            label_imgs.append(Image.open(mask_path).convert('L'))

        label_imgs = [self.label_down_size(label_img) for label_img in label_imgs]
        label = self.get_label(label_imgs)

        info = rgb_path.split('/')[-1]
        return img, valid_mask, label


class AgriTestDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(AgriTestDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.485],
            std=[0.229, 0.224, 0.225, 0.229])

    def __getitem__(self, index):
        sample_odgt = self.list_sample[index]

        rgb_path = os.path.join(self.root_dataset, sample_odgt['fpath_rgb'])
        nir_path = os.path.join(self.root_dataset, sample_odgt['fpath_nir'])

        rgb = Image.open(rgb_path).convert('RGB')
        nir = Image.open(nir_path).convert('L') #Greyscale

        rgb, nir = self.img_down_size(rgb), self.img_down_size(nir)

        # image transform, to torch float tensor 4xHxW
        img = self.img_transform(rgb, nir)

        boundary_path = os.path.join(self.root_dataset, sample_odgt['fpath_boundary'])
        mask_path = os.path.join(self.root_dataset, sample_odgt['fpath_mask'])

        boundary = Image.open(boundary_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        boundary, mask = self.label_down_size(boundary), self.label_down_size(mask)
        valid_mask = self.get_valid_mask(boundary, mask)

        info = rgb_path.split('/')[-1]

        return img, valid_mask, info

    def img_transform(self, rgb, nir):
        # 0-255 to 0-1
        rgb = np.float32(np.array(rgb)) / 255.
        nir = np.float32(np.array(nir)) / 255.

        img = np.concatenate((rgb, np.expand_dims(nir, axis=2)), axis=2) #shape as (512, 512, 4)

        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img
