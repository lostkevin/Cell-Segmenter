# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import contextlib
from genericpath import isdir
import glob
import hashlib
import os
import random
import time
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path
from threading import Thread
from typing import Dict, Iterable, Optional, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile
from itertools import chain
from tqdm import trange
from typing import List
from torchvision.ops import masks_to_boxes
from torch.utils.data import get_worker_info
from skimage import morphology, segmentation
import torch_scatter
from scipy.ndimage import find_objects

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm
from skimage import io, exposure
import tifffile as tif
from torchvision.transforms import functional as F
from torchvision import transforms
import edt

from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str, clip_coords,
                           cv2, is_colab, is_kaggle, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.omnipose import masks_to_flows
from utils.plots import random_color
from utils.torch_utils import torch_distributed_zero_first
from utils.taskrunner import TaskRunner

# Parameters
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash

def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_det_dataloader(path,
                          imgsz,
                          batch_size,
                          hyp=None,
                          rank=-1,
                          workers=8,
                          shuffle=False,
                          filtertype=None,
                          use_blacklist=True, normalize=True, labels_dir='labels'):

    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        if isinstance(path, list):
            ds = []
            for idx, p in enumerate(path):
                ds.append(CellSegDetSet(p, hyp, img_size=imgsz, filtertype=filtertype,
                          use_blacklist=use_blacklist, normalize=normalize, label_dir=labels_dir, sample_by_scale=idx == 0))
            from torch.utils.data.dataset import ConcatDataset
            dataset = ConcatDataset(ds)
        else:
            dataset = CellSegDetSet(path, hyp, img_size=imgsz, filtertype=filtertype, normalize=normalize)
    batch_size = min(batch_size, len(dataset))
    nw = workers  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(0)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  worker_init_fn=seed_worker,
                  generator=generator,
                  multiprocessing_context='spawn',
                  collate_fn=CellSegDetSet.collate_fn, persistent_workers=False), dataset


def create_seg_dataloader(path,
                          imgsz,
                          batch_size,
                          hyp=None,
                          rank=-1,
                          workers=8,
                          shuffle=False,
                          omnipose=True):
    if isinstance(path, list):
        ds = []
        for p in path:
            with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
                ds.append(CellSegTrainSet(p, hyp, img_size=imgsz))
        from torch.utils.data.dataset import ConcatDataset
        dataset = ConcatDataset(ds)
    else:
        with torch_distributed_zero_first(rank):
            dataset = CellSegTrainSet(path, hyp, img_size=imgsz)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = workers  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(0)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  worker_init_fn=seed_worker,
                  generator=generator,
                  multiprocessing_context='spawn',
                  collate_fn=CellSegTrainSet.collate_fn, persistent_workers=True), dataset

class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

def img2label_paths(img_paths, postfix='.txt'):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + postfix for x in img_paths]

def checkDictEq(dict1: Dict, dict2: Dict):
    if type(dict1) != type(dict2) or not isinstance(dict1, dict):
        return False
    if dict1.keys() != dict2.keys():
        return False
    for k in dict1.keys():
        v1, v2 = dict1[k], dict2[k]
        if type(v1) != type(v2):
            return False
        if isinstance(v1, dict) and not checkDictEq(v1, v2):
            return False
        elif not isinstance(v1, dict):
            try:
                f = v1 == v2
            except:
                return False
            if isinstance(f, Iterable):
                try:
                    f = all(f)
                except:
                    f = f.all()
            if not f:
                return False
    return True

class CellSegDetSet(Dataset):
    """
        分割数据集
        返回: 图片, 前景mask, EDT图

        输入路径:
            root/
                images/
                labels/
    """
    VERSION = 3

    def __init__(self,
                 folder,
                 hyps,
                 *,
                 img_size=640,
                 label_postfix='_label.tiff',
                 label_dir='labels',
                 use_cache=False,
                 filtertype=None,
                 use_blacklist=True, 
                 normalize=True, 
                 sample_by_scale=True) -> None:
        super().__init__()
        self.root = folder
        self._img_size = img_size
        self._lb_postfix = label_postfix
        self._image_folder = os.path.join(folder, 'images')

        self._data_folder = os.path.join(folder, 'npy')
        self.hyps = hyps

        if not (os.path.isdir(self._image_folder)):
            raise ValueError('DO NOT FIND DATASET! Exiting...')

        self.BLACKLIST = {}
        if use_blacklist:
            self.BLACKLIST = {'cell_00142.tif', 'cell_00143.tif', 'cell_00144.tif', 'cell_00443.tif',
                              'cell_00528.tif', 'cell_00529.tif', 'cell_00547.tif', 'cell_00548.tif', }

        self.images = sorted(set(os.listdir(self._image_folder)))
        self.indexing = list(range(len(self.images)))
        self._cache = None

        self.pseudo_label = False
        if len(self.images):
            self.pseudo_label = torch.BoolTensor(['unlabel' in self.images[0]])
        if self.pseudo_label:
            LOGGER.info('Find pseudo label images...')

        self._mask_folder = os.path.join(folder, label_dir) if self.pseudo_label else os.path.join(folder, 'labels')

        if not os.path.isdir(self._mask_folder):
            raise ValueError('DO NOT FIND LABEL! Exiting...')

        if use_cache:
            self._cache = [None] * len(self.images)

        # For any images larger than self._imgsize, sample ceil(1.5 * H * w / square(self._imgsize)) + 2 times in one epoch
        self.sample_by_scale = sample_by_scale
        metainfos = None
        if os.path.exists(os.path.join(self.root, 'diams.pth')):
            self.diams = torch.load(os.path.join(self.root, 'diams.pth'))
        else:
            metainfos = self.load_metainfos()
            self.diams = [d['diam'] for d in metainfos]
            torch.save(self.diams, os.path.join(self.root, 'diams.pth'))

        if sample_by_scale or self.pseudo_label:
            if sample_by_scale:
                LOGGER.info(f'Sampling by scale...')
            elif self.pseudo_label:
                LOGGER.info(f'Remove empty images...')
            self.indexing = []
            if filtertype is not None:
                LOGGER.info(f'Train with filtered data.. type: {filtertype}')
            if metainfos is None:
                metainfos = self.load_metainfos()
            self.indexing = self.generate_indexing(metainfos, filtertype)

        oversample = self.hyps.get('oversample', False)
        if oversample and not self.pseudo_label:
            self.indexing = self.indexing * self.hyps.get('os_factor', 1) #! reweight, fl: 30  gs: 8
        
        colorjitter = self.hyps.get('colorjitter', False)
        if colorjitter:
            self.jitter = transforms.ColorJitter(brightness=self.hyps['brightness'],
                                             contrast=self.hyps['contrast'],
                                             saturation=self.hyps['saturation'],
                                             hue=self.hyps['hue'])
        else:
            self.jitter = transforms.ColorJitter() # do nothing, for acceleration

        self.grid = None
        self.normalize = normalize
        self.auto_rescale = self.hyps.get('auto_rescale', False)
        self.gpu_arr = self.hyps.get('workers_gpu', [])

    def __getitem__(self, index):
        index = self.indexing[index]
        imagename = self.images[index]
        start = time.time()
        data = self.naive_load(index)
        mask, imgtype = data['mask'], data['imgtype']
        img, diam = data['image'], self.diams[index]
        #################################################
        #! 我们希望能预测任意大小在 10 - ~400px的细胞
        #! 根据每张图片的细胞直径, 我们允许对输入下采样2^M倍至上采样一倍
        #! 200~400px: 0.5x 0.25x 0.125x
        #! 100~200px: 0.5x 0.25x
        #! 50~100px: 2x 0.5x
        #! <50px: 2x
        scale_choices = [1]
        if self.auto_rescale:
            if diam >= 50:
                scale_choices.append(0.5)
            if diam >= 100:
                scale_choices.append(0.25)
            if diam >= 200:
                scale_choices.append(0.125)
            if diam <= 100:
                scale_choices.append(2)
        scale = random.choice(scale_choices)
        if scale != 1:
            h, w = img.shape[:2]
            h, w = int(scale * h), int(scale * w)
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        #################################################

        if imgtype not in ['fl', 'bf', 'gs']:
            raise ValueError()
        gpu_arr = self.gpu_arr
        device = gpu_arr[get_worker_info().id] if get_worker_info().id < len(gpu_arr) else None

        img, mask = F.to_tensor(img).to(dtype=torch.get_default_dtype()), torch.LongTensor(mask.astype(np.int32))
        t1 = time.time()
        (img, dist), mask, bboxes, candidates = self.constraint_affine_crop([img, ], mask, device)

        flip_dims = []
        if random.random() < self.hyps['flipud'] > 0:
            flip_dims.append(-2)
            bboxes[:, 1] = 1 - bboxes[:, 1]
        if random.random() < self.hyps['fliplr'] > 0:
            flip_dims.append(-1)
            bboxes[:, 0] = 1 - bboxes[:, 0]
        if len(flip_dims) > 0:
            img, mask, dist = torch.flip(img, dims=flip_dims), torch.flip(
                mask, dims=flip_dims), torch.flip(dist, dims=flip_dims)
        bboxes = torch.cat([torch.zeros((bboxes.shape[0], 1), dtype=bboxes.dtype,
                           device=bboxes.device), bboxes], dim=1)  # N x 4 ==> N x (1 + 4)

        if self.hyps['gamma'] > 0:
            gamma = (random.random() - 0.5) * self.hyps['gamma'] + 1  # [1 - gamma / 2, 1 + gamma / 2]
            img = F.adjust_gamma(img, gamma)

        if imgtype == 'bf' and not self.pseudo_label:
            img = self.jitter(img)

        return img, mask, dist, bboxes, self.get_imgpath(imagename), self.pseudo_label

    def __len__(self):
        return len(self.indexing)

    def masks_to_bboxes(self, mask, candidates, device):
        h, w = mask.shape[-2:]
        if self.grid is None:
            grid_i, grid_j = torch.meshgrid([torch.arange(h).to(device), torch.arange(w).to(device)], indexing='ij')
            grid_i, grid_j = grid_i.reshape(-1), grid_j.reshape(-1)
            self.grid = grid_i, grid_j
        grid_i, grid_j = self.grid
        mask_flatten = mask.flatten()
        bboxes = torch.stack([torch_scatter.scatter(grid_j, mask_flatten, reduce='min'),
                              torch_scatter.scatter(grid_i, mask_flatten, reduce='min'),
                              torch_scatter.scatter(grid_j, mask_flatten, reduce='max'),
                              torch_scatter.scatter(grid_i, mask_flatten, reduce='max')], dim=1)
        res = bboxes[candidates[:bboxes.shape[0]]]
        if res.shape[0] != candidates.sum():
            LOGGER.info(f'may wrong, get {candidates}, but compute {res}')
        return res

    def constraint_affine_crop(self, imgs, mask, device=None, benchmark=False):
        # initialization
        time_dict = {}
        mask.unsqueeze_(0)
        safe_percent = 0.2
        BORDER = 200
        l, u, pad = 0, 0, (0, 0, 0, 0)
        relaxed_size = 2 * BORDER + self._img_size

        # 1.padded random crop to relaxed_size
        h, w = mask.shape[-2:]
        if min(h, w) < relaxed_size:
            diff = max(0, relaxed_size - h), max(0, relaxed_size - w)
            pad = (diff[1] // 2, diff[0] // 2, diff[1] - diff[1] // 2, diff[0] - diff[0] // 2)  # l, u, r, d
            mask = F.pad(mask, pad, fill=0)       
            h, w = mask.shape[-2:]

        if max(h, w) > relaxed_size:
            l, u = random.randint(0, w - relaxed_size), random.randint(0, h - relaxed_size)
            mask = F.crop(mask, u, l, relaxed_size, relaxed_size)
            
        if benchmark:
            time_dict['pre_crop'] = time.time()
        
        assert mask.shape[-2:] == (relaxed_size, relaxed_size)

        # generate edt
        n_pixels = torch.bincount(mask.reshape(-1)) #! faster than np.bincount
        np_mask = mask[0].numpy()
        dt = edt.edt(np_mask)
        slices = find_objects(np_mask)
        for i, slc in enumerate(slices):
            if slc is not None:
                msk = np_mask[slc] == i + 1
                dt[slc][msk] = dt[slc][msk] / (np.median(dt[slc][msk]) + 1)
        imgs.append(torch.tensor(dt))
        if benchmark:
            time_dict['edt_generation'] = time.time()

        #! all the following operation will be executed in GPU
        if device is not None:
            mask = mask.cuda(device)

        #! Random affine
        h, w = mask.shape[-2:]
        kwargs = {
            # [-theta/2, +theta/2]  !! degrees=1 means[-pi, pi]
            'angle': (random.random() - 0.5) * self.hyps['degrees'] * 360,
            'translate': [int((random.random() - 0.5) * self.hyps['translate'] * h),  # [-\alpha H/2, \alpha H/2]
                          int((random.random() - 0.5) * self.hyps['translate'] * w)],  # [-\alpha H/2, \alpha H/2]
            'scale': (random.random() - 0.5) * self.hyps['scale'] + 1,  # [1 - scale / 2, 1 + scale / 2]
            # [-theta/2, +theta/2] !! rotate=1 means[-pi, pi]
            'shear': (random.random() - 0.5) * self.hyps['shear'] * 360
        }
        mask = F.affine(mask, **kwargs).squeeze(0)
        mask_unpad = mask[BORDER:-BORDER, BORDER:-BORDER]
        h, w = mask_unpad.shape[-2:]

        #! remove invalid bounding boxes of cells
        full_pixels = (n_pixels * kwargs['scale'] ** 2).to(mask_unpad.device)
        after_pixels = torch.bincount(mask_unpad.flatten(), minlength=n_pixels.shape[0])
        after_pixels[0] = 0
        candidates = (after_pixels >= safe_percent * full_pixels) * (after_pixels > 0)
        cell = candidates.nonzero()[:, 0].view(-1, 1, 1)
        t1 = time.time()
        if True:
            bboxes = self.masks_to_bboxes(mask, candidates, device).float()  # based on scatter_min
        else:
            bboxes = masks_to_boxes(mask.cpu()[None, ...] == cell.cpu())
        t2 = time.time()
        if benchmark:
            time_dict['masks_to_boxes_delta'] = t2 - t1
        bboxes[:, -2:] += 1  # 保证左闭右开, XYXY
        #! 10.20 remove all abnormal bboxes
        centers = bboxes[:, [0, 2]].mean(dim=-1).long(), bboxes[:, [1, 3]].mean(dim=-1).long()  # xy
        in_centers = mask.to(centers[0].device)[centers[1], centers[0]] == cell.view(-1).to(centers[0].device)
        bboxes = bboxes[in_centers]

        # unpad, to xywhn
        bboxes = xyxy2xywhn(bboxes - BORDER, self._img_size, self._img_size)
        mask = mask_unpad
        if benchmark:
            time_dict['affine_crop_inner'] = time.time()

        #! process all images with pad, crop, affine parameters    
        for i in range(len(imgs)):
            if device is not None:
                imgs[i] = imgs[i].cuda(device)
            flag = imgs[i].ndim == 2
            if flag:
                imgs[i].unsqueeze_(0)
            if i < len(imgs) - 1:
                #! Never pad+crop to EDT
                imgs[i] = F.pad(imgs[i], pad, fill=0)
                imgs[i] = F.crop(imgs[i], u, l, relaxed_size, relaxed_size)
            imgs[i] = F.affine(imgs[i], interpolation=F.InterpolationMode.BILINEAR, **kwargs)
            imgs[i] = imgs[i][..., BORDER:-BORDER, BORDER:-BORDER]
            if flag:
                imgs[i].squeeze_(0)
        torch.cuda.empty_cache()
        if benchmark:
            time_dict['affine_crop_others'] = time.time()
            return imgs, mask, bboxes, candidates.nonzero().view(-1), time_dict
        return imgs, mask, bboxes, candidates.nonzero().view(-1)

    def naive_load(self, index):
        imagename = self.images[index]
        imagename_noext, ext = os.path.splitext(imagename)
        if ext in ['.bmp', '.png']:
            img = io.imread(os.path.join(self._image_folder, imagename))
        elif ext in ['.tif', '.tiff']:
            img = tif.imread(os.path.join(self._image_folder, imagename), is_ome=False)
        else:
            raise ValueError(os.path.join(self._image_folder, imagename))
        imgtype = 'bf'  # brightfield
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
            imgtype = 'gs'  # phase and diff
        elif len(img.shape) == 3:
            if img.shape[-1] > 3:
                img = img[:, :, :3]

            red = img[..., 0] / np.clip(img.sum(axis=-1), 1, None)
            red = red.mean()
            if red < 0.05:
                imgtype = 'fl'
            elif (img.max(axis=-1) - img.min(axis=-1)).mean() <= 4:
                imgtype = 'gs'
        mask = tif.imread(os.path.join(self._mask_folder, imagename_noext + self._lb_postfix))

        max_value = img.max()
        # auto-scale
        if self.normalize:
            img = img.astype(np.float32)
            for i in range(3):
                img_channel_i = img[:, :, i]
                if (img_channel_i > 0).sum():
                    img[:, :, i] = normalize_channel(img_channel_i)
        else:
            img = img / 65536 if max_value > 255 else img / 255  # to [0, 1] tensor

        data = {
            'image': img,
            'mask': mask,
            'imgtype': imgtype
        }
        return data

    def get_imgpath(self, imagename):
        return os.path.join(self._image_folder, imagename)

    @staticmethod
    def collate_fn(batch):
        img, mask, dist, bboxes, paths, plbs = zip(*batch)  # transposed
        for idx, bbox in enumerate(bboxes):
            bbox[:, 0] = idx
        return torch.stack(img, 0), torch.stack(mask, 0), torch.stack(dist, 0), torch.cat(bboxes, 0), paths, torch.cat(plbs, 0)

    def generate_indexing(self, metainfos, filtertype):
        indexing = []
        if filtertype is not None:
            LOGGER.info('remove filtertype filtering...')

        for i in range(len(self.images)):
            data, imgname = metainfos[i], self.images[i]
            h, w, empty = data['h'], data['w'], data['empty']
            # if filtertype is not None:
            #     if isinstance(filtertype, str) and imgtype != filtertype:
            #         continue
            #     if isinstance(filtertype, (list, tuple)) and imgtype not in filtertype:
            #         continue
            if imgname in self.BLACKLIST:
                continue
            if self.pseudo_label and empty:
                continue  # ! never sample this nearly empty image
            if (not self.pseudo_label) and max(h, w) > self._img_size:
                indexing.append([i] * (int(np.ceil(1.5 * h * w / (self._img_size ** 2))) + 2))
            else:
                indexing.append([i])
        return list(chain.from_iterable(indexing))

    def load_metainfos(self):
        LOGGER.info('generate metainfos...')
        if os.path.exists(os.path.join(self.root, 'metainfos.pth')):
            return torch.load(os.path.join(self.root, 'metainfos.pth'))

        def _get_metainfo(index):
            imagename = self.images[index]
            imagename_noext, ext = os.path.splitext(imagename)
            mask = tif.imread(os.path.join(self._mask_folder, imagename_noext + self._lb_postfix))
            diam = edt.edt(mask).max() * 2

            return {
                'h': mask.shape[0],
                'w': mask.shape[1],
                'empty': (mask > 0).mean() < 0.05 and mask.max() < 10,
                'version': self.VERSION,
                'diam': diam
            }
        taskrunner = TaskRunner()
        metainfos = taskrunner.map(_get_metainfo, range(len(self.images)), total=len(self.images))
        taskrunner.terminate()
        torch.save(metainfos, os.path.join(self.root, 'metainfos.pth'))
        return metainfos

    def benchmark(self):
        """
            This function benchmarks preprocess time for all images in NeurIPS-Cellseg dataset
            Time and GPU Memory Usage will be collected for all images
        """
        times = {}
        for i in trange(len(self.images)):
            time_dict = {'start': time.time()}
            data = self.naive_load(i)
            time_dict['data_load'] = time.time()

            mask, imgtype = data['mask'], data['imgtype']
            img, diam = data['image'], self.diams[i]
            #################################################
            #! 我们希望能预测任意大小在 10 - ~400px的细胞
            #! 根据每张图片的细胞直径, 我们允许对输入下采样2^M倍至上采样一倍
            #! 200~400px: 0.5x 0.25x 0.125x
            #! 100~200px: 0.5x 0.25x
            #! 50~100px: 2x 0.5x
            #! <50px: 2x
            scale_choices = [1]
            if diam >= 50:
                scale_choices.append(0.5)
            if diam >= 100:
                scale_choices.append(0.25)
            if diam >= 200:
                scale_choices.append(0.125)
            if diam <= 100:
                scale_choices.append(2)
            scale = random.choice(scale_choices)
            if scale != 1:
                h, w = img.shape[:2]
                h, w = int(scale * h), int(scale * w)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            #################################################
            device = 0
            img, mask = F.to_tensor(img).to(dtype=torch.get_default_dtype()), torch.LongTensor(mask.astype(np.int32))

            time_dict['pre_resize'] = time.time()
            (img, dist), mask, bboxes, candidates, td = self.constraint_affine_crop([img, ], mask, device, True)
            for k, v in td.items():
                time_dict[k] = v
            time_dict['affine_crop'] = time.time()

            flip_dims = []
            if random.random() < self.hyps['flipud'] > 0:
                flip_dims.append(-2)
                bboxes[:, 1] = 1 - bboxes[:, 1]
            if random.random() < self.hyps['fliplr'] > 0:
                flip_dims.append(-1)
                bboxes[:, 0] = 1 - bboxes[:, 0]
            if len(flip_dims) > 0:
                img, mask, dist = torch.flip(img, dims=flip_dims), torch.flip(
                    mask, dims=flip_dims), torch.flip(dist, dims=flip_dims)
            bboxes = torch.cat([torch.zeros((bboxes.shape[0], 1), dtype=bboxes.dtype,
                            device=bboxes.device), bboxes], dim=1)  # N x 4 ==> N x (1 + 4)

            if self.hyps['gamma'] > 0:
                gamma = (random.random() - 0.5) * self.hyps['gamma'] + 1  # [1 - gamma / 2, 1 + gamma / 2]
                img = F.adjust_gamma(img, gamma)

            if imgtype == 'bf' and not self.pseudo_label:
                img = self.jitter(img)

            time_dict['other_aug'] = time.time()
            times[self.images[i]] = time_dict
        
        import pandas as pd
        pd.DataFrame(times).to_excel('benchmark.xlsx')


class CellSegTrainSet(Dataset):
    """
        分割数据集
        返回: 图片, 前景mask, EDT图, EDT梯度图

        输入路径:
            root/
                images/
                labels/
                (npy/)
    """

    def __init__(self,
                 folder,
                 hyps,
                 *,
                 omnipose=True,
                 img_size=640,
                 label_postfix='_label.tiff',
                 use_cache=False) -> None:
        super().__init__()
        self.root = folder
        self._img_size = img_size
        self._lb_postfix = label_postfix
        self._image_folder = os.path.join(folder, 'images')
        self._mask_folder = os.path.join(folder, 'labels')
        self._data_folder = os.path.join(folder, 'npy')
        # ! If omnipose is False, dataset behave like 0915 version (except sampling strategy and augmentation)
        self._omni = omnipose
        self.hyps = hyps

        if self._omni:
            LOGGER.info('apply omnipose...')
        if not (os.path.isdir(self._image_folder) and os.path.isdir(self._mask_folder)):
            raise ValueError(f'DO NOT FIND DATASET {self._image_folder}! Exiting...')
        self.images = sorted(os.listdir(self._image_folder))
        self.indexing = list(range(len(self.images)))

    def __getitem__(self, index):
        index = self.indexing[index]
        imagename = self.images[index]
        data = self.naive_load(index)
        img, mask = data['image'], data['mask']
        device = int(os.getenv('RANK', -1))  # 0 1 2 3; 4 5 6
        if device == -1:
            device = None
        # Optional data augmentation
        img, mask = F.to_tensor(img), torch.LongTensor(mask.astype(np.int32))
        # Constraint affine: after crop, ensure 10% cell pixels in the mask
        img, mask = self.constraint_affine_crop(img, mask, device)
        flip_dims = []
        if random.random() < self.hyps['flipud'] > 0:
            flip_dims.append(-2)
        if random.random() < self.hyps['fliplr'] > 0:
            flip_dims.append(-1)
        if len(flip_dims) > 0:
            img, mask = torch.flip(img, dims=flip_dims), torch.flip(mask, dims=flip_dims)

        if self.hyps['gamma'] > 0:
            gamma = (random.random() - 0.5) * self.hyps['gamma'] + 1  # [1 - gamma / 2, 1 + gamma / 2]
            img = F.adjust_gamma(img, gamma)
        flows = None
        if self._omni:
            flows = masks_to_flows(mask, return_tensor=True, device=device)
        return img, mask, flows, self.get_imgpath(imagename)

    def __len__(self):
        return len(self.indexing)

    def constraint_affine_crop(self, img, mask, device=None):
        if device is not None:
            mask = mask.cuda(device)
        mask.unsqueeze_(0)
        BORDER = 200
        l, u = 0, 0
        h, w = img.shape[-2:]
        kwargs = {
            # [-theta/2, +theta/2]  !! degrees=1 means[-pi, pi]
            'angle': (random.random() - 0.5) * self.hyps['degrees'] * 360,
            'translate': [int((random.random() - 0.5) * self.hyps['translate'] * h),  # [-\alpha H/2, \alpha H/2]
                          int((random.random() - 0.5) * self.hyps['translate'] * w)],  # [-\alpha H/2, \alpha H/2]
            'scale': (random.random() - 0.5) * self.hyps['scale'] + 1,  # [1 - scale / 2, 1 + scale / 2]
            # [-theta/2, +theta/2] !! rotate=1 means[-pi, pi]
            'shear': (random.random() - 0.5) * self.hyps['shear'] * 360
        }
        diff = max(0, self._img_size - h), max(0, self._img_size - w)
        pad = (diff[1] // 2 + BORDER, diff[0] // 2 + BORDER, diff[1] - diff[1] //
               2 + BORDER, diff[0] - diff[0] // 2 + BORDER)  # l, u, r, d
        mask = F.pad(mask, pad, fill=0)
        mask = F.affine(mask, **kwargs).squeeze(0)

        mask_unpad = mask[BORDER:-BORDER, BORDER:-BORDER]
        h, w = mask_unpad.shape[-2:]
        mask_unpad_crop = mask_unpad
        # pad crop to img size
        if max(h, w) > self._img_size:
            l, u = random.randint(0, w - self._img_size), random.randint(0, h - self._img_size)
            mask_unpad_crop = F.crop(mask_unpad, u, l, self._img_size, self._img_size)

        mask = mask_unpad_crop  # [imgsize, imgsize]
        if device is not None:
            img = img.cuda(device)
        img = F.pad(img, pad, fill=0)
        img = F.affine(img, interpolation=F.InterpolationMode.BILINEAR, **kwargs)
        img = F.crop(img, u + BORDER, l + BORDER, self._img_size, self._img_size)
        torch.cuda.empty_cache()
        return img, mask

    def generate(self, imagename):
        imagename_noext, ext = os.path.splitext(imagename)
        img, imgtype = cellseg_open(os.path.join(self._image_folder, imagename))
        mask, bd = cellseg_maskopen(os.path.join(self._mask_folder, imagename_noext + self._lb_postfix))
        dt = edt.edt(mask)
        raw_edt = dt.copy()
        for cell_id in np.unique(mask)[1:]:
            dt[mask == cell_id] /= np.median(dt[mask == cell_id]) + 1  # strategy: peak
        # relabel
        bins = np.bincount(mask.flatten())
        idx = 1
        target = np.zeros_like(mask)
        t_bd_cells = np.zeros_like(bd)
        for i, c in enumerate(bins):
            if c > 0 and i > 0:
                target[mask == i] = idx
                t_bd_cells[bd == i] = idx
                idx += 1
        data = {
            'image': img,
            'mask': target,
            'bd_cells': t_bd_cells,
            'edt': dt,
            'raw_edt': raw_edt,
            'imgtype': imgtype
        }
        return data

    def _save_data(self, imagename):
        imagename_noext, ext = os.path.splitext(imagename)
        data = self.generate(imagename)
        np.save(os.path.join(self._data_folder, imagename_noext + '.npy'), data)

    def preprocess(self):
        os.makedirs(self._data_folder, exist_ok=True)
        taskrunner = TaskRunner(cuda_context=True)
        taskrunner.map(self._save_data, self.images, total=len(self.images))
        taskrunner.terminate()

    def load_data(self, index):
        imagename = self.images[index]
        imagename_noext, ext = os.path.splitext(imagename)
        try:
            if self._cache is not None and self._cache[index] is not None:
                return self._cache[index]
            data = np.load(os.path.join(self._data_folder, imagename_noext + '.npy'), allow_pickle=True).item()
            if self._cache is not None:
                self._cache[index] = data
            return data
        except Exception as e:
            file = os.path.join(self._data_folder, imagename_noext + '.npy')
            LOGGER.warning(f'Unable to load {file} \n Error: {e}')
            os.remove(file)
            return None

    def naive_load(self, index):
        imagename = self.images[index]
        imagename_noext, ext = os.path.splitext(imagename)
        if ext in ['.bmp', '.png']:
            img = io.imread(os.path.join(self._image_folder, imagename))
        elif ext in ['.tif', '.tiff']:
            img = tif.imread(os.path.join(self._image_folder, imagename), is_ome=False)
        else:
            raise ValueError(os.path.join(self._image_folder, imagename))
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, None], 3, axis=-1)
        elif len(img.shape) == 3:
            if img.shape[-1] > 3:
                img = img[:, :, :3]
        mask = tif.imread(os.path.join(self._mask_folder, imagename_noext + self._lb_postfix))

        max_value = img.max()
        # auto-scale

        img = img.astype(np.float32)
        for i in range(3):
            img_channel_i = img[:, :, i]
            if (img_channel_i > 0).sum():
                img[:, :, i] = normalize_channel(img_channel_i)

        data = {
            'image': img,
            'mask': mask,
        }
        return data

    def get_imgpath(self, imagename):
        return os.path.join(self._image_folder, imagename)

    @staticmethod
    def collate_fn(batch):
        img, mask, flows, paths = zip(*batch)  # transposed
        return torch.stack(img, 0), torch.stack(mask, 0), torch.stack(flows, 0), paths


class CellSegEvalDataset(Dataset):
    def __init__(self, folder, overlap_factor=0.25, load_labels=False, scale=1, normalize=True) -> None:
        super().__init__()
        if 'images' in os.listdir(folder):
            self.imgfolder = os.path.join(folder, 'images')
        else:
            self.imgfolder = folder if not folder.endswith('/') else folder[:-1]
        self.images = sorted(os.listdir(self.imgfolder))
        self._load_labels = load_labels
        if self._load_labels:
            lbsfolder = os.path.abspath(os.path.join(self.imgfolder, '../labels'))
            if not os.path.isdir(lbsfolder):
                self._load_labels = False
            else:
                self.lbsfolder = lbsfolder

        self.transforms = transforms.ToTensor()
        assert int(int(1 / overlap_factor) * overlap_factor) == 1
        self._stride = int(1 / overlap_factor) * 8
        self.overlap = overlap_factor
        self.scale = scale
        self.normalize = normalize
        # print('use')

    def __getitem__(self, index):
        im_file = os.path.join(self.imgfolder, self.images[index])
        filename_noext, ext = os.path.splitext(self.images[index])
        img, imgtype = cellseg_open(im_file, self.normalize)
        # if self.scale != 1:
        #     h, w = img.shape[:2]
        #     size = (int(self.scale * w), int(self.scale * h))
        #     img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        # raw_h, raw_w = img.shape[:2]
        #! WARNING: return RGB image
        img, pad = self.autopad(img)
        if self._load_labels:
            mask = cellseg_evalmaskopen(os.path.join(self.lbsfolder, filename_noext + '_label.tiff'))
            return filename_noext, F.to_tensor(img).to(dtype=torch.float32), pad, mask, imgtype
        return filename_noext, F.to_tensor(img).to(dtype=torch.float32), pad, imgtype

    def autopad(self, img):
        h, w = img.shape[:2]
        h_pad = 256 - h if h < 256 else int(np.ceil(h / self._stride)) * self._stride - h
        w_pad = 256 - w if w < 256 else int(np.ceil(w / self._stride)) * self._stride - w
        return np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant'), (h_pad, w_pad)

    def __len__(self):
        return len(self.images)
# Ancillary functions --------------------------------------------------------------------------------------------------

def normalize_channel(img, lower=1, upper=99):
    """
        最小的1% 以及最大的1% 会被clip, 其余值线性缩放至0~255
    """
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img.astype(float), in_range=(
            percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8) / 255

def cellseg_open(im_file, normalize=True, old=True):
    _, ext = os.path.splitext(im_file)
    if ext in ['.bmp', '.png']:
        img = io.imread(im_file)
    elif ext in ['.tif', '.tiff']:
        img = tif.imread(im_file, is_ome=False)
    else:
        raise ValueError(im_file)
    imgtype = 'bf'  # brightfield
    if len(img.shape) == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
        imgtype = 'gs'  # phase and diff
    elif len(img.shape) == 3:
        if img.shape[-1] > 3:
            img = img[:, :, :3]

        red = img[..., 0] / np.clip(img.sum(axis=-1), 1, None)
        red = red.mean()
        if red < 0.05:
            imgtype = 'fl'
        elif (img.max(axis=-1) - img.min(axis=-1)).mean() <= 4:
            imgtype = 'gs'

    max_value = img.max()
    # auto-scale
    if normalize:
        img = img.astype(np.float32)
        for i in range(3):
            img_channel_i = img[:, :, i]
            if (img_channel_i > 0).sum():
                img[:, :, i] = normalize_channel(img_channel_i)
    else:
        img = img / 65536 if max_value > 255 else img / 255  # to [0, 1] tensor
    return img.astype(float), imgtype

def cellseg_maskopen(mask_file):
    return cellseg_evalmaskopen(mask_file)

def cellseg_evalmaskopen(mask_file):
    _, ext = os.path.splitext(mask_file)
    if ext in ['.tif', '.tiff']:
        img = tif.imread(mask_file, is_ome=False)
    else:
        raise NotImplementedError()
    return img.astype(np.int32)


if __name__ == '__main__':
    import yaml
    with open('data/hyps/hyp.labeled.yaml', errors='ignore') as f:
        hyp = yaml.safe_load(f)  # load hyps dict
    dataset = CellSegDetSet('/nfs4-p1/hkw/CellSeg_data/NeurIPS_CellSegData/Train_Labeled', hyps=hyp)
    dataset.benchmark()