import os
import torch
import numpy as np
import tifffile as tif
import edt

from typing import *
from monai.inferers import sliding_window_inference
from argparse import ArgumentParser
from torch import nn
from types import MethodType
from tqdm import tqdm
from torch.nn import functional as F
from skimage import exposure, io
from torchvision.transforms import functional as tvF

from models.experimental import attempt_load
from utils.postprocess import Postprocessor
from utils.mask import compute_masks

def detect_sw_forward(self, x):
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
    if hasattr(self, 'seg') and self.seg is not None:
        x[-1] = self.seg(x[-1])
    return tuple(x)

def predict_one(
    model: nn.Module,
    padded_image: torch.Tensor,
    postprocessor: Postprocessor,
    window_size: Tuple = (640, 640),
    sw_batch_size: int = 4,
    overlap: float=0.125,
    *,
    sw_mode: str = 'gaussian',
):
    pred = sliding_window_inference(padded_image, window_size, sw_batch_size, model, mode=sw_mode, overlap=overlap)  # ! 滑窗方法将影响检测的性能
    h, w = padded_image.shape[-2:]
    raw_dets, segs = postprocessor.postprocess(pred)
    dets = postprocessor.nms(raw_dets)

    det, seg = dets[0], None
    if len(det) < 2000:
        det = postprocessor.filter_with_ioa(det)
    seg = segs[0]
    if segs[0].shape[-2:] != (h, w):
        seg = F.interpolate(segs[0], (h, w), mode='bilinear', align_corners=True)
    fgConf, height = seg[0]
    fgConf = fgConf.sigmoid()  # ! FIX
    det = det.cpu().numpy()
    #! typeof mask: np.int32
    mask = postprocessor.generate_mask(fgConf, height, det[:, :4], det[:, -1])
    height = height.cpu().numpy()
    mask, removecnt = postprocessor.refine(mask, height)
    # if removecnt > 0:
    #     print(f'remove {removecnt} cells')
    return mask, fgConf.cpu().numpy(), height

def predict_one_omni(
    model: nn.Module,
    padded_image: torch.Tensor,
    flow_threshold: float,
    velocity: float,
    window_size: Tuple = (640, 640),
    sw_batch_size: int = 4,
    overlap: float=0.25,
    *,
    sw_mode: str = 'gaussian',
):
    h, w = padded_image.shape[-2:]
    pred = sliding_window_inference(padded_image, window_size, sw_batch_size, model, mode=sw_mode, overlap=overlap)  # ! 滑窗方法将影响检测的性能
    pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
    pred = pred[0]
    bd, dist, flows = torch.tensor_split(pred, [1, 2], dim=0)
    mask, _, _ = compute_masks(flows, dist[0], bd[0], flow_threshold=flow_threshold, velocity=velocity)
    return mask, dist.sigmoid().cpu().numpy()[0]

def normalize_channel(img, lower=1, upper=99):
    """
        最小的1% 以及最大的1% 会被clip, 其余值线性缩放至0~255
    """
    non_zero_vals = img[np.nonzero(img)]
    percentiles = np.percentile(non_zero_vals, [lower, upper])
    if percentiles[1] - percentiles[0] > 0.001:
        img_norm = exposure.rescale_intensity(img, in_range=(percentiles[0], percentiles[1]), out_range='uint8')
    else:
        img_norm = img
    return img_norm.astype(np.uint8) / 255 ####! Old Behavior

def cellseg_open(im_file):
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
    
    img = img.astype(np.float32)

    for i in range(3):
        img_channel_i = img[:, :, i]
        if (img_channel_i > 0).sum():
            img[:, :, i] = normalize_channel(img_channel_i)
    return img, imgtype

def autopad(img, stride):
    h, w = img.shape[:2]
    h_pad = 256 - h if h < 256 else int(np.ceil(h / stride)) * stride - h
    w_pad = 256 - w if w < 256 else int(np.ceil(w / stride)) * stride - w
    return np.pad(img, ((0, h_pad), (0, w_pad), (0, 0)), mode='constant'), (h_pad, w_pad)

@torch.no_grad()
def auto_predict(
    inputs_dir: str, 
    outputs_dir: str,
    *,
    models_dir='ckpts',
    overlap=0.125,
    window_size=640,
    sw_batch_size=4,
    flow_threshold=0.4,
    velocity=1.25):

    stride = int(1 / overlap) * 8
    files = os.listdir(inputs_dir)
    window_size = (window_size, window_size)

    model_dict = {
        'bf': 'general.pt',
        'gs': 'grayscale.pt',
        'fl': 'fl.pt',
        'omni': 'omnipose.pt'
    }

    postprocessors_kwargs = {
        'bf': {'conf_thres':0.1, 'iou_thres':0.25},
        'gs': {'conf_thres':0.2, 'iou_thres':0.25},
        'fl': {'conf_thres':0.25, 'iou_thres':0.35},
    }

    """
        Inference will have one image in each folder, 

        load model --> infer --> load model ...
    """
    if len(files) == 1:
        file = files[0]
        filename_noext, ext = os.path.splitext(file)
        image, imgtype = cellseg_open(os.path.join(inputs_dir, file))
        image, (h_pad, w_pad) = autopad(image, stride)
        
        image = tvF.to_tensor(image).unsqueeze(0).cuda() # HWC --> 1CHW, RGB
        h, w = image.shape[-2:]

        model = attempt_load(os.path.join(models_dir, model_dict[imgtype]) , fuse=False).eval().cuda()
        postprocessor = Postprocessor(model.model[-1], **postprocessors_kwargs[imgtype])
        model.model[-1].forward = MethodType(detect_sw_forward, model.model[-1])
        mask, fgConf, height = predict_one(model, image, postprocessor, window_size, sw_batch_size, overlap)
        mask = mask[:h - h_pad, :w - w_pad]
        del model

        if imgtype == 'gs':
            fgConf = fgConf[:h - h_pad, :w - w_pad]
            height = height[:h - h_pad, :w - w_pad]

            quality = 0
            if mask.max() > 0:
                quality =  fgConf[fgConf > 0.5].mean()
            omni_model = attempt_load(os.path.join(models_dir, model_dict['omni']) , fuse=False).eval().cuda()
            omni_mask, fgConf = predict_one_omni(omni_model, image, flow_threshold, velocity, window_size, sw_batch_size, overlap)
            del omni_model
            fgConf = fgConf[:h - h_pad, :w - w_pad]
            omni_mask = omni_mask[:h - h_pad, :w - w_pad]
            omni_quality = fgConf[fgConf > 0.5].mean()
            if quality < 0.8 or omni_quality - quality > 0.05:
                # print(f'{filename_noext}: {omni_quality} || {quality}, apply omnipose')
                mask = omni_mask
            # else:
            #     print(f'{filename_noext}: {omni_quality} || {quality}')
        mask = mask.astype(np.uint16) if mask.max() < 65536 else mask.astype(np.int32)
        tif.imwrite(os.path.join(outputs_dir, f'{filename_noext}_label.tiff'), mask, compression='zlib')
        return     

    #! This branch do not consider 10G limit
    models = {
        k:  attempt_load(os.path.join(models_dir, v) , fuse=False).eval().cuda() for k, v in model_dict.items()
    }

    for k in ['bf', 'gs', 'fl']:
        models[k].model[-1].forward = MethodType(detect_sw_forward, models[k].model[-1])
    omni_model = models['omni']
    for file in tqdm(files):
        #! load one image
        filename_noext, ext = os.path.splitext(file)
        image, imgtype = cellseg_open(os.path.join(inputs_dir, file))
        raw_h, raw_w = image.shape[:2]
        image, (h_pad, w_pad) = autopad(image, stride)
        assert ( raw_h + h_pad, raw_w + w_pad) == image.shape[:2]

        image = tvF.to_tensor(image).unsqueeze(0).cuda() # HWC --> 1CHW, RGB
        h, w = image.shape[-2:]

        model = models[imgtype]
        postprocessor = Postprocessor(model.model[-1], **postprocessors_kwargs[imgtype])
        
        mask, fgConf, height = predict_one(model, image, postprocessor, window_size, sw_batch_size, overlap)
        mask = mask[:h - h_pad, :w - w_pad]

        if imgtype == 'gs':
            fgConf = fgConf[:h - h_pad, :w - w_pad]
            height = height[:h - h_pad, :w - w_pad]

            quality = 0
            if mask.max() > 0:
                quality =  fgConf[fgConf > 0.5].mean()
            omni_mask, fgConf = predict_one_omni(omni_model, image, flow_threshold, velocity, window_size, sw_batch_size, overlap)
            fgConf = fgConf[:h - h_pad, :w - w_pad]
            omni_mask = omni_mask[:h - h_pad, :w - w_pad]
            omni_quality = fgConf[fgConf > 0.5].mean()
            if quality < 0.8 or omni_quality - quality > 0.05:
                print(f'{filename_noext}: {omni_quality} || {quality}, apply omnipose')
                mask = omni_mask
            else:
                print(f'{filename_noext}: {omni_quality} || {quality}')
        mask = mask.astype(np.uint16) if mask.max() < 65536 else mask.astype(np.int32)
        tif.imwrite(os.path.join(outputs_dir, f'{filename_noext}_label.tiff'), mask, compression='zlib')

if __name__ == '__main__':
    parser = ArgumentParser()
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='./inputs', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--ckpt_folder', default='ckpts')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    auto_predict(
        args.input_path,
        args.output_path,
        models_dir=args.ckpt_folder
    )
