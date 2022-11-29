from utils.mask import compute_masks
import os
from monai.inferers import sliding_window_inference
from argparse import ArgumentParser
from torch import nn
from torch.utils.data.dataloader import DataLoader
import torch
from types import MethodType
import numpy as np
import cv2
import tifffile as tif
from tqdm import tqdm
from typing import List, Optional, Dict
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
from compute_metric import generate_match
import edt
from scipy.ndimage import find_objects


from models.experimental import attempt_load
from models.yolo import SegDetect
from utils.general import LOGGER, xywh2xyxy
from utils.plots import Annotator, random_color
from utils.dataloaders import CellSegEvalDataset
from utils.postprocess import Postprocessor

_DEBUG = True

def detect_sw_forward(self, x):
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
    if hasattr(self, 'seg') and self.seg is not None:
        x[-1] = self.seg(x[-1])
    return tuple(x)


def detect_sw_forward_new(self, x: torch.Tensor):
    # todo: test phase augment
    x = x[:self.nl + 1]
    # detect
    for i in range(self.nl):
        x[i] = self.m[i](x[i])  # conv
    # seg, only extract semantic map and edt
    x[-1] = self.seg(x[-1])[:, [0, 2]]

    return tuple(x)


def detect_sw_forward_omni(self, x):
    x = x[:self.nl + 1]
    for i in range(self.nl):
        x[i] = self.m[i](x[i])
    x[-1] = self.seg(x[-1])
    return tuple(x)


def parse_opt():
    parser = ArgumentParser()
    # Dataset parameters
    parser.add_argument('-i', '--input_path', default='./inputs', type=str,
                        help='training data path; subfolders: images, labels')
    parser.add_argument("-o", '--output_path', default='./outputs', type=str, help='output path')
    parser.add_argument('--ckpt', default=None, required=True)
    parser.add_argument('--seg_ckpt', default=None)
    # Model parameters
    parser.add_argument('--input_size', default=640, type=int, help='segmentation classes')
    parser.add_argument('--overlap', default=0.125, type=float, help='overlap factor')
    parser.add_argument('--naive_seg', default=False, action='store_true')
    parser.add_argument('--omni', default=False, action='store_true')
    parser.add_argument('--cls', default=False, action='store_true')
    parser.add_argument('--conf', default=0.15, type=float, help='overlap factor')
    parser.add_argument('--iou_thres', default=0.25, type=float, help='overlap factor')
    parser.add_argument('--plb', default=False, action='store_true')
    parser.add_argument('--only_label', default=False, action='store_true')
    parser.add_argument('--remove_bad', default=False, action='store_true')
    args = parser.parse_args()
    return args


@torch.no_grad()
def predict_generator(
    model: nn.Module,
    bs1_valloader: DataLoader,
    window_size: int = 640,
    sw_batch_size: int = 4,
    conf: float = 0.25,
    iou_thres: float = 0.2,
    *,
    timer: bool = False,
    sw_mode: str = 'gaussian',
    use_default: bool = True,
    skip_low_conf: bool = False,
    show=True,
    remove_holes=False,
):

    # #! check batch size
    # if bs1_valloader.batch_size > 1:
    #     LOGGER.error('Predict function only support dataloader with batch size 1, try to reset...(may cause bug)')
    #     bs1_valloader.batch_size = 1
    # if batch_size > 1, pad to same size
    try:
        overlap = getattr(bs1_valloader.dataset, 'overlap')
    except:
        LOGGER.warning('dataset object do not have overlap attribute, maybe wrong dataloader...')
        overlap = 0.125

    device = next(model.parameters()).device

    #! model initialization
    raw_forward_method = model.model[-1].forward
    if isinstance(model.model[-1], SegDetect):
        model.model[-1].forward = MethodType(detect_sw_forward, model.model[-1])
    else:
        model.model[-1].forward = MethodType(detect_sw_forward_new, model.model[-1])
    if use_default:
        # ~0.3% IMPROVEMENT on the Labeled Set
        postprocessors = {
            'bf': Postprocessor(model.model[-1], conf_thres=0.1, iou_thres=0.25),
            'gs': Postprocessor(model.model[-1], conf_thres=0.2, iou_thres=0.25),
            'fl': Postprocessor(model.model[-1], conf_thres=0.25, iou_thres=0.35),
        }
    else:
        postprocessors = {
            'bf': Postprocessor(model.model[-1], conf_thres=conf, iou_thres=iou_thres),
            'gs': Postprocessor(model.model[-1], conf_thres=conf, iou_thres=iou_thres),
            'fl': Postprocessor(model.model[-1], conf_thres=conf, iou_thres=iou_thres),
        }

    #! model warmup
    window_size = (window_size, window_size)
    # model(torch.zeros(1, 3, *window_size, device=device))
    if show:
        bs1_valloader = tqdm(bs1_valloader)
    for data_tup in bs1_valloader:
        filename_noext, image, pad = data_tup[:3]
        imgtype = data_tup[-1][0]

        filename_noext = filename_noext[0]
        image = image.to(device)

        # TODO: branch
        # pad to N x M
        import time
        pred = sliding_window_inference(image, window_size, sw_batch_size, model,
                                        mode=sw_mode, overlap=overlap)  # ! 滑窗方法将影响检测的性能
        h, w = image.shape[-2:]
        # apply NMS
        postprocessor = postprocessors[imgtype]
        raw_dets, segs = postprocessor.postprocess(pred)
        t = time.time()
        dets = postprocessor.nms(raw_dets)
        t1 = time.time()
        # print(f'nms in {t1 - t} secs')

        det, seg = dets[0], None
        if len(det) < 2000:
            det = postprocessor.filter_with_ioa(det)
        if segs is None:
            raise ValueError('No segmentation!')
        seg = segs[0]
        if segs[0].shape[-2:] != (h, w):
            seg = F.interpolate(segs[0], (h, w), mode='bilinear', align_corners=True)

        seg = seg[0]
        if seg.shape[0] == 3:
            pred, height = torch.split(seg, 2)
            fgConf = pred.softmax(dim=0)[1]
        else:
            fgConf, height = seg
            fgConf = fgConf.sigmoid()  # ! FIX
        det = det.cpu().numpy()
        # if isinstance(model.model[-1], SegDetect):
        # height = height.sigmoid()
        mask = postprocessor.generate_mask(fgConf, height, det[:, :4], det[:, -1], skip_low_confd=skip_low_conf, imgtype=imgtype)
        t2 = time.time()
        # print(f'build in {t2 - t1} secs')
        # mask, markers = postprocessor.generate_mask_based_on_cluster(fgConf, height, det[:, :4], det[:, -1])

        mask = mask[:h - int(pad[0]), :w - int(pad[1])]
        if _DEBUG:
            height = height[:h - int(pad[0]), :w - int(pad[1])].cpu().numpy()
            fgConf = fgConf[:h - int(pad[0]), :w - int(pad[1])].cpu().numpy()

        mask = mask.astype(np.int32) if mask.max() > 65535 else mask.astype(np.uint16)
        yield filename_noext, mask, (data_tup[3][0] if len(data_tup) > 4 else None), imgtype, \
            {
                'raw_dets': raw_dets[0].flatten(0, -2).cpu().numpy(),
                'dets': det,
                'edt': height,
                'seg': fgConf,
                # 'markers': markers

        } if _DEBUG else {}  # debuginfo

    model.model[-1].forward = raw_forward_method
    model.float()

def check_quality(mask, height, fgConf):
    if mask.max() == 0:
        return False
    dist = edt.edt(mask)
    slices = find_objects(mask)
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = mask[slc] == i + 1
            dist[slc][msk] = dist[slc][msk] / (np.median(dist[slc][msk]) + 1)

    loss1 = ((height - dist)**2)[mask > 0].mean()
    loss2 = fgConf[fgConf > 0.5].mean()
    return loss1 <= 0.3 and loss2 >= 0.85


def predict_new(
    model: nn.Module,
    bs1_valloader: DataLoader,
    window_size: int = 640,
    sw_batch_size: int = 4,
    conf: float = 0.25,
    iou_thres: float = 0.2,
    *,
    timer: bool = False,
    visualize: bool = False,
    visualize_path: Optional[str] = None,
    sw_mode: str = 'gaussian',
    skip_low_confidence: bool = False,
    multi_threading=False,
    show=True,
    only_label=False,
    remove_bad=False
):
    if visualize and visualize_path is None:
        LOGGER.warning('Empty visualize_path encountered... set visualize to False...')
        visualize = False

    if visualize_path is not None:
        os.makedirs(visualize_path, exist_ok=True)

    generator = predict_generator(model, bs1_valloader, window_size, sw_batch_size, timer=timer,
                                  sw_mode=sw_mode, use_default=True, skip_low_conf=skip_low_confidence, show=show, conf=conf, iou_thres=iou_thres)
    for filename_noext, mask, gt, _, debug_infos in generator:

        if os.path.isdir(visualize_path):
            if remove_bad and not check_quality(mask, debug_infos['edt'], debug_infos['seg']):
                tif.imwrite(os.path.join(visualize_path, f'{filename_noext}_label.tiff'), np.zeros_like(
                    mask), compression='zlib')
                LOGGER.info(f'skip {filename_noext}')
            else:
                tif.imwrite(os.path.join(visualize_path, f'{filename_noext}_label.tiff'), mask, compression='zlib')
            if not only_label:
                colors = random_color(max((gt.max() if hasattr(gt, 'max') else 0), mask.max()) + 1)
                cv2.imwrite(os.path.join(visualize_path, f'{filename_noext}_color.png'), colors[mask])
                height = (debug_infos['edt'].clip(0, 1) * 255).astype(np.uint8)
                fgConf = (debug_infos['seg'] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(visualize_path, filename_noext + f'_fgConf.png'), fgConf)

                # visualize dets center
                dets, conf = debug_infos['raw_dets'][:, :4], debug_infos['raw_dets'][:, 4]
                dets = dets[conf > 0.25]
                dets = xywh2xyxy(dets)
                center = [(dets[:, 0] + dets[:, 2]) // 2, (dets[:, 1] + dets[:, 3]) // 2]  # N * 2, XY
                # draw points
                out = np.stack([height, height, height], axis=-1)
                x, y = center[0].astype(np.int32).clip(
                    0, out.shape[1] - 1), center[1].astype(np.int32).clip(0, out.shape[0] - 1)
                # x, y = debug_infos['markers'].T
                out[y, x, :] = [0, 0, 255]
                cv2.imwrite(os.path.join(visualize_path, filename_noext + f'_centers.png'), out)

                if not check_quality(mask, debug_infos['edt'], debug_infos['seg']):
                    print(f'{filename_noext} need omnipose')

                if hasattr(gt, 'max'):
                    h, w = gt.shape[-2:]
                    gt = gt.numpy()
                    v = [('det', colors[gt].astype(np.uint8))]
                    for name, im in v:
                        annotator = Annotator(im, line_width=1, example='', font_size=6, pil=True)
                        det = debug_infos['dets'][..., :5]
                        np.clip(det[..., [0, 2]], 0, w, det[..., [0, 2]])
                        np.clip(det[..., [1, 3]], 0, h, det[..., [1, 3]])
                        for *xyxy, conf in det:
                            annotator.box_label(xyxy, f'{conf:.2f}', color=(255, 0, 0))
                        annotator.im.save(os.path.join(visualize_path, filename_noext + f'_{name}.png'))

                    # visualize all wrong match
                    matches = generate_match(gt, mask)
                    # missed cells
                    gt[np.isin(gt, matches[:, 0])] = 0
                    cv2.imwrite(os.path.join(visualize_path, f'{filename_noext}_missed.png'), colors[gt])
                    # over segmented cells
                    mask[np.isin(mask, matches[:, 1])] = 0
                    cv2.imwrite(os.path.join(visualize_path, f'{filename_noext}_wrong.png'), colors[mask])
                else:
                    h, w = mask.shape[-2:]
                    v = [('det', colors[mask].astype(np.uint8))]
                    for name, im in v:
                        annotator = Annotator(im, line_width=1, example='', font_size=6, pil=True)
                        det = debug_infos['dets'][..., :5]
                        np.clip(det[..., [0, 2]], 0, w, det[..., [0, 2]])
                        np.clip(det[..., [1, 3]], 0, h, det[..., [1, 3]])
                        for *xyxy, conf in det:
                            annotator.box_label(xyxy, f'{conf:.2f}', color=(255, 0, 0))
                        annotator.im.save(os.path.join(visualize_path, filename_noext + f'_{name}.png'))                  


if _DEBUG:
    predict = predict_new
else:
    predict = predict_detect


@torch.no_grad()
def predict_generator_omni(
    model: nn.Module,
    bs1_valloader: DataLoader,
    window_size: int = 640,
    sw_batch_size: int = 4,
    *,
    timer: bool = False,
    sw_mode: str = 'gaussian',
    flow_threshold=0.4,
    velocity=1.5,
    ddir=None
):

    #! check batch size
    if bs1_valloader.batch_size > 1:
        LOGGER.error('Predict function only support dataloader with batch size 1, try to reset...(may cause bug)')
        bs1_valloader.batch_size = 1
    try:
        overlap = getattr(bs1_valloader.dataset, 'overlap')
    except:
        LOGGER.warning('dataset object do not have overlap attribute, maybe wrong dataloader...')
        overlap = 0.125

    device = next(model.parameters()).device
    #! model warmup
    window_size = (window_size, window_size)
    model(torch.zeros(1, 3, *window_size, device=device))
    LOGGER.info(f'use velocity {velocity}')
    kwargs = {'debug_dir': ddir} if ddir is not None else {}
    for data_tup in tqdm(bs1_valloader):
        filename_noext, image, pad, _ = data_tup[:4]
        filename_noext = filename_noext[0]
        image = image.to(device)

        pred = sliding_window_inference(image, window_size, sw_batch_size, model,
                                         mode=sw_mode, overlap=overlap)  # ! 滑窗方法将影响检测的性能
        h, w = image.shape[-2:]
        bd, dist, flows = torch.tensor_split(pred[0], [1, 2], dim=0)
        mask, p, trace = compute_masks(
            flows, dist[0], bd[0], filename_noext=filename_noext, flow_threshold=flow_threshold, velocity=velocity, **kwargs)

        yield filename_noext, mask[:h - int(pad[0]), :w - int(pad[1])], data_tup[3][0] if len(data_tup) > 3 else None
    model.float()


def predict_omni(
    model: nn.Module,
    bs1_valloader: DataLoader,
    window_size: int = 640,
    sw_batch_size: int = 4,
    *,
    timer: bool = False,
    sw_mode: str = 'gaussian',
    visualize_path=None
):
    if visualize_path is not None:
        os.makedirs(visualize_path, exist_ok=True)

    generator = predict_generator_omni(model, bs1_valloader, window_size, sw_batch_size,
                                       timer=timer, sw_mode=sw_mode, ddir=visualize_path)
    for filename_noext, mask, _ in generator:

        colors = random_color(mask.max() + 1)
        if os.path.isdir(visualize_path):
            cv2.imwrite(os.path.join(visualize_path, f'{filename_noext}_color.png'), colors[mask])
            tif.imwrite(os.path.join(visualize_path, f'{filename_noext}_label.tiff'), mask, compression='zlib')
        pass

def main(args):
    # model initialization
    model: nn.Module = attempt_load(args.ckpt, fuse=False).eval().cuda()
    # model.model[-1].forward = MethodType(detect_sw_forward, model.model[-1]) # replace detect forward method
    # preprocessing
    imageset = CellSegEvalDataset(args.input_path, args.overlap, False)
    if args.omni:
        # imageset.images = ['cell_00142.tif','cell_00143.tif', 'cell_00144.tif', 'cell_00528.tif', 'cell_00529.tif','cell_00547.tif', 'cell_00548.tif',]
        pass
    # imageset.images = [i for i in imageset.images if '00548' in i]
    imageloader = DataLoader(imageset, batch_size=1, num_workers=8)
    if args.omni:
        predictions = predict_omni(model,
                                   imageloader,
                                   args.input_size,
                                   visualize_path=args.output_path)
    else:
        predictions = predict(model,
                              imageloader,
                              args.input_size,
                              conf=args.conf,
                              iou_thres=args.iou_thres,
                              visualize=True,
                              visualize_path=args.output_path,
                              skip_low_confidence=args.plb, only_label=args.only_label, remove_bad=args.remove_bad)

if __name__ == '__main__':
    args = parse_opt()
    os.makedirs(args.output_path, exist_ok=True)
    main(args)
