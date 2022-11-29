# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import math
from typing import List
from typing import Any, Dict, Optional, Tuple, List
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch_scatter
from utils.general import LOGGER, xywhn2xyxy, xyxy2xywhn
import random
from torch.utils.tensorboard.writer import SummaryWriter
from matplotlib import pyplot as plt
import os
import time

from utils.metrics import bbox_iou
from utils.plots import random_color
from utils.torch_utils import de_parallel

RANK = int(os.getenv('RANK', -1))


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.75):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False, resample=False, segment=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
            [h['obj_pw']], device=device), reduction='none' if resample else 'mean')

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        assert m.nc == 1, 'Only support single cls!'
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.resample = resample
        self.segment = segment
        self.BCEseg = nn.BCEWithLogitsLoss()

    def __call__(self, p, targets, masks=None):  # predictions, targets
        lseg = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        if masks is not None:
            *p, p_masks = p
            p_masks = F.interpolate(p_masks, masks.shape[-2:], mode='bilinear', align_corners=True)
            lseg += nn.CrossEntropyLoss()(p_masks, masks.to(p_masks.device))
        tcls, tbox, indices, anchors, segs = self.build_targets(p, targets)  # targets
        batch_size = p[0].shape[0]
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                if self.segment and masks is None:
                    pxy, pwh, _, pseg = pi[b, a, gj, gi].split(
                        (2, 2, 1 + self.nc, 400), 1)  # target-subset of predictions
                else:
                    pxy, pwh, _ = pi[b, a, gj, gi].split((2, 2, 1 + self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # per-box iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # segment
                if self.segment and masks is None:
                    lseg += self.BCEseg(pseg, segs[i])

            if self.resample:
                #! box-wise mean: divide loss by per-image gt box number
                nbox = torch.bincount(targets[:, 0].long(), minlength=batch_size)
                w = 1 / (1e-10 + nbox)
                w[nbox == 0] = 0
                obji = self.BCEobj(pi[..., 4], tobj)  # confidence/objectness is the iou of bbox
                obji = (obji.mean(dim=(1, 2, 3)) * w / w.sum()).sum()
            else:
                obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lseg *= self.hyp['seg']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lseg) * bs, torch.cat((lbox, lobj, lseg)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        if not self.segment:
            targets = targets[:, :6]
        tcls, tbox, indices, anch, segs = [], [], [], [], []
        gain = torch.ones(targets.shape[1] + 1, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
                # j = torch.ones_like(j).unsqueeze(0)
                # offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            if t.shape[1] != 7:
                raise NotImplementedError()

            #! remove outliers
            gij = (t[:, 2:4] - offsets).long()
            o = (gij[:, 0] < 0) | (gij[:, 0] >= shape[3]) | (gij[:, 1] < 0) | (gij[:, 1] >= shape[2])
            t = t[~o]

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj, gi))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch, segs


def _conv2d(mask: torch.Tensor, weight: torch.Tensor):
    kernel_size = weight.shape[-2:]
    dilation = 1
    padding = tuple(it // 2 for it in kernel_size)
    stride = 1
    h_o, w_o = mask.shape[-2:]

    wei_res = weight.reshape(weight.size(0), weight.size(1), -1).permute((1, 2, 0))  # I K*K O
    inp_unf = nn.Unfold(kernel_size=kernel_size, dilation=dilation,
                        padding=padding, stride=stride)(mask.float())  # B K*K N
    inp_unf = inp_unf.view(mask.size(0), mask.size(1), wei_res.size(1), h_o, w_o)  # B I K*K H W
    inp_unf = (inp_unf == inp_unf[:, :, [12]]).float()  # Assume [12] represents center pixel
    return torch.einsum('ijkmn,jkl->ilmn', inp_unf, wei_res)


def getOffset(instanceMask: torch.Tensor) -> torch.Tensor:
    weight = torch.Tensor(np.array([
        [-0.2, -0.1, 0, 0.1, 0.2],
        [-0.2, -0.1, 0, 0.1, 0.2],
        [-0.2, -0.1, 0, 0.1, 0.2],
        [-0.2, -0.1, 0, 0.1, 0.2],
        [-0.2, -0.1, 0, 0.1, 0.2],
    ])).to(instanceMask.device)  # 1 x 1 x 5 x 5, gx

    gx, gy = _conv2d(instanceMask, weight.reshape((1, 1, *weight.shape))
                     ), _conv2d(instanceMask, weight.T.reshape((1, 1, *weight.shape)))
    return torch.cat([gx, gy], dim=1) / 1.5


class ComputeNewLoss(ComputeLoss):

    def __call__(self, p, targets, masks=None, dts=None):
        lseg1 = torch.zeros(1, device=self.device)  # class loss
        lseg2 = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss

        if masks is not None:
            masks, bd_cells = masks
            *p, p_masks = p
            p_masks, p_height = torch.split(F.interpolate(
                p_masks, masks.shape[-2:], mode='bilinear', align_corners=True), 2, dim=1)
            p_height = p_height.sigmoid()

            lseg1 += nn.CrossEntropyLoss()(p_masks, (masks > 0).long().to(p_masks.device))
            flag = ~torch.stack([sum((m == i for i in bs), start=torch.zeros_like(m)).bool()
                                for m, bs in zip(masks, bd_cells)], dim=0)
            lseg2 += nn.MSELoss()(p_height[flag.unsqueeze(1)], dts[flag].to(p_height.device))

        tcls, tbox, indices, anchors, segs = self.build_targets(p, targets)  # targets
        batch_size = p[0].shape[0]
        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                if self.segment and masks is None:
                    pxy, pwh, _, pseg = pi[b, a, gj, gi].split(
                        (2, 2, 1 + self.nc, 400), 1)  # target-subset of predictions
                else:
                    pxy, pwh, _ = pi[b, a, gj, gi].split((2, 2, 1 + self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # per-box iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # segment
                if self.segment and masks is None:
                    lseg += self.BCEseg(pseg, segs[i])

            if self.resample:
                #! box-wise mean: divide loss by per-image gt box number
                nbox = torch.bincount(targets[:, 0].long(), minlength=batch_size)
                w = 1 / (1e-10 + nbox)
                w[nbox == 0] = 0
                obji = self.BCEobj(pi[..., 4], tobj)  # confidence/objectness is the iou of bbox
                obji = (obji.mean(dim=(1, 2, 3)) * w / w.sum()).sum()
            else:
                obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lseg1 *= self.hyp['seg']
        lseg2 *= self.hyp['seg']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lseg1 + lseg2) * bs, torch.cat((lbox, lobj, lseg1, lseg2)).detach()


class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, eps=1e-8):
        # x is a sigmoid prob tensor with any shape
        logN = math.log(float(x.shape[0]))
        neg_entropy = x * x.clamp(eps, 1).log() + (1 - x) * (1 - x).clamp(eps, 1).log()
        return - neg_entropy.mean() / logN  # normalize to [0, 1]


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, probs, targets):
        bs = targets.size(0)
        smooth = 1

        m1 = probs.view(bs, -1)
        m2 = targets.view(bs, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / bs
        return score


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, probs, dist_maps):
        return (probs * dist_maps).mean()


class ComputeDetLoss:

    # Compute losses
    def __init__(self, model, resample=False, writer=None):
        device = next(model.parameters()).device  # get model device
        self.h = model.hyp  # hyperparameters
        self._writer: SummaryWriter = writer if RANK in [-1, 0] else None
        self._int = 100
        self._count = 0

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.na = m.na  # number of anchors
        assert m.nc == 1, 'Only support single cls!'
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device
        self.sort_obj_iou = False
        self.gr = 1
        self.focus_boundary = self.h.get('focus_b', False)  # 对小物体也有一定作用
        self.focus_small = self.h.get('focus_small_object', False)
        self.add_small_weight = self.h.get('add_small_weight', False)

        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.h['obj_pw']], device=device), reduction='none')
        self.pBCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
            [self.h['pobj_pw']], device=device), reduction='none')
        self.resample = resample
        if 'seg_pw' in self.h:
            self.BCEseg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
                [self.h['seg_pw']], device=device), reduction='none')
        else:
            self.BCEseg = nn.BCEWithLogitsLoss(reduction='none')
        self.diceloss = SoftDiceLoss()  # mean
        self.bdloss = BoundaryLoss()
        self.MSE = nn.MSELoss(reduction='none')
        self.entropy = nn.BCEWithLogitsLoss()

    def __call__(self, imgs: torch.Tensor, preds: List[torch.Tensor], **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
            Loss list:
                Detection:
                    YOLOV5 Loss (remove grids on image border)

                Segmentation 
                (Ours):
                    # Segmantation
                    Semantic Mask: MSELoss
                    EDT: use OmniPose's result
        """
        masks, dists, bboxes = kwargs['masks'], kwargs['dists'], kwargs['bboxes']
        plbs = torch.zeros(imgs.shape[0], dtype=bool, device=masks.device)
        if 'plbs' in kwargs:
            plbs = kwargs['plbs']
        detect_loss, detect_lossitems, targets = self.compute_detect_loss(preds[:self.nl], bboxes, plbs)
        seg_loss, seg_lossitems = self.compute_seg_loss(preds[-2], masks, dists, plbs)
        if self._writer is not None and self._count % self._int == 0:
            self.visualize_seg(imgs, preds, masks, dists, bboxes)
        self._count += 1
        vis = {**detect_lossitems, **seg_lossitems}
        return (sum(detect_loss.values()) + sum(seg_loss.values()) * self.h['seg']) * preds[-1], vis

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image, x, y, w, h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tbox, indices, anch = [], [], []
        gain = torch.ones(targets.shape[1] + 1, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        # append anchor indices (image, x, y, w, h, indices)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[1:5] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,6)
            if nt:
                # Matches
                r = t[..., 3:5] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.h['anchor_t']  # compare
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 1:3]  # grid xy
                gxi = gain[[1, 2]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

                #! remove outliers
                gij = (t[:, 1:3] - offsets).long()
                o = (gij[:, 0] < 0) | (gij[:, 0] >= shape[3]) | (gij[:, 1] < 0) | (gij[:, 1] >= shape[2])
                t, offsets = t[~o], offsets[~o]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, gxy, gwh, a = torch.tensor_split(t, [1, 3, 5], dim=1)  # image, grid xy, grid wh, anchors
            a, (b) = a.long().view(-1), b.long().view(-1)  # anchors, image
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj, gi))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
        return tbox, indices, anch

    def compute_detect_loss(self, p, bboxes, plbs):
        targets = bboxes
        tbox, indices, anchors = self.build_targets(p, targets)  # targets

        batch_size = p[0].shape[0]
        lbox, lobj = torch.zeros([], device=p[0].device), torch.zeros([], device=p[0].device)
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pxy, pwh, _ = pi[b, a, gj, gi].split((2, 2, 1 + self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox = lbox + (1.0 - iou).mean()  # per-box iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

            if self._writer is not None and self._count % self._int == 0:
                for i in range(tobj.shape[1]):
                    fig = plt.figure()
                    plt.imshow(tobj[0, i].cpu().float().numpy())
                    self._writer.add_figure(f'gt/obj_{i}', fig, self._count)
                    fig = plt.figure()
                    plt.imshow(pi[0, i, ..., 4].detach().cpu().float().numpy())
                    self._writer.add_figure(f'pred/obj_{i}', fig, self._count)

            src, dst = pi[..., 4], tobj
            if True:
                #! box-wise mean: divide loss by per-image gt box number
                nbox = torch.bincount(targets[:, 0].long(), minlength=batch_size)
                w = 1 / (1e-10 + nbox)
                w[nbox == 0] = 0
                obji = self.BCEobj(src, dst)  # confidence/objectness is the iou of bbox
                obji[plbs] = self.pBCEobj(src, dst)[plbs] * self.h['pobj']
                obji = (obji.mean(dim=(1, 2, 3)) * w / w.sum()).sum()
            else:
                obji = self.BCEobj(src, dst)
                # obji[plbs] = self.pBCEobj(src, dst)[plbs] * self.h['pobj']
                obji = obji.mean()

            lobj = lobj + obji * self.balance[i]  # obj loss

        return {'obj': lobj * self.h['obj'], 'box': lbox * self.h['box']}, {'obj': lobj.detach().item(), 'box': lbox.detach().item()}, targets

    def compute_seg_loss(self, p, masks, dists, plbs):
        if p.shape[-2:] != masks.shape[-2:]:
            p = F.interpolate(p, masks.shape[-2:], mode='bilinear', align_corners=False)
        p_binary, p_edt = torch.tensor_split(p, [1], dim=1)
        """
            对于GT标签, 直接交叉熵&MSE

            对于伪标签, 只计算高置信部分的EDT & seg
        """
        if (~plbs).sum() > 0:
            p_binary_gt, p_edt_gt, masks_gt, dists_gt = p_binary[~plbs], p_edt[~plbs], masks[~plbs], dists[~plbs]
            binary = masks_gt > 0
            dists_gt.unsqueeze_(1)

            edt_loss = self.MSE(p_edt_gt, dists_gt)
            seg_loss = self.BCEseg(p_binary_gt, binary.unsqueeze(1).float()).mean()
            edt_loss = edt_loss.mean()

            probs = p_binary_gt.sigmoid()
            dice_loss = self.diceloss(probs, binary)
        else:
            edt_loss = torch.zeros([], device=p.device)
            seg_loss = torch.zeros([], device=p.device)
            dice_loss = torch.zeros([], device=p.device)
        boundaryloss = torch.zeros([], device=p.device)

        if plbs.sum() > 0:
            p_binary_p, p_edt_p, dists_p = p_binary[plbs], p_edt[plbs], dists[plbs]
            prob = p_binary_p.sigmoid()
            high_pos = (prob - 0.5).abs() > 0.2
            pos = (prob > 0.7).sum() / high_pos.numel()
            neg = (prob < 0.3).sum() / high_pos.numel()
            p_seg_loss = self.entropy(p_binary_p[high_pos], prob[high_pos])
            p_edt_loss = self.MSE(p_edt_p, dists_p.unsqueeze(1)).mean()
        else:
            p_edt_loss = torch.zeros([], device=p.device)
            p_seg_loss = torch.zeros([], device=p.device)
            pos = torch.zeros([], device=p.device)
            neg = torch.zeros([], device=p.device)

        frac_gt, frac_ps = (~plbs).sum() / plbs.shape[0], (plbs).sum() / plbs.shape[0]

        losses = {
            'edt': edt_loss * self.h['edt'] * frac_gt,
            'seg': seg_loss * self.h['binary'] * frac_gt,
            'dice': dice_loss * self.h['dice'] * frac_gt,
            'boundary': boundaryloss * self.h['boundary'] * frac_gt,

            'p_edt': p_edt_loss * self.h['pedt'] * frac_ps,
            'p_seg': p_seg_loss * self.h['pbinary'] * frac_ps,
        }

        loss_items = {
            'edt': edt_loss,
            'binary': seg_loss,
            'p_edt': p_edt_loss,
            'p_seg': p_seg_loss,
            'pos': pos,
            'neg': neg,
            'dice': dice_loss,
            'boundary': boundaryloss
        }
        for k in loss_items:
            loss_items[k] = loss_items[k].detach().item()
        return losses, loss_items

    def visualize_seg(self, imgs, p, masks, dists, targets):
        """
            Pred: binary_mask, p_boundary, edt, fY, fX
            GT: boundary, edt, weight, fY, fX
        """
        start = time.time()
        sample_id = 0
        targets = targets[targets[:, 0] == sample_id]
        h, w = masks.shape[-2:]
        imgs = imgs[sample_id].cpu().numpy()
        masks = masks[sample_id].cpu().numpy()
        dists = dists[sample_id].cpu().numpy()
        preds = p[-2][sample_id].detach().cpu().float().numpy()
        # 1. draw masks with BBox
        colors = random_color(masks.max() + 1)
        masks = colors[masks.astype(np.int32)]
        self._writer.add_image_with_boxes(
            'gt/imgs', imgs, xywhn2xyxy(targets[:, 1:], w, h), dataformats='CHW', global_step=self._count)
        self._writer.add_image_with_boxes(
            'gt/masks', masks, xywhn2xyxy(targets[:, 1:], w, h), dataformats='HWC', global_step=self._count)
        # 2. draw EDT, fYX, obj
        fig = plt.figure()
        plt.imshow(dists)
        self._writer.add_figure(f'gt/edt', fig, self._count)

        # 3. draw p/(EDT)
        for k, v in {'bin': 0, 'edt': 1}.items():
            fig = plt.figure()
            plt.imshow(preds[v])
            self._writer.add_figure(f'pred/{k}', fig, self._count)
        self._writer.add_scalar(f'time/plot', time.time() - start, self._count)


class DerivativeLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, Y, w, mask):
        #! This code will not be correctly runned in PyTorch 1.10.0
        #! ref: https://github.com/pytorch/pytorch/issues/67919
        #! the else branch is a workaround
        if len(y.shape) == 4:
            dim = y.shape[1]
            dims = [k for k in range(-dim, 0)]
            dy = torch.stack(torch.gradient(y, dim=dims))
            dY = torch.stack(torch.gradient(Y, dim=dims))
            return torch.mean(torch.square((dy - dY) / 5.).sum(dim=0)[mask] * w[mask])
        else:
            dims = [-1]
            dy = torch.stack(torch.gradient(y, dim=dims))
            dY = torch.stack(torch.gradient(Y, dim=dims))
            return torch.mean(torch.square((dy - dY) / 5.).sum(dim=0)[mask] * w[mask])

# I suspect that, of all the loss functions, this one would be the one that suffers most from 16 bit precision


class ArcCosDotLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, w, mask):
        """
            x, y: shape N 2 H W
            w: N H W
            mask: N H W
        """
        eps = 1e-12
        denom = torch.linalg.norm(x, dim=1) * torch.linalg.norm(y, dim=1) + eps
        dot = (x * y).sum(dim=1)
        phasediff = torch.acos(torch.clip(dot / denom, -0.999999, 0.999999)) / np.pi
        return torch.mean((torch.square(phasediff[mask])) * w[mask])


class NormLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, Y, w, mask):
        ny = torch.linalg.norm(y, dim=1, keepdim=False) / 5.
        nY = torch.linalg.norm(Y, dim=1, keepdim=False) / 5.
        diff = (ny - nY)
        return torch.mean(torch.square(diff[mask]) * w[mask])


class DivergenceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, Y, mask=None):
        divy = self.divergence(y)
        divY = self.divergence(Y)
        if mask is None:
            mask = torch.abs(divY) > 1
        diff = (divY - divy) / 5.
        return torch.mean(torch.square(diff[mask]))

    def divergence(self, y):
        axes = [k for k in range(len(y[0]))]
        dim = y.shape[1]
        return sum(torch.gradient(y[:, ax], dim=ax - dim)[0] for ax in axes)


class ComputeSegLoss:

    # Compute losses
    def __init__(self, model, resample=False, writer=None):
        device = next(model.parameters()).device  # get model device
        self.h = model.hyp  # hyperparameters
        self._writer: SummaryWriter = writer if RANK in [-1, 0] else None
        self._int = 100
        self._count = 0
        self.device = device
        self.gr = 1

        self.MSE = nn.MSELoss()
        self._omni = True
        self.BCEseg = nn.BCEWithLogitsLoss()
        self.arccosdot = ArcCosDotLoss()
        self.deriative = DerivativeLoss()
        self.divergence = DivergenceLoss()
        self.normloss = NormLoss()

    def __call__(self, imgs: torch.Tensor, preds: List[torch.Tensor], **kwargs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
            Loss list:
                Detection:
                    YOLOV5 Loss (remove grids on image border)

                Segmentation 
                (OmniPose):
                    flows: WeightedMSELoss (Loss1) ArcCosDotLoss (Loss2) DerivativeLoss (Loss3) MSELoss on Norm (Loss5) DivergenceLoss (Loss8)
                    boundary: BCELogits (Loss4)
                    edt: WeightedMSELoss (Loss6) DerivativeLoss (Loss7)
                (Ours):
                    # Segmantation
                    Semantic Mask: MSELoss
                    EDT: use OmniPose's result
        """
        masks, flows = kwargs['masks'], kwargs['flows']
        omni_loss, omni_lossitems = self.compute_omni_loss(preds, masks, flows)
        if self._writer is not None and self._count % self._int == 0:
            self.visualize_seg(imgs, preds, masks, flows)
        self._count += 1
        vis = {**omni_lossitems}
        # for k, v in omni_loss.items():
        #     vis['omni_' + k] = v.detach().item()
        return sum(omni_loss.values()) * imgs.shape[0], vis

    def compute_omni_loss(self, p, masks, flows):
        """
            Pred: p_boundary, edt, fY, fX
            GT: boundary, edt, weight, fY, fX
        """
        # p = F.interpolate(p, flows.shape[-2:], mode='bilinear', align_corners=False)
        p_boundary, p_binary, p_fYX = torch.tensor_split(p, [1, 2], dim=1)
        boundary, _, weight, fYX = torch.tensor_split(flows, [1, 2, 3], dim=1)
        binary = masks > 0
        flow_losses = {
            'WeightedMSEonFlows': torch.mean(torch.square(p_fYX - fYX) * weight) / 25,
            'ArccosDot': self.arccosdot(p_fYX, fYX, weight.squeeze(1), binary),
            'DerivativeOnBoundary': self.deriative(p_fYX, fYX, weight.repeat((1, 2, 1, 1)), binary.unsqueeze(1).repeat((1, 2, 1, 1))),
            'MSEonFlowsNorm': self.normloss(p_fYX, fYX, weight.squeeze(1), binary),
            'DivergenceLoss': self.divergence(p_fYX, fYX, binary)
        }
        flow_loss = self.h['mseflows'] * flow_losses['WeightedMSEonFlows'] + \
            self.h['arccos'] * flow_losses['ArccosDot'] + \
            self.h['donb'] * flow_losses['DerivativeOnBoundary'] + \
            self.h['msenorm'] * flow_losses['MSEonFlowsNorm'] + \
            self.h['divergence'] * flow_losses['DivergenceLoss']

        boundary_loss = self.BCEseg(p_boundary, boundary)
        edt_loss = self.BCEseg(p_binary, binary.float().unsqueeze(1))
        losses = {
            'flow': flow_loss,
            'binary': edt_loss,
            'boundary': boundary_loss * self.h['boundary']
        }
        loss_items = {**flow_losses, 'boundary': boundary_loss, 'binary': edt_loss}
        for k in loss_items:
            loss_items[k] = loss_items[k].detach().item()
        return losses, loss_items

    def visualize_seg(self, imgs, p, masks, flows):
        """
            Pred: binary_mask, p_boundary, edt, fY, fX
            GT: boundary, edt, weight, fY, fX
        """
        start = time.time()
        sample_id = 0

        imgs = imgs[sample_id].cpu().numpy()
        masks = masks[sample_id].cpu().numpy()
        flows = flows[sample_id].cpu().float().numpy()
        preds = p[sample_id].detach().cpu().float().numpy()
        # 1. draw masks with BBox
        colors = random_color(masks.max() + 1)
        masks = colors[masks.astype(np.int32)]
        self._writer.add_image('gt/imgs', imgs, dataformats='CHW', global_step=self._count)
        self._writer.add_image('gt/masks', masks, dataformats='HWC', global_step=self._count)
        # 2. draw EDT, fYX, obj

        for k, v in {'edt': 1, 'fY': 3, 'fX': 4}.items():
            fig = plt.figure()
            plt.imshow(flows[v])
            self._writer.add_figure(f'gt/{k}', fig, self._count)

        # 3. draw p/(EDT, fYX, obj)
        for k, v in {'boundary': 0, 'edt': 1, 'fY': 2, 'fX': 3}.items():
            fig = plt.figure()
            plt.imshow(preds[v])
            self._writer.add_figure(f'pred/{k}', fig, self._count)
        self._writer.add_scalar(f'time/plot', time.time() - start, self._count)


def buildBBox(masks: torch.Tensor, omit_cells: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
    """
        Given a batch of masks [shape: NxHxW], where cells are marked
        as different positive intergers and background 0

        This function generate all instance bboxes with format normalized xywh.

        ! Any cells on the border will be omitted

        Complexity: O(KHW)
    """
    dets = []
    h, w = masks.shape[-2:]
    b = masks.shape[0]
    if omit_cells is None:
        omit_cells = [None] * b
    for img_id, (mask, omit_cell) in enumerate(zip(masks, omit_cells)):
        pi_dets = []  # perimg_dets
        cell = torch.unique(mask)[1:]  # all cells
        if omit_cell is not None:
            omit_cell = omit_cell.to(mask.device)
            try:
                max_id = max(cell.max(), omit_cell.max())
            except:
                max_id = 0
            bins = torch.bincount(cell, minlength=1 + max_id) - torch.bincount(omit_cell, minlength=1 + max_id)
            cell = (bins == 1).nonzero().flatten()  # all internal cells
        for cell_id in cell:
            ys, xs = torch.where(mask == cell_id)
            pi_dets.append([img_id, xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])  # 保证左闭右开, XYXY
        if len(pi_dets):
            det = torch.tensor(pi_dets, device=mask.device).float()
            det[:, 1:] = xyxy2xywhn(det[:, 1:], h, w)
            dets.append(det)
    return torch.cat(dets, dim=0)
