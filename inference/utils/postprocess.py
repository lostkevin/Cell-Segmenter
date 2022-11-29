import torch
import torchvision
import cv2
import numpy as np

from skimage import morphology, segmentation, measure
import edt
from scipy.ndimage import find_objects


from models.yolo import SegDetect
from utils.general import check_version, xywh2xyxy
from utils.mask import fill_holes_and_remove_small_masks

class Postprocessor:
    def __init__(self, detect: SegDetect, *, conf_thres=0.2, iou_thres=0.6) -> None:
        self.anchors = detect.anchors  # na 2 Tensor
        self.stride = detect.stride  # scalar
        self.nl = detect.nl
        self.na = detect.na
        self.no = detect.no
        self._mesh = {}
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def postprocess(self, pred):  # ! pred is a tuple of N C H W tensor
        na, no = self.na, self.no
        z = []  # output
        for i in range(self.nl):
            bs, _, ny, nx = pred[i].shape  # x(bs,6,20,20) to x(bs,1,20,20,6)
            p = pred[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if (ny, nx, i) not in self._mesh:
                d = self.anchors.device
                t = self.anchors.dtype
                shape = 1, na, ny, nx, 2  # grid shape
                y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
                if check_version(torch.__version__, '1.10.0'):  # torch>=1.10.0 meshgrid workaround for torch>=0.7 compatibility
                    yv, xv = torch.meshgrid(y, x, indexing='ij')
                else:
                    yv, xv = torch.meshgrid(y, x)
                grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
                anchor_grid = (self.anchors[i] * self.stride[i]).view((1, na, 1, 1, 2)).expand(shape)
                self._mesh[(ny, nx, i)] = grid, anchor_grid
            grid, anchor_grid = self._mesh[(ny, nx, i)]
            y = p.sigmoid()
            #! always inplace
            y[..., 0:2] = (y[..., 0:2] * 2 + grid) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
            z.append(y)

        return (torch.cat(z, 1), pred[self.nl:]) if len(pred) > self.nl else (torch.cat(z, 1), None)

    def run(self, pred):
        det, seg = self.postprocess(pred)
        return self.nms(det), seg

    def _nms(self, det: torch.Tensor, return_final: bool=True):
        """
            Given det with shape [KxD], apply nms and return index,

            if return_final is False, return x[f][i] (Kx6) instead of final dets [Kx5]
        """
        xc = det[..., 4] > self.conf_thres
        x = det[xc]

        conf = x[:, 5:] * x[:, 4:5]  # K x 1, objectness * predicted_iou
        box = xywh2xyxy(x[:, :4])  # K x 4

        #! filter by conf
        conf = conf.view(-1)
        f = conf > self.conf_thres
        box, conf = box[f], conf[f]
        if len(conf) > 0:
            rank = conf.argsort(descending=True)
            box, conf = box[rank], conf[rank]

            i = torchvision.ops.nms(box, conf, self.iou_thres)  # NMS
            if return_final:
                return torch.cat([box[i], conf[i].reshape(-1, 1)], dim=1) #! shape: Nx5, xyxy
            return x[f][i] #! shape: Nx6, xywh
        else:
            if return_final:
                return torch.zeros([0, 5], device=det.device) #! shape: Nx5, xyxy
            return torch.zeros([0, 6], device=det.device) #! shape: Nx6, xywh

    def nms(self, det: torch.Tensor):
        """
            Non-Maximum Suppression

            Input and Output shape: [N x na * nx * ny x D] where K is the amount of Dets
            for each image,
                Input: shape [K x (5 + 1 + seg)]---[xywh, obj_conf, cls1_conf, seg]
                Output: shape [K x (5 + seg)]---[xyxy, conf]

            ! always assume nc = 1 
            ! conf = obj_conf * cls_conf
            ! Only process boxes with objectness > conf_thres and conf > conf_thres
        """
        MAX_DETS = 20000
        bs = det.shape[0]  # batch size
        assert bs == 1
        if det.shape[-1] != 6:
            raise NotImplementedError()
        
        #! filter by objectness
        xc = det[..., 4] > self.conf_thres
        dets_size = xc.sum()
        if dets_size <= MAX_DETS:
            # do nms straightforwardly
            return [self._nms(det.view(-1, 6))] #! bs na nx ny no => -1 no
        
        nx, ny, no = det.shape[2:]
        PATCH = 64 # REGION: 512 * 512
        kx, ky = int(np.ceil(nx / PATCH)), int(np.ceil(ny / PATCH))
        first_nms = []
        for i in range(kx):
            for j in range(ky):
                first_nms.append(self._nms(det[..., i * PATCH: (i + 1) * PATCH, j * PATCH: (j + 1) * PATCH, :].reshape(-1, no), False))
        det = torch.cat(first_nms, dim=0)
        return [self._nms(det)]

    def filter_with_ioa(self, det: torch.Tensor, threshold=0.8, eps=1e-7):
        """
            ioa filter

            Input and Output shape: [N x K x D] where K is the amount of Dets
            for each image,
                Input: shape [K x (5 + 1 + seg)]---[xywh, obj_conf, cls1_conf, seg]
                Output: shape [K x (5 + seg)]---[xyxy, conf]

            ! ensure threshold >= 2 / (1 / nms_iou_T + 1) to avoid two both ioa of two dets larger than threshold

            # nms_iou_T = 0.35 ==> threshold >= 0.5
        """
        def box_area(box): return (box[2] - box[0]) * (box[3] - box[1])

        (a1, a2), (b1, b2) = det[:, None, :4].chunk(2, 2), det[:, :4].chunk(2, 1)  # N 1 4 ==> N1(22) N4=>N(22)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
        ioa_matrix = inter / (box_area(det.T)[:, None] + eps)
        # remove diag
        ioa_matrix -= torch.diag_embed(torch.diag(ioa_matrix))
        removed_box = (ioa_matrix > threshold).sum(1) > 0
        # ideally, (ioa_matrix > threshold).sum(1).max() <= 1
        return det[~removed_box]

    def generate_mask(self, fgConf, height, dets, conf, imgtype=None, dtype=np.uint16, skip_low_confd=False, debug=False):
        """
            mask: a [2, H, W] tensor to determine where is cell pixel (semantic fgConf and normed euclid distance)
            dets: many xyxy bboxes represents cells

            #! always assume one detection box have only one cell
            #! first label non-overlapped pixels with floodFill, then label other pixels with watershed
        """
        fgConf, height = fgConf.cpu().numpy(), height.cpu().numpy()
        if len(height.shape) == 3:
            if height.shape[0] != 1:
                raise ValueError()
            height = height[0]

        fgMask = fgConf > 0.5  # can apply two threshold
        h, w = fgMask.shape
        dets = np.clip(dets.astype(int), 0, [[w, h, w, h]])

        results = np.zeros_like(fgMask, dtype=np.float32)  # 24 valid bits ~~ np.uint24
        for ptr in range(len(dets)):
            d = dets[ptr]
            center = (d[0] + d[2]) // 2, (d[1] + d[3]) // 2  # XY
            cv2.circle(results, center, 2, color=ptr + 1, thickness=-1)

        # T1 defines borders' pixel, increase T1 leading to more split result and remove many edge det points
        T1 = 0.80
        T2 = 1.25
        # if imgtype == 'fl':
        results[height < T1] = 0
        results[~fgMask] = 0
        #! In the second step, we want to fill all unlabeled fg pixels with watershed algorithm
        unlabeled = np.ones_like(fgMask, dtype=np.int32)
        unlabeled[height < T1] = 0
        mask1 = segmentation.watershed(-height, results, mask=unlabeled)  # cell from det

        unlabeled = np.ones_like(fgMask, dtype=np.int32)
        unlabeled[height < T1] = 0
        unlabeled[mask1 > 0] = 0
        marker_edt = height >= T2
        marker_edt[mask1 > 0] = 0
        morphology.remove_small_objects(marker_edt, 5, out=marker_edt)
        mask2 = segmentation.watershed(-height, measure.label(marker_edt,
                                        connectivity=2), mask=unlabeled)  # cell from edt
        mask2[mask2 > 0] += mask1.max()
        mask = mask1 + mask2
        unlabeled = np.ones_like(fgMask, dtype=np.int32)
        unlabeled[~fgMask] = 0
        mask = segmentation.watershed(-height, mask, mask=unlabeled)

        mask = segmentation.relabel_sequential(mask)[0]
        threshold = 10
        if imgtype == 'gs':
            areas = np.bincount(mask.flatten())[1:]
            threshold = min(areas.mean() / 5, 150)

        mask = fill_holes_and_remove_small_masks(mask, threshold, 10)
        return mask.astype(np.int32)

    def refine(self, mask, p_dist, threshold=0.4):
        # generate edt based on predict mask, then compute loss
        # remove all masks with loss larger than threshold

        dt = edt.edt(mask)
        slices = find_objects(mask)
        cnt = 0
        for i, slc in enumerate(slices):
            if slc is not None:     
                msk = mask[slc] == (i + 1)   
                data = dt[slc][msk]
                data /= np.median(data) + 1  # gt
                edt_loss = ((p_dist[slc][msk] - data)**2).mean()
                if edt_loss > threshold:
                    mask[slc][msk] = 0
                    cnt += 1
        return segmentation.relabel_sequential(mask)[0], cnt

