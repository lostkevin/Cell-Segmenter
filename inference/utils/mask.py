from typing import Optional
import torch
import numpy as np
from skimage import measure
from torch.nn import functional as F
import torch_scatter
from skimage.morphology import remove_small_holes
from scipy.ndimage import find_objects

from utils.omnipose import masks_to_flows, safe_divide
from utils.torch_helpers import interpolate


def step_factor(t):
    return (1 + t)


def normalize99(
        Y: torch.Tensor,
        lower=0.0001,
        upper=0.9999):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile 
    Upper and lower percentile ranges configurable. 

    Parameters
    ----------
    Y: ndarray, float
        Component array of lenth N by L1 by L2 by ... by LN. 
    upper: float
        upper percentile above which pixels are sent to 1.0

    lower: float
        lower percentile below which pixels are sent to 0.0

    Returns
    --------------
    normalized array with a minimum of 0 and maximum of 1

    """
    return interpolate(Y, torch.quantile(Y, torch.tensor([lower, upper], device=Y.device)), torch.tensor([0., 1.], device=Y.device))


def div_rescale(dP, mask):
    """
    Normalize the flow magnitude to rescaled 0-1 divergence. 

    Parameters
    -------------
    dP: [2 H W], ND array
        flow field 
    mask: int, ND array
        label matrix

    Returns
    -------------
    dP: rescaled flow field
    """
    dP = mask * dP
    dP = safe_divide(dP, torch.sqrt(torch.nansum(dP**2, dim=0)))
    return dP * normalize99(divergence(dP))


def divergence(
        f: torch.Tensor,
        sp: Optional[torch.Tensor] = None):
    """ Computes divergence of vector field

    Parameters
    -------------
    f: [2 H W], float
        vector field components [Fx,Fy,Fz,...]
    sp: ND array, float
        spacing between points in respecitve directions [spx, spy, spz,...]

    f = f(x, y)
    div f = \frac{\parital f_x}{\partial x} + \frac{\parital f_y}{\partial y} 
    """
    return sum((torch.gradient(f[i], dim=(i,))[0] for i in range(f.shape[0])))


def get_masks(p, bd, dist, mask, inds, cluster=False, prefix=''):
    """Omnipose mask recontruction algorithm.

    This function is called after dynamics are run. The final pixel coordinates are provided, 
    and cell labels are assigned to clusters found by labelling the pixel clusters after rounding
    the coordinates (snapping each pixel to the grid and labelling the resulting binary mask) or 
    by using DBSCAN or HDBSCAN for sub-pixel clustering. 

    Parameters
    -------------
    p: float32, ND array
        final locations of each pixel after dynamics
    bd: float, ND array
        boundary field
    dist: float, ND array
        distance field
    mask: bool, ND array
        binary cell mask
    inds: int, ND array 
        initial indices of pixels for the Euler integration [npixels x ndim]
    nclasses: int
        number of prediciton classes

    Returns
    -------------
    mask: int, ND array
        label matrix
    labels: int, list
        all unique labels 
    """
    cell_px = tuple(inds)
    newinds = p[(Ellipsis,) + cell_px]
    mask = torch.zeros(p.shape[1:], dtype=torch.long, device=p.device)

    newinds = newinds.round_().long()
    new_px = tuple(newinds)

    # 原后处理: 对于diam < threshold的情况使用cluster, 反之对更新后的流使用连通域算法
    if False:
        #! 聚类能提高一些效果, 但消耗时间增加
        newinds = newinds.cpu().numpy().T
        from sklearn.cluster import DBSCAN
        from sklearn.neighbors import NearestNeighbors
        clusterer = DBSCAN(eps=2 * 2**0.5, min_samples=50, n_jobs=-1)
        clusterer.fit(newinds)
        labels = clusterer.labels_
        nearest_neighbors = NearestNeighbors(n_neighbors=50)
        neighbors = nearest_neighbors.fit(newinds)
        o_inds = np.where(labels == -1)[0]
        if len(o_inds) > 1:
            outliers = [newinds[i] for i in o_inds]
            distances, indices = neighbors.kneighbors(outliers)
            ns = labels[indices]
            l = [n[np.where(n != -1)[0][0] if np.any(n != -1) else 0] for n in ns]
            labels[o_inds] = l

        mask[cell_px] = torch.from_numpy(labels + 1).to(mask.device)  # outliers have label -1
    else:
        skelmask = torch.zeros_like(dist, dtype=bool)
        skelmask[new_px] = 1
        # disconnect skeletons at the edge, 5 pixels in
        # border_mask = torch.zeros_like(skelmask, dtype=bool)
        # border_px = torch.zeros_like(skelmask)
        # border_mask[5:-5, 5:-5] = 0
        # border_px[border_mask] = skelmask[border_mask]
        # can use boundary to erase joined edge skelmasks
        # border_px[bd > -1] = 0
        # skelmask[border_mask] = border_px[border_mask]
        # split cell with border
        skelmask[bd > 0] = 0
        # skelmask = binary_opening(skelmask.cpu().numpy(), border_value=0, iterations=3)
        labels = measure.label(skelmask.cpu().numpy(), connectivity=skelmask.ndim)
        labels = torch.from_numpy(labels).long().to(mask.device)
        mask[cell_px] = labels[new_px]
    return mask, labels


def steps_interp(p: torch.Tensor, flows: torch.Tensor, niter, calc_trace=False, diam=None, velocity=1):
    """Euler integration of pixel locations p subject to flow dP for niter steps in N dimensions. 

    Parameters
    ----------------
    p: float32, ND array
        pixel locations [2 H W] (start at initial meshgrid)
    flows: float32, [2 H w]
    niter: number of iterations of dynamics to run

    Returns
    ---------------
    p: float32, ND array
        final locations of each pixel after dynamics

    """
    FACTOR = 12  # small factor will lead to fragment (in long cell), and reduce
    if diam is None:
        diam = FACTOR
    align_corners = True
    mode = 'bilinear'
    d, *shape = flows.shape  # 2, (H, W)
    inds = list(range(d))[::-1]  # grid_sample requires a particular ordering
    shape = np.array(shape)[inds] - 1.  # dP is d.Ly.Lx, inds flips this to flipped X-1, Y-1, ...

    # for grid_sample to work, we need im,pt to be (N,C,H,W),(N,H,W,2) or (N,C,D,H,W),(N,D,H,W,3). The 'image' getting interpolated
    # is the flow, which has d=2 channels in 2D and 3 in 3D (d vector components). Output has shape (N,C,H,W) or (N,C,D,H,W)
    pt = p[inds].T.double()
    for k in range(d):
        pt = pt.unsqueeze(0)  # get it in the right shape
    flow = flows[inds].double().unsqueeze(0)

    # we want to normalize the coordinates between 0 and 1. To do this,
    # we divide the coordinates by the shape along that dimension. To symmetrize,
    # we then multiply by 2 and subtract 1. I
    # We also need to rescale the flow by the same factor, but no shift of -1.

    for k in range(d):
        pt[..., k] = 2 * pt[..., k] / shape[k] - 1
        flow[:, k] = 2 * flow[:, k] / shape[k]

    # make an array to track the trajectories
    if calc_trace:
        trace = torch.clone(pt).detach()

    # init
    dPt0 = F.grid_sample(flow, pt, mode=mode, align_corners=align_corners)
    # here is where the stepping happens
    for t in range(niter):
        if calc_trace:
            trace = torch.cat((trace, pt))
            # trace[t] = pt.detach()
        # align_corners default is False, just added to suppress warning
        dPt = F.grid_sample(flow, pt, mode=mode, align_corners=align_corners)  # see how nearest changes things
        # here is where I could add something for a potential, random step, etc.

        dPt = (dPt + dPt0) / 2.  # average with previous flow
        dPt0 = dPt.clone()  # update old flow
        dPt *= diam / FACTOR * velocity
        dPt /= t + 1

        for k in range(d):  # clamp the final pixel locations
            pt[..., k] = torch.clamp(pt[..., k] + dPt[:, k], -1., 1.)

    # undo the normalization from before, reverse order of operations
    pt = (pt + 1) * 0.5
    for k in range(d):
        pt[..., k] *= shape[k]

    if calc_trace:
        trace = (trace + 1) * 0.5
        for k in range(d):
            trace[..., k] *= shape[k]
    tr = None
    if calc_trace:
        tr = trace[..., inds].squeeze().T
    p = pt[..., inds].squeeze().T
    return p, tr


def follow_flows(flows: torch.Tensor, inds: torch.Tensor, niter=200, calc_trace=False, diam=None, velocity=1):
    """ define pixels and run dynamics to recover masks in 2D

    Pixels are meshgrid. Only pixels with non-zero cell-probability
    are used (as defined by inds)

    Parameters
    ----------------
    flows: float32, [2 H W] array
    inds: int, ND array 
        initial indices of pixels for the Euler integration 
    niter: int 
        number of iterations of dynamics to run
    calc_trace: bool 
        flag to store and retrun all pixel coordinates during Euler integration (slow)

    Returns
    ---------------
    p: float32, ND array
        final locations of each pixel after dynamics
    inds: int, ND array
        initial indices of pixels for the Euler integration [npixels x ndim]
    tr: float32, ND array
        list of intermediate pixel coordinates for each step of the Euler integration

    """
    d, *shape = flows.shape  # 2, (H, W)
    device = flows.device
    grid = [torch.arange(shape[i], device=device) for i in range(d)]
    p = torch.stack(torch.meshgrid(*grid, indexing='ij')).to(device).double()

    if inds.ndim < 2 or inds.shape[0] < d:
        # added inds for debugging while preserving backwards compatibility
        print(inds.shape, d)
        return p, inds, None

    cell_px = (Ellipsis,) + tuple(inds)
    p_interp, tr = steps_interp(p[cell_px], flows, niter, calc_trace=calc_trace, diam=diam, velocity=velocity)
    # print(f'average move distance {torch.norm(p_interp - p[cell_px], p=2, dim=0).mean()}')
    p[cell_px] = p_interp
    return p, inds, tr


def remove_bad_flow_masks(
        masks: torch.Tensor,
        flows: torch.Tensor,
        threshold=0.4):
    """ remove masks which have inconsistent flows 

    Uses metrics.flow_error to compute flows from predicted masks 
    and compare flows to predicted flows from network. Discards 
    masks with flow errors greater than the threshold.

    Parameters
    ----------------
    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    flows: float, 3D or 4D array
        flows [axis x Ly x Lx] or [axis x Lz x Ly x Lx]
    threshold: float
        masks with flow error greater than threshold are discarded

    Returns
    ---------------
    masks: int, 2D or 3D array
        masks with inconsistent flow masks removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """
    dP_masks = masks_to_flows(masks, return_tensor=True)[-2:].to(flows.device)
    merrors = torch_scatter.scatter_mean((torch.norm(dP_masks - flows, p=2, dim=0)).view(-1), masks.view(-1)) / 25
    badi = (merrors > threshold).nonzero()[:, 0]
    masks[torch.isin(masks, badi)] = 0
    return masks


def fill_holes_and_remove_small_masks(masks, min_size=15, hole_size=3, scale_factor=1):
    """ fill holes in masks (2D/3D) and discard masks smaller than min_size (2D)

    fill holes in each mask using scipy.ndimage.morphology.binary_fill_holes

    Parameters
    ----------------
    masks: int, 2D or 3D array
        labelled masks, 0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]
    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1

    Returns
    ---------------
    masks: int, 2D or 3D array
        masks with holes filled and masks smaller than min_size removed, 
        0=NO masks; 1,2,...=mask labels,
        size [Ly x Lx] or [Lz x Ly x Lx]

    """

    hole_size *= scale_factor
    slices = find_objects(masks)
    j = 0
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            npix = msk.sum()
            if min_size > 0 and npix < min_size:
                masks[slc][msk] = 0
            else:
                hsz = npix * hole_size / 100  # turn hole size into percentage
                pad = 1
                unpad = tuple([slice(pad, -pad)] * msk.ndim)
                padmsk = remove_small_holes(np.pad(msk, pad, mode='constant'), hsz)
                msk = padmsk[unpad]
                masks[slc][msk] = (j + 1)
                j += 1
    return masks


def compute_masks(
        flows: torch.Tensor,
        dist: torch.Tensor,
        bd: torch.Tensor,
        niter=200,
        mask_threshold=0.0,
        flow_threshold=0.4,
        min_size=9,
        calc_trace=False,
        filename_noext='',
        velocity=1.5,
        cluster_thres=8,
        debug_dir='/nfs4-p1/hkw/debug',
        debug=True):
    """
    Compute masks using dynamics from dP, dist, and boundary outputs.

    Parameters
    -------------
    dP: float, [2 H W] array
        flow field components
    dist: float, ND array
        smoothed distance field (H, W)
    bd: float, ND array
        boundary field
    p: float32, ND array
        initial locations of each pixel before dynamics,
        size [2 H W]
    inds: int32, 2D array
        non-zero pixels to run dynamics on [npixels x N]
    niter: int32
        number of iterations of dynamics to run
    rescale: float (optional, default None)
        resize factor for each image, if None, set to 1.0   
    resize: int, tuple
        shape of array (alternative to rescaling)  
    mask_threshold: float 
        all pixels with value above threshold kept for masks, decrease to find more and larger masks 
    flow_threshold: float 
        flow error threshold (all cells with errors below threshold are kept) (not used for Cellpose3D)
    min_size: int (optional, default 15)
        minimum number of pixels per mask, can turn off with -1
    calc_trace: bool 
        calculate pixel traces and return as part of the flow

    Returns
    -------------
    mask: int, ND array
        label matrix
    p: float32, ND array
        final locations of each pixel after dynamics,
        size [axis x Ly x Lx] or [axis x Lz x Ly x Lx]. 
    tr: float32, ND array
        intermediate locations of each pixel during dynamics,
        size [axis x niter x Ly x Lx] or [axis x niter x Lz x Ly x Lx]. 
        For debugging/paper figures, very slow. 

    """
    import os
    mask = dist > 0
    # mask = filters.apply_hysteresis_threshold(dist.cpu().numpy(), mask_threshold - 1, mask_threshold )
    # mask = torch.from_numpy(mask).to(flows.device)
    inds = mask.nonzero().long().T
    if debug:
        pass

    # omnipose主要问题: 容易出现小杂质
    if mask.any():
        dP_ = div_rescale(flows, mask)
        # diam = dist[~mask].mean() * (-2)  # EDT事实上能预测图内细胞平均大小
        p, inds, tr = follow_flows(dP_, inds, niter=niter, calc_trace=calc_trace, velocity=velocity)
        mask, _ = get_masks(p, bd, dist, mask, inds, prefix=filename_noext)
        shape0 = p.shape[1:]
        mean_area = 0
        if mask.max() > 0 and flow_threshold > 0:
            mask = remove_bad_flow_masks(mask, flows, threshold=flow_threshold)
            _, mask = torch.unique(mask, return_inverse=True)
            mask = mask.reshape(shape0).long()

        mask = mask.cpu().numpy()
        areas = np.bincount(mask.reshape(-1))[1:]
        mean_area = (areas.sum() / ((areas > 0).sum() + 1e-10))
        mask = fill_holes_and_remove_small_masks(mask, mean_area / 5, 10)
    else:
        p = np.zeros([2, 1, 1])
        tr = []
        mask = np.zeros_like(dist.cpu().numpy()).astype(np.int32)
    return mask, p, tr
