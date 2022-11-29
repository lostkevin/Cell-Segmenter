from multiprocessing.sharedctypes import Value
from typing import List, Optional, Union
import torch
import numpy as np
import edt
import tifffile as tif
import os
from torch.nn import functional as F
from torchvision.transforms import functional as tvF
import time
import random

from utils.general import LOGGER
from utils.taskrunner import TaskRunner

__all__ = ['labels_to_flows', 'masks_to_flows']

def get_niter(dists):
    """
    Get number of iterations. 
    
    number of iterations empirically found to be the lower bound for convergence 
    of the distance field relaxation method
    """
    return min(200, torch.ceil(dists.max()*1.16).long().cpu().item()+1)

def diameters(masks: torch.Tensor, dt: torch.Tensor, dist_threshold: float=0):
    
    """
    Calculate the mean cell diameter from a label matrix. 
    
    Parameters
    --------------
    masks: ND array, float
        label matrix 0,...,N
    dt: ND array, float
        distance field
    dist_threshold: float
        cutoff below which all values in dt are set to 0. Must be >=0. 
        
    Returns
    --------------
    diam: float
        a single number that corresponds to the average diameter of labeled regions in the image, see dist_to_diam()
    
    #! diameter(x) the average distance to the border (√)
    """
    if dt is None:
        raise NotImplementedError()
    dist_threshold = max(0, dist_threshold)
    dt_pos = dt[dt>dist_threshold].abs()
    diam = 0
    if dt_pos.any():
        diam = 6 * dt_pos.abs().mean()
    return diam

def labels_to_flows(labels, files=None):
    """ Convert labels (list of masks or flows) to flows for training model.

    if files is not None, flows are saved to files to be reused

    Parameters
    --------------
    labels: list of ND-arrays
        labels[k] can be 2D or 3D, if [3 x Ly x Lx] then it is assumed that flows were precomputed.
        Otherwise labels[k][0] or labels[k] (if 2D) is used to create flows.
    files: list of strings
        list of file names for the base images that are appended with '_flows.tif' for saving. 
    Returns
    --------------
    flows: list of [5 x H x W] arrays
        flows[k][0] is labels[k], flows[k][1] is cell distance transform, flows[k][2:2+dim] are the 
        YX flow components, and flows[k][-1] is heat distribution / smooth distance 
    """
    if labels[0].ndim != 5: 
        runner = TaskRunner()
        # mask_to_flows: label edt mu T
        # cat: label dist vec heat
        if files is not None:
            if len(files) != len(flows):
                raise ValueError()
            def _gen(labels, files):
                for l, f in zip(labels, files):
                    yield {
                        'masks': l,
                        'file': f
                    }
            flows = runner.run(masks_to_flows, _gen(labels, files), total=len(labels))
        else:
            flows = runner.map(masks_to_flows, labels, total=len(labels))
        runner.terminate()
    else:
        flows = [l.astype(np.float32) for l in labels]
    return flows

def masks_to_flows(masks: Union[np.ndarray, torch.Tensor], 
                    file: Optional[str]=None, 
                    dists: Optional[Union[np.ndarray, torch.Tensor]]=None,
                    is_cuda: bool = True,
                    return_tensor: bool = False,
                    device = None
                    ) -> np.ndarray:
    """
        build flows from masks, always assume masks with correct format
        if dists is None, use default edt map

        return: 3-D array with shape [5, H, W]
        boundary, edt, weight, fY, fX
    """ 
    if isinstance(masks, np.ndarray):
        masks = torch.Tensor(masks.astype(np.int32))
    if dists is None:
        cells = torch.unique(masks)
        if cells[0] == 0:
            cells = cells[1:]
        dists = edt.edt(masks.cpu().numpy())
    bg_edt = torch.Tensor(edt.edt((masks == 0).cpu().numpy(), black_border=True))
    
    if isinstance(dists, np.ndarray):
        dists = torch.Tensor(dists)
    if is_cuda:
        if device is None:
            device = random.choice(range(torch.cuda.device_count()))
        dists, masks = dists.cuda(device), masks.cuda(device)
    cutoff = diameters(masks, dists) / 2
    pad = int(cutoff)
    if pad > 0:
        #! reflect padding on border cells with large distance border distance
        #! pad width = diameter // 2
        masks_pad = F.pad(get_edge_masks(masks, dists=dists).unsqueeze(0).float(), (pad, ) * (2 * masks.ndim), mode='reflect')[0]
        masks_pad[pad:-pad, pad:-pad] = masks
        mu, T = masks_to_flows_torch(masks_pad, dists)
        mu, T = mu[:, pad:-pad, pad:-pad], T[:, pad:-pad, pad:-pad]
    else:
        mu, T = masks_to_flows_torch(masks, dists)
    
    dists = dists[None] # HW => 1HW
    boundary = dists == 1
    T[dists <= 0] = -cutoff
    mu *= 5
    bg_edt = bg_edt.to(masks.device)[None] # HW => 1HW
    bg_edt = tvF.gaussian_blur(1 - bg_edt.clamp_(0, cutoff) / (1e-10 + cutoff), 3, 1.0) + 0.5
    if bg_edt.min() < 0:
        raise ValueError('?')
    flow = torch.cat([boundary, T, bg_edt, mu], dim=0).float()
    if file is not None:
        LOGGER.info('Save in masks_to_flows is deprecated... DO NOTHING')
    return flow.cpu() if return_tensor else flow.cpu().numpy()

def get_edge_masks(labels: torch.Tensor, dists: torch.Tensor):
    """
    Finds and returns masks that are largely cut off by the edge of the image.
    """
    if labels.ndim != 2:
        raise ValueError(f'Find abnormal labels with shape: {labels.ndim} required ndim==2')
    border = torch.ones_like(labels, dtype=bool)
    border[1:-1, 1:-1] = 0
    clean_labels = torch.zeros_like(labels)
    for cell_ID in torch.unique(labels[border])[1:]:
        mask = labels == cell_ID 
        max_dist = dists[mask * border].max()
        dist_thresh = torch.quantile(dists[mask], 0.75)
        # we only want to keep cells whose distance at the boundary is not too small
        if max_dist >= dist_thresh: 
            clean_labels[mask] = cell_ID
    return clean_labels

def safe_divide(num,den):
    """ Division ignoring zeros and NaNs in the denominator.""" 
    out = torch.zeros_like(num)
    mask = (den != 0) * (~torch.isnan(den)) 
    out[:, mask] = num[:, mask] / den[None, mask]
    return out     

def masks_to_flows_torch(masks: torch.Tensor, dists: torch.Tensor):
    """Convert ND masks to flows. 
    
    Omnipose find distance field, Cellpose uses diffusion from center of mass.

    Parameters
    -------------

    masks: int, 2-D Tensor with shape [H, W]
        labelled masks, 0 = background, 1,2,...,N = mask labels
    dists: 2-D Tensor with shape [H, W], float
        array of (nonnegative) distance field values
    Returns
    -------------
    mu: float, 3D Tensor [2 x H x W]
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z or T = mu[0].
    dist: float, 3D Tensor [1 x H x W]
        smooth distance field (Omnipose)
    """
    
    if masks.any():
        masks_padded = F.pad(masks, (1, ) * (2 * masks.ndim),'constant', 0)
        # run diffusion 
        mu, T = _extend_centers_torch(masks_padded, n_iter=get_niter(dists))
        # normalize field
        mu = safe_divide(mu, torch.sqrt(torch.nansum(mu**2,dim=0)))
        mu0 = torch.zeros((mu.shape[0],)+masks.shape, device=masks.device, dtype=mu.dtype)
        mu0[(Ellipsis,)+masks.nonzero(as_tuple=True)] = mu
        return mu0, T[:, 1:-1, 1:-1]
    else:
        return torch.zeros((masks.ndim,)+masks.shape, device=masks.device), torch.zeros((1, *masks.shape), device=masks.device)

def _extend_centers_torch(masks: torch.Tensor, n_iter:int=200) -> List[torch.Tensor]:
    """ runs diffusion on GPU to generate flows for training images or quality control
    PyTorch implementation is faster than jitted CPU implementation, therefore only the 
    GPU optimized code is being used moving forward. 
    
    Parameters
    -------------

    masks: int, 2D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    n_inter: int
        number of iterations
        
    Returns
    -------------
    mu: float, 3D or 4D array 
        flows in Y = mu[-2], flows in X = mu[-1].
        if masks are 3D, flows in Z (or T) = mu[0].
    dist: float, 2D or 3D array
        the smooth distance field (Omnipose)
        or temperature distribution (Cellpose)
         
    """
        
    d = 2
    coords = masks.nonzero().T # [d x K]
    idx = 4 # center pixel index

    # step: [d x 9]
    steps = torch.stack(torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing='ij')).flatten(1).to(masks.device)
    # pt: [d x 9 x K]
    pt = steps.unsqueeze(-1) + coords.unsqueeze(1)

    sign = steps.abs().sum(dim=0)
    uniq = torch.unique(sign)
    inds = [torch.where(sign==i)[0] for i in uniq] # 2D: [4], [1,3,5,7], [0,2,6,8]. 1-7 are y axis, 3-5 are x, etc. 
    fact = uniq.sqrt()
    # neighbor_masks: [9 x K]
    neighbor_masks = masks[tuple(pt)] #extract list of label values, 
    
    T = torch.zeros((1, *masks.shape), dtype=torch.double, device=masks.device)
    #! masks contain cell id, therefore here we compute masks where value equals to its center
    isneigh = neighbor_masks == neighbor_masks[idx]  # isneigh is <3**d> x <number of points in mask>
    
    mask_pix = (...,) + tuple(pt[:,idx]) #indexing for the central coordinates 
    for _ in range(n_iter):
        T[mask_pix] = eikonal_update_torch(T,pt,isneigh,d,inds,fact) ##### omnipose.core.eikonal_update_torch
        
    # There is still a fade out effect on long cells, not enough iterations to diffuse far enough I think 
    # The log operation does not help much to alleviate it, would need a smaller constant inside. 
    idx = inds[1]
    # prevent bleedover, big problem in stock Cellpose that got reverted! 
    #! The indexing is magic, do not change
    grads = T[(...,) + tuple(pt[:, idx])] * isneigh[idx] 
    mu_torch = torch.stack([(grads[:,-(i+1)]-grads[:,i]).squeeze() for i in range(0, grads.shape[1]//2)])/2
    #! Fix: special case: [2, 1] -> [2]
    if mu_torch.ndim == 1:  
        mu_torch.unsqueeze_(1)
    return mu_torch, T.float()

def eikonal_update_torch(T,pt,isneigh,d=None,index_list=None,factors=None):
    """Update for iterative solution of the eikonal equation on GPU."""
    # Flatten the zero out the non-neighbor elements so that they do not participate in min
    
    Tneigh = T[(Ellipsis,)+tuple(pt)]
    Tneigh *= isneigh
    # preallocate array to multiply into to do the geometric mean
    phi_total = torch.ones_like(Tneigh[0,0,:])
    # loop over each index list + weight factor 
    for inds,fact in zip(index_list[1:],factors[1:]):
        mins = [torch.minimum(Tneigh[:,inds[i],:],Tneigh[:,inds[-(i+1)],:]) for i in range(len(inds)//2)] 
        phi = update_torch(torch.cat(mins),fact)
        phi_total *= phi    
    return phi_total**(1/d)

def update_torch(a,f):
    sum_a = torch.cumsum(a,dim=0)
    sum_a2 = torch.cumsum(a**2,dim=0)
    d = torch.cumsum(torch.ones_like(a),dim=0)
    radicand = sum_a**2-d*(sum_a2-f**2)
    mask = radicand>=0
    d = torch.count_nonzero(mask,dim=0)
    r = torch.arange(0,a.shape[-1])
    ad = sum_a[d-1,r]
    rd = radicand[d-1,r]
    return (1/d)*(ad+torch.sqrt(rd))

### Omnipose Postprocessing