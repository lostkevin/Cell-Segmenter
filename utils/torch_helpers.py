import torch
import contextlib

def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])
    ndim = x.ndim
    indicies = torch.sum(torch.ge(x[..., None], xp.view((1,) * ndim + (-1,)) ), -1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * x + b[indicies]

from typing import *
def clamp_on_dimension(tensor: torch.Tensor, 
                        vmin: Union[torch.Tensor, float], 
                        vmax: Union[torch.Tensor, float], 
                        dims: Optional[Union[torch.Size, int]]=None):
    """
        Input shape: [K1, K2, ..., Kn]
        dims: the dimensions with thresholds
        vmin, vmax: tensors which can be broadcast to shape[sorted(dims)]

        ! Always return copy
    """
    if isinstance(dims, int):
        dims = (dims,)
    if not isinstance(vmin, torch.Tensor):
        vmin = torch.tensor(vmin, device=tensor.device)
    if not isinstance(vmax, torch.Tensor):
        vmax = torch.tensor(vmax, device=tensor.device)
    if dims is None or len(dims) == 0:
        return tensor.clamp(vmin, vmax)
    dims = sorted(dims)
    shape = torch.tensor(tensor.shape)[dims]
    vmin = torch.broadcast_to(vmin, torch.Size(shape))
    vmax = torch.broadcast_to(vmax, torch.Size(shape))
    new_shape = torch.ones(tensor.ndim, device=tensor.device, dtype=torch.long)
    new_shape[dims] = shape
    return torch.max(torch.min(tensor, vmax.view(torch.Size(new_shape))), vmin.view(torch.Size(new_shape)))