
import torch
# from util import *
from torch import float32 as t32
from visTools import scatter_xyz


def z_plane_xy_interval_array(x_dot_num, y_dot_num,
                              x_interval, y_interval,
                              center_xyz=(0., 0., 0.)):
    """
    mesh grid array defined by x and y intervals on a z=constant plane
    :param x_dot_num:
    :param y_dot_num:
    :param x_interval:
    :param y_interval:
    :param center_xyz:
    :return:
    """
    assert x_dot_num > 1 and y_dot_num > 1 and x_interval > 0 and y_interval > 0
    half_x_len = (x_dot_num - 1) * x_interval / 2.
    half_y_len = (y_dot_num - 1) * y_interval / 2.
    x0, y0 = torch.meshgrid(
        torch.linspace(-half_x_len, half_x_len, x_dot_num),
        torch.linspace(-half_y_len, half_y_len, y_dot_num)
    )
    center_xyz = torch.tensor(center_xyz, dtype=t32).reshape(1, 1, -1)
    z0 = torch.zeros_like(x0)
    xyz000 = torch.stack([x0.permute(1, 0), y0.permute(1, 0), z0.permute(1, 0)], dim=-1)
    # .permute(1, 0) to be consistent with tensorflow style, which is also used daily in math
    xyz = xyz000 + center_xyz
    return xyz


def xy_range_array(x_dot_num, y_dot_num,
                   x_low_bound, x_high_bound,
                   y_low_bound, y_high_bound):
    """
    mesh grid array defined by x and y boundaries
    :param x_dot_num:
    :param y_dot_num:
    :param x_low_bound:
    :param x_high_bound:
    :param y_low_bound:
    :param y_high_bound:
    :return:
    """

    assert x_dot_num > 1 and y_dot_num > 1 and x_low_bound < x_high_bound and y_low_bound < y_high_bound
    x0, y0 = torch.meshgrid(
        torch.linspace(x_low_bound, x_high_bound, x_dot_num),
        torch.linspace(y_low_bound, y_high_bound, y_dot_num)
    )
    xy = torch.stack([x0.permute(1, 0), y0.permute(1, 0)], dim=-1)
    # .permute(1, 0) so that in consistent with tensorflow style, which is also used daily in math
    return xy


def add_rand_noise_for_xyz(xyz, add_to=2, z_rand_amp=1.):
    """
    :param xyz:
    :param add_to: 0,1,2 represent x,y,z
    :param z_rand_amp: amplitude to mul the rand noise of (-0.5, 0.5)
    :return:
    """
    xyz[:, :, add_to] += (torch.rand_like(xyz[:, :, add_to]) - 0.5) * z_rand_amp
    return xyz


if __name__ == '__main__':
    xyz = z_plane_xy_interval_array(3, 3, 3, 3, (1, 2, 3))
    # min_xyz = [torch.min(xyz[:, :, i]).item() for i in range(3)]
    # max_xyz = [torch.max(xyz[:, :, i]).item() for i in range(3)]
    #
    # min_source_xyz, _ = torch.min(xyz, dim=0)
    # min_source_xyz.shape
    scatter_xyz(xyz)
