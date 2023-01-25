import torch
from matplotlib import pyplot as plt
from util import detach_all, to_numpy


def adapt_axes_xyz(xyz, real_props=True, ax=None):
    """
    :param xyz: shape=(m,n,3)
    :param real_props:  if True, adjust axes units to be the same, for real world dimension proportions
                        if False, use default proportion adaptations by matplotlib, with bounding ranges
    :param ax:
    :return ax:
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    xyz_min = [torch.min(xyz[..., i]).item() * 1.1 for i in range(3)]
    xyz_max = [torch.max(xyz[..., i]).item() * 1.1 for i in range(3)]
    if real_props:
        xyz_mean = [torch.mean(xyz[..., i]).item() for i in range(3)]
        cx, cy, cz = xyz_mean
        # abs_xyz_min = [torch.abs(_) for _ in xyz_min]
        half_disp_wide = max([xyz_max[i] - xyz_mean[i] for i in range(3)])
        ax.set_xlim(cx - half_disp_wide, cx + half_disp_wide)
        ax.set_ylim(cy - half_disp_wide, cy + half_disp_wide)
        ax.set_zlim(cz - half_disp_wide * 0.75, cz + half_disp_wide * 0.75)
    else:
        for settings, min_, max_ in zip(
                [ax.set_xlim, ax.set_ylim, ax.set_zlim],
                xyz_min,
                xyz_max
        ):
            if max_ - min_ < 1:
                min_ -= 0.5
                max_ += 0.5
            settings(min_, max_)

    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    return ax


def scatter_xyz(xyz, ax=None, color='r', size=20):
    """
    draw scatter xyz
    :param xyz:  shape=(...,3<xyz dims>)
    :param ax:
    # :param color:
    :return ax:
    """
    xyz, = detach_all(xyz)
    ax = adapt_axes_xyz(xyz) if ax is None else ax
    x, y, z = torch.split(xyz, 1, dim=-1)
    ax.scatter(x, y, z, s=size, c=color, marker='.')
    # plt.show()
    return ax


def quiver_xyz(xyz, uvw, ax=None, color='b'):
    """
    draw quiver xyz =----> xyz + uvw
    :param xyz:
    :param uvw:
    :param ax:
    :param color:
    :return ax:
    """
    xyz, uvw = detach_all(xyz, uvw)
    ax = adapt_axes_xyz(xyz + uvw) if ax is None else ax
    x, y, z = torch.split(xyz, 1, dim=-1)
    u, v, w = torch.split(uvw, 1, dim=-1)
    ax.quiver(x, y, z, u, v, w, arrow_length_ratio=0.02, color=color, alpha=0.2)
    # plt.show()
    return ax


def show_z_surf(xyz, ax=None):
    """
    plot z(x,y) in 3D perspective
    :param xyz:shape=(m,n,3)
    :param ax:
    :return:
    """
    ax = adapt_axes_xyz(xyz) if ax is None else ax
    xyz = to_numpy(xyz)
    plot_surf = ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2])#, surf,cmap=cm.ocean)
    return ax
