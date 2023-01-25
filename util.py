import os
from scipy import io
import torch
import numpy as np
from torch.nn.functional import relu


# from tqdm import tqdm


# from pytorch_lightning.callbacks import TQDMProgressBar
#
#
# class NoValProgressBar(TQDMProgressBar):
#     """
#     the val progress bar of lightning keep reprints in pycharm each time it refreshes, annoying.
#     This simply disables it
#     """
#
#     def init_validation_tqdm(self):
#         bar = tqdm(
#             disable=True,
#         )
#         print('validating...')
#         return bar
#

def detach_all(*torch_tensors):
    torch_tensors = [t_.detach() for t_ in torch_tensors]
    return torch_tensors


def deshape(shape):
    ss = 1
    for s in shape:
        ss *= s
    return ss


def standardized(arr):
    return arr / arr.max()


def normalized(t):
    return t / torch.norm(t, dim=-1, keepdim=True)


def write_2d_vec_as_3d(t):
    return torch.cat([t, torch.zeros_like(t[..., 0:1])], dim=-1)


def to_numpy(tensor_):
    try:
        tensor_ = tensor_.detach().cpu()
    except:
        pass
    array = np.array(tensor_)
    return array

def illum_field_save_txt(txt_save_path, tensor_):
    arr_ = to_numpy(tensor_)
    m, n = arr_.shape
    np.savetxt(txt_save_path, arr_,
               header='MESH:	{}	{}	-12.827	-12.827	12.827	12.827\n##	Mesh cell values'.format(m,
                                                                                                                   n),
               delimiter='\t', comments='')

def mat_save_dict(dict_, path_):
    # mind that if some folder in the path_ does not exist, this can fail without warning
    head_path = os.path.split(path_)[0]
    assert os.path.exists(head_path)
    assert path_[-4:] == '.mat'
    for key, val in dict_.items():
        dict_[key] = to_numpy(val)
    io.savemat(path_, dict_)
    # print('saved ', f'{file_name}.mat ', 'to ', root_path)


def mat_load_dict(path_):
    assert path_[-4:] == '.mat'
    dict_ = io.loadmat(path_)
    return dict_


def mat_easy_save(tensor, path_):
    dict_ = {'key': tensor}
    mat_save_dict(dict_, path_)


def mat_easy_load(path_):
    dict_ = mat_load_dict(path_)
    return dict_['key']


def sigmoid_step(x, x0=0., steepness=1.):
    """u(x-x0)
    # sigmoid soften step u(x-x0),
    :param steepness>0
    # x0 is the central coordinate of sigmoid function
    """
    return torch.sigmoid(steepness * (x - x0))


def round_gaussian_spot(mu_xy, sigma, xy):
    """
    rotational symmetric gaussian spot centering at (mu_x,mu_y)
    :param mu_x:
    :param mu_y:
    :param sigma:
    :param x:
    :param y:
    :return:
    """

    rho_sq = torch.sum((xy - mu_xy) ** 2, dim=-1)
    z = 1. / np.sqrt(2 * np.pi) / sigma * torch.exp(-rho_sq / 2 / sigma ** 2)
    return z


def cart2pol_2d(xx, yy):
    """
    transform cartesian coordiante grid to polar coordinate grid (using 'tensorflow')
    """

    rho = torch.sqrt(xx ** 2 + yy ** 2)
    theta = torch.atan2(yy, xx) + np.pi
    return theta, rho


def xy_seen_as_in_unit_box(xy, xy_min, xy_max):
    """
    standardize xy coordinates with range xy_min to xy_max
    take the range in actual space with xy_min, xy_max and see it as a unit square between (0,0) and (1,1) return the
        corresponding coordinates in this unit square of 'xy' defined points in actual space
    :param xy:shape=(...,2) input xy
    :param xy_min: iterable, representation of min corner point coordinates of the box range as the range to transform to unit box
    :param xy_max: iterable, representation of max corner point coordinates of the box range as the range to transform to unit box
    :return: xy corresponding coordinates in this unit square
    """
    # for i in range(2):
    #     assert xy[..., i].min() > xy_min[i] and xy[..., i].max() < xy_max[i]

    xy_min = xy_min.reshape(1, 1, 2)
    xy_max = xy_max.reshape(1, 1, 2)
    d = xy_max - xy_min
    return (xy - xy_min) / d


def area_tri_3p_set(v_3d):
    """
    :param v: aka point_group_3p_2d vertices, shape=(...,3,2).
    :return: bright_ness shape=(...,1,1).
    """
    o = v_3d[..., 0:1, :]
    a = v_3d[..., 1:2, :] - o
    b = v_3d[..., 2:, :] - o
    cross_ = torch.cross(a, b)
    area = torch.norm(cross_, dim=-1, keepdim=True) / 2.
    return area


def sum_cos_diff_vecs_xz(p0_xz):
    # x_z diff dot product
    diff_vec_xz = p0_xz[:, :-1, :] - p0_xz[:, 1:, :]
    normed_diff_vec_xz = diff_vec_xz / (torch.norm(diff_vec_xz, dim=-1, keepdim=True) + 1e-6)
    cos_diff_vecs_xz = torch.sum(normed_diff_vec_xz[:, :-1, :] * normed_diff_vec_xz[:, 1:, :], dim=-1)
    return cos_diff_vecs_xz


def local_thresh_cos_loss(point_grid3d, thresh_ang_in_rad=10.):
    """

    :param point_grid3d:(,,3) 3D tensor, like surf point of a tri_refractive_surf is one
    :param thresh_ang_in_rad 0~pi deg
    :return:
    """
    # thresh = torch.cos(thresh_ang_in_deg / 180. * np.pi)

    p0_xz = point_grid3d[..., [0, 2]]
    p0_yz = point_grid3d[..., [1, 2]]

    cos_xz = sum_cos_diff_vecs_xz(p0_xz)
    cos_yz = sum_cos_diff_vecs_xz(p0_yz.permute(1, 0, 2))
    all_cos = torch.stack([cos_xz, cos_yz])
    threshed_all_cos = torch.relu(all_cos - thresh_ang_in_rad)

    loss = torch.mean(threshed_all_cos)
    return loss


def local_diff_loss(point_grid3d, thresh_abs_z_diff):
    """

    :param point_grid3d:(,,3) 3D tensor, like surf point of a tri_refractive_surf is one
    :param thresh_ang_in_rad 0~pi deg
    :return:
    """

    z = point_grid3d[..., 2]

    row_diff_z = z[1:, ...] - z[:-1, ...]
    col_diff_z = z[..., 1:] - z[..., :-1]

    all_diff = torch.stack([row_diff_z, col_diff_z.permute(1, 0)])
    abs_all_diff = torch.abs(all_diff)
    threshed_all_abs_diff = torch.relu(abs_all_diff - thresh_abs_z_diff)

    loss = torch.mean(threshed_all_abs_diff)
    return loss


if __name__ == '__main__':
    """"# deb<
    p3 = torch.tensor([
        [-2, 0],
        [0, -1],
        [1, 0],

    ], dtype=torch.float32)  # anti-clockwise# (4, 2)
    p3_ = torch.rand([6, 7, 3, 2])
    v_3d = write_2d_vec_as_3d(p3)
    area_ = area_tri_3p_set(v_3d)
    print(area_.shape)
        """  # deb>
    a = torch.rand((65, 65, 3))
    # loss = local_diff_loss(a, 0.01)
    path = r'./lightning_logs/test/test.mat'
    mat_easy_save(a, path)
    b = mat_easy_load(path)
    are_close = torch.all(a == torch.tensor(b))
    print(are_close)
