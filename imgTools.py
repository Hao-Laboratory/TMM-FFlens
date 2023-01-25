"""
tools related to image preparation read write ect
"""
import math

import cv2
import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt
import time
import os.path as path
from pathlib import Path
from imageio import imsave  # be called as imgTools.imsave
from util import to_numpy
from os import mkdir
import torch.nn.functional as F

import matplotlib


def read_csv_illumfield_and_standardized(csv_path):
    with open(csv_path, "r") as csv_file:
        illum_field = np.loadtxt(csv_file, delimiter="\t")
    # flip it upside down to match orientation of aim target file
    # illum_field = torch.flip(illum_field, [0, 1])# filp updown
    # standardize it to 0~1
    illum_field = illum_field / np.max(illum_field)
    return illum_field


def read_as_gray_resized_and_standardized(img_file_path, resize_shape=None):
    # read file in
    input_gray_image = im2gray(img_file_path, 'cv2')
    org_size = input_gray_image.shape
    # resize
    want_gray = cv2.resize(input_gray_image, resize_shape) if resize_shape is not None else input_gray_image
    print('input size {} reshaped to {}'.format(org_size, resize_shape))
    _want_gray = torch.tensor(want_gray, dtype=torch.float32)
    aim_dstbn = _want_gray / torch.max(_want_gray)  # pix to range 0~1 standardize
    return aim_dstbn, input_gray_image


def inscrib_circle_cropped(square_mat):
    """find the num center of the mat and use distance to crop"""

    m, n = square_mat.shape
    assert m == n
    r = m / 2.
    x, y = np.meshgrid(range(m), range(n))
    c = (m - 1) / 2.  # center
    d = np.sqrt((x - c) ** 2 + (y - c) ** 2)
    return np.where(d <= r, square_mat, 0.)


# matplotlib.use('Qt5Agg')
def batch_imresize(img, size):
    return F.interpolate(img, size=size, mode='bicubic')


def imresize(img, size_2d):
    size = size_2d
    if len(img.shape) == 3:
        img = torch.unsqueeze(img, 0)
        img = img.permute(0, 3, 1, 2)
        resize_img = torch.squeeze(batch_imresize(img, size), dim=0)
        return resize_img.permute(1, 2, 0)
    elif len(img.shape) == 2:
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0)
        resize_img = torch.squeeze(batch_imresize(img, size))
        return resize_img
    else:
        return None


def group_cat_img(img_group, dim=1, regular=True):
    assert isinstance(dim, int)
    if not isinstance(img_group, list):
        img_group = [img_group]
    img_group = [to_numpy(img) for img in img_group]
    if regular:
        img_group = [regular_img(img) for img in img_group]
    # interp imgs to the same size
    equal_dim = 1 - dim
    assert -1 < equal_dim < 2
    intered_img_group = interp_for_cat(img_group, dim=equal_dim)
    cat_img = np.concatenate(intered_img_group, axis=dim)
    return cat_img


def group_img_save(save_dir, group_name, img_group, dim=1):
    cat_img = group_cat_img(img_group, dim=dim)
    file_save_path = path.join(save_dir, group_name + '.jpg')
    imsave(file_save_path, cat_img)


class SeriesImgSaver:
    def __init__(self, series_name, save_root=None):
        save_root = Path(__file__).parents[2].joinpath('images') if save_root is None else save_root
        self.save_dir = path.join(save_root, series_name)
        print('CHECKOUT!\nThe files passed in this SeriesImgSaver.save() will be saved to here:\n',
              self.save_dir,
              '\nwith suffix ".jpg"')
        if not path.exists(self.save_dir):
            mkdir(self.save_dir)
            print('---path folder created---')

    def save(self, group_name, img_group):
        group_img_save(self.save_dir, group_name, img_group)


def interp_for_cat(array_series, dim=1):
    first_dim_n = len(array_series[0].shape)
    for arr in array_series[1:]:
        assert len(arr.shape) == first_dim_n  # assert that they all have the same dim
    max_dim_len = 0  # i_of_the_max_in_dim
    old_shapes = []
    # old_dim_lens = []
    for i in range(len(array_series)):
        this_shape = array_series[i].shape
        this_dim_len = this_shape[dim]
        old_shapes.append(this_shape[:2])
        # old_dim_lens.append(this_dim_len)
        if max_dim_len < this_dim_len:
            max_dim_len = this_dim_len
    new_shapes = [scale_shape_make_dim_n(shape, dim, max_dim_len) for shape in old_shapes]
    #  calc new shapes for the arrs
    interped_arr_series = [imresize(torch.tensor(arr), new_shape) for arr, new_shape in
                           zip(array_series, new_shapes)]
    return interped_arr_series


def scale_shape_make_dim_n(shape, dim: int, dim_to_pix_n: int):
    assert isinstance(dim, int)
    assert isinstance(dim_to_pix_n, int)
    assert dim >= 0 and dim < 2
    shape_tail = list(shape[2:])
    shape = shape[:2]
    dim_n_in_shape = shape[dim]
    scale_ratio = float(dim_to_pix_n) / float(dim_n_in_shape)
    new_shape = [int(shape[i] * scale_ratio) if i != dim else dim_to_pix_n for i in range(len(shape))]
    return tuple(new_shape + shape_tail)


gray_args = {'cmap': 'gray', 'vmin': 0, 'vmax': 255}


def strech_img(x):  # normalize x into [0, 1]
    return (x - x.min()) / (x.max() - x.min())


def pad_half_syspsf_gen(sys, feed_num, feed_numdata_func, *args, **kwargs):
    """
    all img yielded looks like padded with half width of psf scope
    Arguments:
        sys: a FourFSetup class including psf scope
        feed_num: total number of object
        feed_numdata_func: a function for data generation
    Returns:
        An iterator with padding
    """
    return make_padded_gen(int(0.5 * sys.psf_scope.shape[0]), feed_num, feed_numdata_func, sys.obj_scope.shape, *args,
                           **kwargs)


def pad_half_syspsf_block(sys, feed_numdata_func, *args, **kwargs):
    return padded_block(feed_numdata_func, int(0.5 * sys.psf_scope.shape[0]), sys.obj_scope.shape, *args, **kwargs)


def make_padded_gen(pad_pix_num, total_obj_num, feed_func, raw_obj_shape, *args, **kwargs):
    """
    Arguments:
        pad_pix_num: padded pixel number of raw object
        total_obj_num: total number of object
        feed_func: function for object generation
        raw_obj_shape: shape of raw object
    Returns:
        An iterator with padding
    """

    def padded_func(raw_obj_shape, *args, **kwargs):
        return padded_block(feed_func, pad_pix_num, raw_obj_shape, *args, **kwargs)

    return make_gen(total_obj_num, padded_func, raw_obj_shape, *args, **kwargs)


def padded_block(block_func, pad_pix_num, shape, *args, **kwargs):
    core_shape, pad_ind = pad_adapt_shape(pad_pix_num, shape)
    core = block_func(core_shape, *args, **kwargs)
    padded = torch.pad(core, pad_ind)
    return padded


def pad_adapt_shape(pad_pix_num, shape):
    core_shape = []
    pad_ind = []
    for i, di in enumerate(shape):  # cut 2*padpixnum down form the first 2 dims of shape
        if i < 2:
            assert di > 2 * pad_pix_num
            core_shape.append(di - 2 * pad_pix_num)
            pad_ind.append([pad_pix_num] * 2)
        else:
            core_shape.append(di)
            pad_ind.append([0] * 2)
    return tuple(core_shape), tuple(pad_ind)


def make_gen(total_obj_num, func, *args, **kwargs):  # decrator that converts block func to block gen
    np.random.seed(666)
    torch.random.set_seed(666)
    if total_obj_num is None:
        while True:
            yield func(*args, **kwargs)
    else:
        for i in range(total_obj_num):
            yield func(*args, **kwargs)


def regular_img(img_ten, colorful=True):
    img_ten = np.squeeze(img_ten)
    img_ten = img_ten / (np.amax(img_ten) + 1e-9)  # to[0,1]
    if len(img_ten.shape) == 2:
        if colorful:
            img_ten = apply_colormap(img_ten)  # this does tiling inside
        else:
            img_ten = np.tile(img_ten[..., None], (1, 1, 3)).astype(np.uint8)
        assert len(img_ten.shape) == 3
    else:
        img_ten *= 255.
    return img_ten.astype(np.uint8)


def apply_colormap(img_mat, cmap=cv2.COLORMAP_VIRIDIS):
    assert len(img_mat.shape) == 2
    img_mat_ = img_mat - img_mat.min()
    u8_img = np.array(img_mat_ / (img_mat_.max() + 1e-9) * 255).astype(np.uint8)
    p_img_ = cv2.applyColorMap(u8_img, cmap)
    p_img = p_img_[:, :, ::-1]  # to RGB order
    return p_img


def simple_spiral(z, n_rounds, a_max, b_max, shrink=0.9):
    """

    :param z:
    :param n_rounds:
    :param a_max:
    :param b_max:
    :param shrink: so the dot doesn't hit any edge
    :return:
    """
    normal_z = (z - np.amin(z)) / (np.amax(z) - np.amin(z))
    a = a_max * normal_z * shrink
    b = b_max * normal_z * shrink
    x = a * np.cos(normal_z * 2 * n_rounds * np.pi) + a_max
    y = b * np.sin(normal_z * 2 * n_rounds * np.pi) + b_max
    return x, y, z


def spiral_dot_block(shape, n_rounds):
    block = np.zeros(shape)
    a_max = shape[0] // 2
    b_max = shape[1] // 2
    z = np.arange(0., shape[-1])
    for x, y, z in zip(*simple_spiral(z, n_rounds, a_max, b_max)):
        block[int(x), int(y), int(z)] = 1.
    return block


def diagonal_dot_block(shape, dot_num):
    dot_num = int(dot_num)
    block = np.zeros(shape)
    xyz_inds = [np.linspace(0, n - 1, dot_num) for n in shape]
    for x, y, z in zip(*xyz_inds):
        block[int(x), int(y), int(z)] = 1.
    return block


def rand_dot_block_tf(shape, density=0.01):
    block_01 = torch.random.uniform(shape, 0., 1.)
    block = torch.where(block_01 < density, 1., 0.)
    block = block * block_01
    return block


def rand_dot_block_np(shape, density=0.01):
    block_01 = np.random.random(shape)
    block = np.where(block_01 < density, 1., 0.)
    return block


def timer_decorator(func):
    def decorated(*args, **kwargs):
        print("start timing")
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        print('the function {},run for {:.2f}seconds'.format(func.__name__, end - start))
        return ret

    return decorated


def grid_spot_img(m, n, stride=20, brightness=255, t=3):
    """
    read or create grid_spot img for psf viewing
    :param m: num of rows, odds preferred for absolute symmetry
    :param n: odds preferred for absolute symmetry
    :param stride:
    :param brightness: dot gray level
    :param t: num of channel
    :return: grid_you_want img array
    """
    save_name = 'grid{}_{}_{}s{}.jpg'.format(m, n, t, stride)
    try:
        assert cv2.imread(save_name)
    except ValueError:
        print('using stored img file')
        return cv2.imread(save_name)
    except AssertionError:
        print('stored file not found, creating...')
        channel_input = np.zeros((m, n))
        #  fix center
        m_cen, n_cen = m // 2, n // 2
        # channel_input[m_cen, n_cen] = brightness
        for i in range(m_cen):
            for j in range(n_cen):
                if not (i % stride):
                    if not (j % stride):
                        channel_input[m_cen + i, n_cen + j] = channel_input[m_cen - i, n_cen + j] = channel_input[
                            m_cen + i, n_cen - j] = channel_input[
                            m_cen - i, n_cen - j] = brightness
        grid_you_want = np.dstack([channel_input] * t)
        cv2.imwrite(save_name, grid_you_want)
        return grid_you_want


def channel_split_view(input_img, use_colormap=True, channel_last=True, figure_title=None):
    """
    split an input image to see all its channels
    :param input_img:
    :param use_colormap
    :return:None
    """

    img_array = img_path_compatible(input_img)
    img_array = to_numpy(img_array)

    if not channel_last:  # then assume it's channel first
        img_array = np.transpose(img_array, [1, 2, 0])
    m, n, t = img_array.shape
    title_content_dic = dict()
    if not use_colormap:
        for ch in range(t):
            title_content_dic['channel {}'.format(ch)] = img_array[:, :, ch]
        dict_array_plots(cmap='gray', figure_title=figure_title, **title_content_dic)
    else:
        for ch in range(t):
            title_content_dic['channel {}'.format(ch)] = img_array[:, :, ch]
        dict_array_plots(figure_title=figure_title, **title_content_dic)


def im2gray(im, channel_mode='plt'):
    """
    im2gray by some recognized principle
    :param im:
    :param channel_mode:
    :return:gray
    """
    im = img_path_compatible(im)
    RGBweight = np.array([0.299, 0.587, 0.114])

    if channel_mode == 'cv2':
        # cv2 uses BGR order
        w = RGBweight[::-1]
    else:
        w = RGBweight
    gray = np.dot(im, w)
    return gray


def dict_array_plots(cmap=None, axis_off=True, figure_title=None, **title_content_dict):
    """
    plot N subgraphs in auto adapted rows with it's title dictated by the dict passed in
    note that for single channel image input, the plt.imshow() adds colormap by default for it when display
    you may control it with the param: cmap
    :param title_content_dict:
    :return:None
    """
    for key, val in title_content_dict.copy().items():
        if type(val) == str:
            title_content_dict[key] = plt.imread(val)
    N = len(title_content_dict)
    rn = int(np.ceil(np.sqrt(N)))
    cn = N // rn
    if cn < N / float(rn):
        cn += 1
    rn, cn = (rn, cn) if rn <= cn else (cn, rn)
    plt.figure(figure_title)
    for i, (title, channel) in enumerate(title_content_dict.items()):
        plt.subplot(rn, cn, i + 1)
        plt.imshow(channel, cmap=cmap)
        plt.title(title)
        if axis_off:
            plt.axis('off')
    plt.show()


def list_array_plots(img_list, cmap=None):
    dict_array_plots(cmap=cmap, **{f'{i}': img for i, img in enumerate(img_list)})


def stack_imshow(st):
    m, n = st.shape[-2:]
    st = to_numpy(st).reshape(-1, m, n)
    list_array_plots(st)


def cv2_show_img(img, hold=True, colorful=False):
    """
    show img in its real size by openCV in a window
    :param img:
    :param hold:
    :param colorful:
    :return: None
    """
    # print(img)
    img = img_path_compatible(img, 'cv2')
    img = np.array(img, np.uint8)
    # img = img.astype('int32')
    if colorful:
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    cv2.imshow('show_img() ', img)

    # the following 2 lines are essential, in order to display
    if hold:
        cv2.waitKey(0)
    else:
        cv2.waitKey(250)
    cv2.destroyAllWindows()
    # the 2 lines above are essential, in order to display


def img_path_compatible(input, pack_mode='plt'):
    """
    automatic conversion, compatible for image path and img array
    :param input:
    :param pack_mode: lib switch
    :return img_array
    """
    if pack_mode == 'cv2':
        pack = cv2
    else:
        pack = plt
    if type(input) == str:
        img_array = pack.imread(input)
    else:
        img_array = input
    return img_array


def cropImgs(input, imsize):
    assert len(input.shape) == 2
    assert input.shape[0] >= imsize[0] and input.shape[1] >= imsize[1]
    return input[:imsize[0], :imsize[1]]


def ez_anim(img_seq, due=0.01):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('off')
    for i, im in enumerate(img_seq):
        shown = ax.imshow(im)
        ax.set_title('frame {}'.format(i))
        plt.pause(due)
        shown.remove()


def plot(x, y, *args, x_label=None, y_label=None, title=None, grid_on=True, ax=None, **kwargs):
    x = to_numpy(x)
    y = to_numpy(y)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.plot(x, y, *args, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if grid_on:
        ax.grid()
    plt.show()
    return ax


def semilogx(x, y, *args, x_label=None, y_label=None, title=None, grid_on=True, ax=None, **kwargs):
    x = to_numpy(x)
    y = to_numpy(y)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.semilogx(x, y, *args, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if grid_on:
        ax.grid()
    plt.show()
    return ax


def scatter(x, y, x_label=None, y_label=None, title=None, grid_on=True, ax=None, *args, **kwargs):
    x = to_numpy(x)
    y = to_numpy(y)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    ax.sc(x, y, *args, **kwargs)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if grid_on:
        ax.grid()
    plt.show()
    return ax


def ez_plot(y, *args, **kwargs):
    y = to_numpy(y)
    plt.plot(y, *args, **kwargs)


def imshow(img, title=None, *args, colorbar=False, extent=None, **kwargs):
    """
    quick imshow inspectation
    :param img:
    :param title:
    :param args:
    :param colorbar:
    :param extent:
    :param kwargs:
    :return:
    """
    img = to_numpy(img)
    ti = '' if title is None else '"' + title + '"'
    print(f'img {ti} shape :', img.shape, '|| min=', np.amin(img), '|| max=', np.amax(img), )
    plt.figure()
    plt.imshow(img, *args, extent=extent, **kwargs)
    plt.title(title)
    if colorbar:
        plt.colorbar()
    plt.show()


def imshow_xy(img, xy, **kwargs):
    """
    矩阵的列标号向右为x正向，行标号向下为y正向
    :param img:
    :param xy:
    :return:
    """
    (min_x, min_y), _ = torch.min(xy.reshape(-1, 2), dim=0)
    (max_x, max_y), _ = torch.max(xy.reshape(-1, 2), dim=0)
    extent = [min_x, max_x, min_y, max_y]
    imshow(torch.flip(img, dims=(0,)), extent=extent, **kwargs)


def imshow_complex(cplx_img, title=None, *args, **kwargs):
    title = '' if title is None else title
    imshow(torch.real(cplx_img), title='real ' + title, *args, **kwargs)
    imshow(torch.imag(cplx_img), title='imag ' + title, *args, **kwargs)


def im_hist(arr, n_bins=None):
    arr = to_numpy(arr)
    if n_bins is None:
        u_x = np.unique(arr)
        n_bins = len(u_x)
        print(f'{n_bins} unique values:')
        print(u_x)
        print(f'shape :', arr.shape, '| min = ', np.amin(arr), '| max = ', np.amax(arr), )
    if n_bins > 200:
        print('n_bins is too big to plot the histogram, aborted.')
    else:
        plt.figure()
        plt.hist(arr.ravel(), bins=n_bins)
        plt.show()


def close_all():
    plt.close('all')


def phase_wrap_np(tensor_):
    array = to_numpy(tensor_)
    wrapped = np.mod(array, 2 * np.pi)  # if modder=2 * np.pi, np.angle(np.exp(1j * img_mat_))
    return wrapped


def phase_img_std(phase_img):
    phase_img = torch.remainder(phase_img, 2 * np.pi)
    phase_img = phase_img.cpu()
    return phase_img / (2 * np.pi)


def imshow_phase(phase):
    wrapped = phase_wrap_np(phase)
    imshow(wrapped, vmin=0., vmax=2 * np.pi)


# gamma correction
def gamma_correction(img, c=1, g=2.2):
    out = img
    out = (1 / c * out) ** (1 / g)
    out /= out.max()
    out *= 255
    return out


def show_psf_stack(psf_3d, wavelengths, scene_distances, show_colorbar=False):
    if type(psf_3d) is not np.ndarray:
        psf_3d = psf_3d.cpu().detach().numpy()
    if type(wavelengths) is not np.ndarray:
        wavelengths = wavelengths.cpu().detach().numpy()
    if type(scene_distances) is not np.ndarray:
        scene_distances = scene_distances.cpu().detach().numpy()

    C, D, H, W = psf_3d.shape

    fig = plt.figure()
    if D > 1:
        for wl_i in range(C):
            for depth_i in range(D):
                plt.subplot(C, D, D * wl_i + depth_i + 1)
                plt.imshow(psf_3d[wl_i, depth_i, ...])
                plt.xticks([])
                plt.yticks([])
                wavelength = 1e9 * wavelengths[wl_i]
                if depth_i == 0:
                    plt.ylabel('%d nm' % wavelength)

                depth = scene_distances[depth_i]
                if wl_i == 0:
                    plt.title('%.2f m' % depth)
                if show_colorbar:
                    plt.colorbar()
    else:
        for depth_i in range(D):
            for wl_i in range(C):
                plt.subplot(D, C, wl_i + D * depth_i + 1)
                plt.imshow(psf_3d[wl_i, depth_i, ...])
                plt.xticks([])
                plt.yticks([])
                wavelength = 1e9 * wavelengths[wl_i]
                if depth_i == 0:
                    plt.title('%d nm' % wavelength)

                depth = scene_distances[depth_i]
                if wl_i == 0:
                    plt.ylabel('%.2f m' % depth)
                if show_colorbar:
                    plt.colorbar()
    plt.tight_layout(pad=0)
    plt.show()
    return fig


def show_psf_rgb(psf_3d, scene_distances, stretch=True, show_colorbar=False):
    if type(psf_3d) is not np.ndarray:
        psf_3d = psf_3d.cpu().detach().numpy()
    if type(scene_distances) is not np.ndarray:
        scene_distances = scene_distances.cpu().detach().numpy()

    C, D, H, W = psf_3d.shape

    if stretch:
        psf_3d = psf_3d / psf_3d.max(axis=(-1, -2, 0), keepdims=True)
    psf_3d = regular_img(psf_3d, colorful=True)

    fig = plt.figure()
    for depth_i in range(D):
        plt.subplot(1, D, depth_i + 1)
        plt.imshow(psf_3d[:, depth_i, ...].transpose(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
        depth = scene_distances[depth_i]
        plt.title('%.2f m' % depth)
        if show_colorbar:
            plt.colorbar()
    plt.tight_layout(pad=0)
    plt.show()
    return fig


def mask_shift_8_dir(mask):
    """
    func shifting a mask arr in 8 directions
    # the 4 slicing index (a,b,c,d)in the '[a:b,c:d]' above
    # a mask arr in 8 directions, folling the directions 1~8, see the mizi_anti_clock_shift_index comments
    :param mask:
    :return:
    """
    mizi_anti_clock_shift_index = (  # (a,b,c,d)in the '[a:b,c:d]' above
        (2, None, 1, -1),  # s,1
        (2, None, 0, -2),  # ws,2
        (1, -1, 0, -2),  # w,3
        (0, -2, 0, -2),  # nw,4
        (0, -2, 1, -1),  # n,5
        (0, -2, 2, None),  # en,6
        (1, -1, 2, None),  # e,7
        (2, None, 2, None),  # se,8
    )
    # mind that plt.imshow() filp the direction of y axis, visual therefore filps clockwise with it
    mask_core = mask[1:-1, 1:-1, ...]
    shift_mask_list = []
    for a, b, c, d in mizi_anti_clock_shift_index:
        shifted_mask = torch.zeros_like(mask)
        shifted_mask[a:b, c:d] = mask_core
        shift_mask_list.append(shifted_mask)
    return shift_mask_list


def nearsest_resize_times(org_img, times):
    m, n = org_img.shape
    resized_times = cv2.resize(org_img, dsize=(times * m, times * n), interpolation=cv2.INTER_NEAREST)
    return resized_times


if __name__ == '__main__':
    """make and save inscribed .bmp file of the target field """
    from imageio import imsave, imread

    img_file_path = r'.\\datasets\target_pictures\house_original_4.1.05.tiff'
    resize_shape = (32, 32)
    input_gray_image = im2gray(img_file_path, 'cv2')
    org_size = input_gray_image.shape
    # resize
    want_gray = cv2.resize(input_gray_image, resize_shape) if resize_shape is not None else input_gray_image
    print('input size {} reshaped to {}'.format(org_size, resize_shape))
    if True:
        target_field = inscrib_circle_cropped(want_gray)
    imsave(r'.\\datasets\target_pictures\saved_house_target_in_circle.bmp', target_field)
    read = imread(r'.\\datasets\target_pictures\saved_house_target_in_circle.bmp')
    imshow(read)  # use windows draw to see the bmp without interp effect
