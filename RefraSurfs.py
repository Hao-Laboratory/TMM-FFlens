import torch
import imgTools
import pointArrays
import util
import visTools
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from matplotlib import cm
from warnings import warn

surf_cmap = None


# surf_cmap = cm.gist_earth


# other colormap ['flag', 'prism', 'ocean', 'gist_earth', 'terrain',
#  'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap',
#  'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet',
#  'turbo', 'nipy_spectral', 'gist_ncar']


def snells_law(ni, no, iv, nv):
    """
    # snell's law
    ref：https://physics.stackexchange.com/questions/435512/snells-law-in-vector-form
    :param ni: refractive index on incident side
    :param no: refractive index on the out(exit) side
    :param iv:(...,3) incident vector
    :param nv:(...,3) normal vector of plane fragment
    :return: ov: (...,3) output(exit) vector, by the math here, the output ov is normalized
    """

    norm_nv = torch.norm(nv, dim=-1, keepdim=True)
    norm_iv = torch.norm(iv, dim=-1, keepdim=True)

    zero_nv = norm_nv == 0.
    if torch.any(zero_nv):
        print('zero vectors at:')
        print(torch.where(zero_nv))
        raise ValueError('normal vector should never be zero vector.')

    nv = nv / norm_nv  # nv
    iv = iv / norm_iv  # iv

    iv_dot_nv = torch.sum(iv * nv, dim=-1, keepdim=True)
    nv = torch.where(iv_dot_nv < 0., -nv, nv)
    iv_dot_nv = torch.abs(iv_dot_nv)

    u = ni / no
    nv_dot_ov_square = 1 - u ** 2 * (1 - iv_dot_nv ** 2)
    total_internal_reflection = nv_dot_ov_square < 0
    nv_dot_ov_positive = torch.sqrt(nv_dot_ov_square)
    ov = nv_dot_ov_positive * nv + u * (iv - iv_dot_nv * nv)
    # this ov here is already normalized
    if torch.any(total_internal_reflection):
        print('total internal reflection occurred in these indices of incident vec')
        print(torch.where(total_internal_reflection))
        warn('total internal reflection occurrences replaced with facet norm vector')
        ov = torch.where(total_internal_reflection, nv, ov)  # TIR filtered

    return ov


def fresnel_transmission_ratio(th1, n1=1.491756, n2=1):
    sinth1 = torch.sin(th1)
    costh1 = torch.cos(th1)
    sqrt_term = torch.sqrt(1 - (n1 / n2 * sinth1) ** 2)
    n1sqrt_term = n1 * sqrt_term
    n2sqrt_term = n2 * sqrt_term
    n1costh1 = n1 * costh1
    n2costh1 = n2 * costh1
    Rs = ((n1costh1 - n2sqrt_term) / (n1costh1 + n2sqrt_term)) ** 2
    Rp = ((n1sqrt_term - n2costh1) / (n1sqrt_term + n2costh1)) ** 2
    return (2 - Rs - Rp) / 2
    # return (1 - Rs+1 - Rp)/2, Rs, Rp


class CircleParallelSourcy(pl.LightningModule):
    """

    :param aper_r aperture radius
    :param num_want_mizi sys finds nearest val >= num_want_mizi to fill aperture
    :param ni: refractive index on incident side, 1.491756 PMMA @ D-light 589nm as default
    :param no: refractive index on the out(exit) side
    :param allow_xy_move: default False, if true this is allowing xy to be set as variables too 
    """

    def __init__(self, aper_r, num_want_mizi, ni=1.491756, no=1.,
                 in_k=torch.tensor([0., 0., 1.]), allow_xy_move=False):
        super(CircleParallelSourcy, self).__init__()
        self.num_want_mizi = num_want_mizi
        self.aper_r = aper_r
        self.ni = ni
        self.no = no
        self.register_buffer('in_k', in_k)
        self.allow_xy_move = allow_xy_move

        edge_num = int(np.sqrt(num_want_mizi / 3. * 4)) - 1
        if edge_num < 1:
            edge_num = 1
        num_mizi = 0
        while num_mizi < num_want_mizi:
            edge_num += 1
            facet_bound_array = pointArrays.xy_range_array(edge_num, edge_num, -1., 1., -1., 1.)
            x, y = torch.split(facet_bound_array, 1, dim=-1)
            r = torch.sqrt(x ** 2 + y ** 2)
            r = r.squeeze(0)
            mask = r < 1.

            edge_vertex_num = edge_num * 2 - 1
            vertex_xy_array = pointArrays.xy_range_array(edge_vertex_num, edge_vertex_num, -1., 1., -1., 1.)
            vertex_array_mask = torch.nn.functional.interpolate(
                mask.to(torch.float).squeeze().unsqueeze(0).unsqueeze(0),
                scale_factor=2,
                mode='nearest', align_corners=None).squeeze()
            vertex_array_mask_left_shift = torch.zeros_like(vertex_array_mask, dtype=torch.bool)
            vertex_array_mask_left_shift[:, :-1, ...] = vertex_array_mask[:, 1:, ...]
            vertex_array_mask_horizontal = torch.logical_and(vertex_array_mask,
                                                             vertex_array_mask_left_shift)[:-1, :-1, ...]

            vertex_array_mask_up_shift = torch.zeros_like(vertex_array_mask, dtype=torch.bool)
            vertex_array_mask_up_shift[:-1, :, ...] = vertex_array_mask[1:, :, ...]
            vertex_array_mask_vertical = torch.logical_and(vertex_array_mask, vertex_array_mask_up_shift)[:-1, :-1, ...]

            vertex_array_mask = torch.logical_and(vertex_array_mask_horizontal, vertex_array_mask_vertical)
            sub_quater_edge_num = edge_vertex_num // 2
            vertex_array_mask[-sub_quater_edge_num:, -sub_quater_edge_num:, ...] = torch.flip(
                vertex_array_mask[:sub_quater_edge_num, :sub_quater_edge_num, ...], [0, 1])

            odd_true_mask = torch.zeros_like(vertex_array_mask, dtype=torch.bool)
            odd_true_mask[1::2, 1::2] = True
            # imgTools.imshow(odd_true_mask, 'odd_true_mask')

            mizi_centers_mask = torch.logical_and(vertex_array_mask, odd_true_mask)
            num_mizi = len(mizi_centers_mask[mizi_centers_mask == True])
        # > find nearest num_mizi>=num_want_mizi
        self.edge_num = edge_num
        self.pix_width = 2. * self.aper_r / edge_num
        self.smooth_cos_thresh = np.pi / 14  # 14 is the edge_num of want_mizi=100 tuned
        # self.clip_thick_bounds = (-self.pix_width / 2, self.pix_width / 2)



        self.diff_regu_thresh = self.pix_width / 2.
        self.num_mizi = num_mizi
        self.edge_vertex_num = edge_vertex_num
        self.vertex_array_mask = vertex_array_mask
        self.mizi_centers_mask = mizi_centers_mask
        # print('num_mizi=', num_mizi)  # deb
        # imgTools.imshow(mask, 'mask')  # deb
        # imgTools.imshow(mizi_centers_mask, 'mizi_centers_mask')

        vertex_xy_in_circle = vertex_xy_array[vertex_array_mask] * aper_r
        self.register_buffer('vertex_xy_in_circle', vertex_xy_in_circle)
        self.z_in_circle = torch.nn.Parameter(  # z as variable
            torch.zeros_like(vertex_xy_in_circle[..., 0:1])
        )
        if allow_xy_move:  # make xy also variable
            self.vertex_xy_in_circle = torch.nn.Parameter(vertex_xy_in_circle)  #
        self.shift_8_center_masks = imgTools.mask_shift_8_dir(
            self.mizi_centers_mask)
        self.refresh_vertex_arr()

    def refresh_vertex_arr(self):
        self.vertex_in_circle_3d = torch.cat([self.vertex_xy_in_circle, self.z_in_circle], dim=-1)
        vertex_with_var_array = torch.zeros((self.edge_vertex_num, self.edge_vertex_num, 3), device=self.device)
        vertex_with_var_array[self.vertex_array_mask] = self.vertex_in_circle_3d
        self.vertex_with_var_array = vertex_with_var_array  # shape=(self.edge_vertex_num, self.edge_vertex_num, 3)
        self.mizi_centers = vertex_with_var_array[self.mizi_centers_mask]
        around_8_points_seq_list = [self.vertex_with_var_array[shift_mask] for shift_mask in self.shift_8_center_masks]


        self.around_8_points_tensor = torch.stack(around_8_points_seq_list, dim=0)  # indice 0~7
        self.around_8_points_shifted_tensor = torch.stack(around_8_points_seq_list[1:] + [around_8_points_seq_list[0]],
                                                          dim=0)  # indice 1~7，0

        mizi_centers_8_copies, _ = torch.broadcast_tensors(self.mizi_centers, self.around_8_points_tensor)
        self.mizi_facet_group_vertex = torch.stack([mizi_centers_8_copies,
                                                    self.around_8_points_tensor,
                                                    self.around_8_points_shifted_tensor],
                                                   dim=0)  # (3, 8, num_mizi, 3)
        return self.mizi_facet_group_vertex  # torch.Size([3, 8, 32, 3])


    def plot_vertex_3d(self):
        self.refresh_vertex_arr()
        ax = visTools.scatter_xyz(self.vertex_with_var_array)
        visTools.scatter_xyz(self.mizi_centers, color='g', size=100, ax=ax)

    def facet_normals(self):

        b = self.around_8_points_shifted_tensor - self.around_8_points_tensor  # torch.Size([8, num_mizi, 3])
        a = self.around_8_points_tensor - self.mizi_centers
        # a = util.write_2d_vec_as_3d(a)
        # b = util.write_2d_vec_as_3d(b)  
        n_vec_facet = torch.cross(a, b)
        # no normalization needed here, snell's law func has it
        #
        return n_vec_facet  # torch.Size([8, num_mizi, 3])

    def out_p_k_parallel_in(self):
        """

        :param in_k: incident direction vector (okay with non-normalized)
        :return:
        """
        p = self.refresh_vertex_arr()  # torch.Size([3, 8, num_mizi, 3])
        n_vec_facet = self.facet_normals()  # torch.Size([8, num_mizi, 3])
        out_k = snells_law(self.ni, self.no, self.in_k, n_vec_facet)
        return p, out_k

    def forward(self):
        p, out_k = self.out_p_k_parallel_in()
        # the following lines put all points on one dim
        p, out_k = torch.broadcast_tensors(p, out_k)
        # p = torch.reshape(p, (-1, 3)) #
        # out_k = torch.reshape(out_k, (-1, 3))
        p = p.permute(2, 1, 0, 3)
        out_k = out_k.permute(2, 1, 0, 3)
        return p, out_k  # torch.Size([num_mizi,8<facets>,3<vertex of each facet>,  3<xyz dims>]

    def clip_z_of_vertexes(self):
        with torch.no_grad():  # clipping
            self.z_in_circle.clamp_(self.clip_thick_bounds[0], self.clip_thick_bounds[1])

    def local_smooth_thresh_cos_regu(self):
        regu = util.local_thresh_cos_loss(self.vertex_with_var_array, self.smooth_cos_thresh)
        return regu

    def local_diff_relu_regu(self):
        regu = util.local_diff_loss(self.vertex_with_var_array, self.diff_regu_thresh)
        return regu

    def dummy_test_output(self):
        p, out_k = self.forward()
        return torch.sum(p) + torch.sum(out_k)


class Traditional(pl.LightningModule):
    """

    """

    def __init__(self, init_xyz):
        super(Traditional, self).__init__()
        self.x, self.y, z = torch.split(init_xyz, 1, dim=-1)
        self.z = torch.nn.Parameter(z)
        self.lu_code, self.ld_code, self.rd_code, self.ru_code = [torch.tensor(item) for item in
                                                                  [[0, 1, 3], [0, 1, 2], [1, 2, 3], [0, 2, 3]]]
        (m, n, t) = self.x.shape
        J, I = torch.meshgrid(torch.arange(n - 1), torch.arange(m - 1))
        I = I.permute(1, 0)
        self.I = I.unsqueeze(-1)
        shift_unit = (self.x[0, 1, 0] - self.x[0, 0, 0]) / 4.
        __, II = torch.meshgrid(torch.arange(n), torch.arange(m))
        II = II.permute(1, 0).unsqueeze(-1)
        self.x = torch.where(II % 2 == 0,
                             self.x - shift_unit,
                             self.x + shift_unit)

        self.y = self.y * np.sqrt(3.) / 2.

        self.make_3p_set()
        # init of 'self.surf_points' and 'self.triangle_surf_3p' are included for instant plotting after init

    def forward(self, iv):
        return self.mean_y()

    def mean_y(self):
        return torch.max(self.z) - torch.min(self.z)

    def xyz(self):
        return torch.cat([self.x, self.y, self.z], dim=-1)

    def make_3p_set(self):  #
        self.surf_points = self.xyz()  # refresh
        self.srd_point_set = torch.stack(
            [
                self.surf_points[:-1, :-1, :],
                self.surf_points[1:, :-1, :],
                self.surf_points[1:, 1:, :],
                self.surf_points[:-1, 1:, :]
            ],
            dim=0)

        lu_rd_3p_set = torch.stack(
            [
                self.srd_point_set[self.lu_code],
                self.srd_point_set[self.rd_code]
            ],
            dim=0)
        ru_ld_3p_set = torch.stack(
            [
                self.srd_point_set[self.ru_code],
                self.srd_point_set[self.ld_code]
            ],
            dim=0)  # (2<ru,ld>,3<points>,m-1,n-1,3<vec dims>)
        self.triangle_surf_3p_ = torch.where(self.I % 2 == 0,
                                             lu_rd_3p_set,
                                             ru_ld_3p_set)
        # (2<u,d>,3<points>,m-1,n-1,3<vec dims>)

        # self.triangle_surf_3p_ = lu_rd_3p_set
        self.triangle_surf_3p = self.triangle_surf_3p_.permute(2, 3, 0, 1, 4)
        # (m-1,n-1，2<ru,ld>,3<points>,3<vec dims>)

    def face_centers(self):
        return torch.mean(self.triangle_surf_3p, dim=-2, keepdim=True)

    def normals(self):
        """calc face normals from self.triangle_surf_3p"""

        o_ = self.triangle_surf_3p[..., 0:1, :]  # (m-1,n-1，2<ru,ld>,3<points>,3<vec dims>)
        a = self.triangle_surf_3p[..., 1:2, :] - o_  # (m-1,n-1，2<ru,ld>,3<points>,3<vec dims>)
        b = self.triangle_surf_3p[..., 2:, :] - o_  # (m-1,n-1，2<ru,ld>,3<points>,3<vec dims>)
        nv_ = torch.cross(b, a, dim=-1)
        # torch.testing.assert_allclose(torch.sum(nv_*(b-a),dim=-1),0.) #test
        return nv_

    def ov_with_parallel_iv(self, ni, no, parallel_iv=torch.tensor([0., 0., 1.])):
        """
        refract using normal
        :param ni:
        :param no:
        :param parallel_iv: direction vector of the parallel incident beam
        :return ov: exit direction vectors w.r.t each surface fragment
        """
        assert parallel_iv.shape[-1] == 3  # to make sure it can broadcast with nv
        nv = self.normals()
        return snells_law(ni, no, parallel_iv, nv)

    def make_triangulation(self):
        """
        different from the self.make_3p_set(), this use a number matrix
        :return:
        """
        self.surf_points = self.xyz()
        sp_shape = self.surf_points.shape
        first_2_shape = sp_shape[:-1]
        numbered_mat = torch.reshape(torch.arange(util.deshape(first_2_shape)), first_2_shape)
        srd_number_set = torch.stack(
            [
                numbered_mat[:-1, :-1],
                numbered_mat[1:, :-1],
                numbered_mat[1:, 1:],
                numbered_mat[:-1, 1:]
            ]
            , dim=0)

        lu_rd_3p_number_set = torch.stack(
            [
                srd_number_set[self.lu_code],
                srd_number_set[self.rd_code]
            ],
            dim=0)  # (2<lu,rd>,3<points>,m-1,n-1,3<vec dims>)
        ru_ld_3p_number_set = torch.stack(
            [
                srd_number_set[self.ru_code],
                srd_number_set[self.ld_code]
            ],
            dim=0)  # (2<ru,ld>,3<points>,m-1,n-1,2<vec dims>)

        triangle_surf_3p_number_ = torch.where(torch.squeeze(self.I) % 2 == 0,
                                               lu_rd_3p_number_set,
                                               ru_ld_3p_number_set)

        # triangle_surf_3p_number_ = lu_rd_3p_number_set
        triangle_surf_3p_number = triangle_surf_3p_number_.permute(0, 2, 3, 1)
        triangles = torch.reshape(triangle_surf_3p_number, (-1, 3))
        sp = self.surf_points
        spx, spy, spz = sp[:, :, 0].detach().numpy().ravel(), sp[:, :, 1].detach().numpy().ravel(), sp[:, :,
                                                                                                    2].detach().numpy().ravel()
        self.triangles = triangles
        self.sp_raveled = (spx, spy, spz)
        return self.triangles, self.sp_raveled

    def plot_triangulation(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        tris, (xp, yp, zp) = self.make_triangulation()
        for trii in tris:
            ttrii = trii.numpy().tolist()
            ttrii.append(ttrii[0])
            plt.plot(xp[ttrii], yp[ttrii], color='b')
        plt.show()

