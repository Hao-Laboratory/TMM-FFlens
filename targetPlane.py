import numpy as np
import torch

import imgTools
import pointArrays
import util
from util import write_2d_vec_as_3d, normalized, standardized, deshape, relu, round_gaussian_spot, \
    xy_seen_as_in_unit_box, sigmoid_step
from pointArrays import xy_range_array
from imgTools import stack_imshow
import pytorch_lightning as pl
import torch.nn.functional as F


pl.seed_everything(666)


def triangular_prism_function(v_2d, p, steepness=1.):
    """

    :param v_2d: aka point_group_3p_2d vertices, shape=(...,3,2).
    e.g.(6,3,2) means 6 spots, each have 3 vertex, each vertex 2 dims
    :param p: shape=(m,n,2) sampling point array on the plane
    :param steepness: the bigger the steeper
    :return: light spot for each 3p_vertex_group(3,2) (...,m,n)

    """

    v_3d = write_2d_vec_as_3d(v_2d)  # (...,3,3)
    p = write_2d_vec_as_3d(p)  # (m,n,3)
    v = v_3d.unsqueeze(-2).unsqueeze(-2)  # (...,3,1,1,3)vertices
    v_shift = v[..., [1, 2, 0], :, :, :]
    side_v = normalized(v_shift - v)  # (...,3<edges>,1,1,3) side vectors of triangles
    v_p = p - v  # # (...,3,m,n,3) vertices =--------> p
    side_v, v_p = torch.broadcast_tensors(side_v, v_p)
    cross__side_v__v_p = torch.cross(side_v, v_p, dim=-1)  # (...,3,m,n,3)
    p_to_line_distance = cross__side_v__v_p[..., -1]  # (...,3,m,n)
    # p_to_line_distance = torch.sum(cross__side_v__v_p * torch.tensor([[[0., 0., 1.]]]), dim=-1)
    side_steps = sigmoid_step(p_to_line_distance, steepness=steepness)  # (...,3,m,n)
    tri_hats_ = torch.prod(side_steps, dim=-3)  # (...,m,n)
    areas = util.area_tri_3p_set(v_3d)
    tri_hats = tri_hats_ / areas
    return tri_hats

class CircleBasinLoss(pl.LightningModule):
    def __init__(self, aper_r):
        super(CircleBasinLoss, self).__init__()
        self.register_buffer('aper_r', torch.tensor(aper_r))
        self.register_buffer('range_xy_min', -torch.tensor([aper_r, aper_r]))
        self.register_buffer('range_xy_max', torch.tensor([aper_r, aper_r]))

    def forward(self, p_des_xy):
        p_des_xy_standardized = xy_seen_as_in_unit_box(p_des_xy, self.range_xy_min, self.range_xy_max)
        x = p_des_xy_standardized[..., 0]
        y = p_des_xy_standardized[..., 1]
        rho = torch.sqrt(x ** 2 + y ** 2)  # (...)
        z = relu(rho - 1)  # (...)

        return z

class RectangleBasinLoss(pl.LightningModule):
    def __init__(self, range_xy_min, range_xy_max):

        super(RectangleBasinLoss, self).__init__()
        self.register_buffer('range_xy_min', torch.tensor(range_xy_min))
        self.register_buffer('range_xy_max', torch.tensor(range_xy_max))
        # xmin, ymin=range_xy_min
        # xmax, ymax=range_xy_max

        unit_sqare_edge_vecs = torch.tensor([
            [-1, 0],
            [0, -1],
            [1, 0],
            [0, 1],
        ], dtype=torch.float32)  # anti-clockwise# (4, 2)
        unit_sqare_corner_points = torch.tensor([
            [1, 1],
            [0, 1],
            [0, 0],
            [1, 0],
        ], dtype=torch.float32)  # anti-clockwise# (4, 2)

        self.register_buffer('unit_sqare_edge_vecs_3d', write_2d_vec_as_3d(unit_sqare_edge_vecs))  # (4,3)
        self.register_buffer('unit_sqare_corner_points', write_2d_vec_as_3d(unit_sqare_corner_points))  # (4,3)

    def forward(self, p_des_xy):
        p_des_xy_standardized = xy_seen_as_in_unit_box(p_des_xy, self.range_xy_min, self.range_xy_max)
        p = write_2d_vec_as_3d(p_des_xy_standardized).unsqueeze(-2)  # (...,1,3)
        v_p = p - self.unit_sqare_corner_points  # (...,4,3)
        side_v, v_p = torch.broadcast_tensors(self.unit_sqare_edge_vecs_3d, v_p)  # (...,4,3)
        cross__side_v__v_p = torch.cross(side_v, v_p, dim=-1)  # (...,4,3)
        p_to_line_distance = cross__side_v__v_p[..., -1]  # (...,4)

        out_side_slopes = relu(-p_to_line_distance)  # (...,4)
        z = torch.sum(out_side_slopes, dim=-1)
        return z


def basin_of_triangle_set(tri_v, eval_xy, range_xy_min, range_xy_max):
    """

    :param tri_v: triangle vertices shape=(..., 3, 2) should be in anti-clockwise order
        , you may use anti_clockwise_triangle_vertices() to ensure that
    :param eval_xy: xy position where you want to evaluate z = basin(eval_x, eval_y), shape=(...,2)
    :param  range_xy_min,
    :param  range_xy_max:

    :return: z = basin(eval_x, eval_y)

    """

    num_tri = deshape(tri_v.shape[:-2])
    tri_v = tri_v.reshape(num_tri, 3, 2)

    v = xy_seen_as_in_unit_box(tri_v, range_xy_min, range_xy_max)
    v = write_2d_vec_as_3d(v)  # (...,3,3)
    p = write_2d_vec_as_3d(eval_xy)  # (m,n,3)
    v = v.unsqueeze(-2)  # (...,3,  1,3)vertices
    v = v.unsqueeze(-2)  # (...,3,1,1,3)vertices
    v_shift = v[..., [1, 2, 0], :, :, :]
    side_v = normalized(v_shift - v)  # (...,3<edges>,1,1,3) side vectors of triangles
    v_p = p - v  # # (...,3,m,n,3) vertices =--------> p
    side_v, v_p = torch.broadcast_tensors(side_v, v_p)  # torch.cross does not broadcast automatically
    cross__side_v__v_p = torch.cross(side_v, v_p, dim=-1)  # (...,3,m,n)# (...,3,m,n)
    p_to_line_distance = cross__side_v__v_p[..., -1]

    out_side_slopes = relu(-p_to_line_distance)  # (...,3,m,n)
    tri_basins = torch.sum(out_side_slopes, dim=-3)  # (...,m,n)
    z = torch.prod(tri_basins, dim=0) ** (1. / num_tri)
    return z


class RectangleZ(pl.LightningModule):
    """
    a plane perpendicular with z axis with no bounds
    """

    def __init__(self, z):
        super(RectangleZ, self).__init__()
        self.z = z

    def intersect_ray_p_k(self, p, k):
        """
        
        :param p: point of ray field: (...,3)
        :param k: k of ray field: (...,3) The direction of the applied light field
        :return: intersect_p :intersections (...,3)
        """

        z_p = p[..., -1:]  # (..., 1)
        z_k = k[..., -1:]  # (..., 1)
        m = (self.z - z_p) / z_k  # (..., 1)
        # m = torch.expand_dims(m, axis=-1)
        des_p = p + m * k

        return des_p


def anti_clockwise_triangle_vertices(v_2d):
    """
    ensuring the v_2d (vertices of each triangle) to have anti-clockwise following the order of them in v_2d alone dim:-2
    this check and corrects the order
    :param v_2d: 2d vertices  shape = (..., 3<vertices>, 2<dims>>)
    :return: (..., 3<vertices>, 2<dims>>)
    """

    v_ = write_2d_vec_as_3d(v_2d)
    v_0, v_1, v_2 = torch.split(v_, 1, dim=-2)
    a_ = v_1 - v_0  # (...,1<points>,3<vec dims>)
    b_ = v_2 - v_0  # (...,1<points>,3<vec dims>)
    a__cross_b_ = torch.cross(a_, b_, dim=-1)  # (...,1<vec>,3<vec dims>)
    z_ = a__cross_b_[..., 2:]
    z_positive_cond = z_ >= 0
    if not torch.all(z_positive_cond):
        v1 = torch.where(z_ >= 0, v_1, v_2)
        v2 = torch.where(z_ >= 0, v_2, v_1)
        v = torch.cat([v_0, v1, v2], dim=-2)
        print('Order corrected---anti_clockwise_triangle_vertices')
    else:
        v = v_
        print('NOTHING TO CHANGE---anti_clockwise_triangle_vertices')
    return v[..., :-1]  # (...,1<points>,2<vec dims>)

