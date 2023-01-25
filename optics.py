import os
import copy
import torch
import RefraSurfs
import targetPlane
import imgTools
import util
import visTools
import pointArrays
import pytorch_lightning as pl
from torchmetrics.functional import psnr, ssim
from data.illum_sys_dummy_dataset import y_illum_field_extract
from matplotlib import pyplot as plt


class Parallel_1ffSurf_Picture(pl.LightningModule):
    """
    :param hps hyperparms, a parsed arg obj

    """

    def __init__(self, hps, round_basin=True):

        """

        """
        super(Parallel_1ffSurf_Picture, self).__init__()
        # attrs
        self.hps = copy.deepcopy(hps)
        self.save_hyperparameters(self.hps)
        self.lr = float(hps.lr)
        # self.lr_decay_period = float(hps.lr_decay_period)
        # self.lr_decay_factor = float(hps.lr_decay_factor)
        # self.image_log_period = int(hps.image_log_period)
        self.steepness = hps.steepness
        self.illum_field_resolution = hps.resize_times * hps.target_img_res
        self.field_weight_factor = hps.field_weight_factor

        #  sub models
        self.ff_surf = RefraSurfs.CircleParallelSourcy(hps.aper_r,
                                                       hps.num_want_mizi,
                                                       allow_xy_move=hps.allow_xy_move)
        self.target_plane = targetPlane.RectangleZ(hps.target_plane_z)
        if round_basin:
            self.basin = targetPlane.CircleBasinLoss(self.hps.aper_r)
        else:
            self.basin = targetPlane.RectangleBasinLoss(
                [hps.basin_xmin, hps.basin_ymin],
                [hps.basin_xmax, hps.basin_ymax])
        self.register_buffer('field_sample_array', pointArrays.xy_range_array(self.illum_field_resolution,
                                                                              self.illum_field_resolution,
                                                                              hps.basin_xmin,
                                                                              hps.basin_xmax,
                                                                              hps.basin_ymin,
                                                                              hps.basin_ymax))
        self.register_buffer('y_illum_field', torch.tensor(y_illum_field_extract(hps)))
        self.register_buffer('cached_illum_field', torch.zeros_like(self.y_illum_field))

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):


        optimizer.step()

    def configure_optimizers(self):
        if self.hps.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.hps.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError

        # self.logger.experiment.add_text(  # log_hyperparams() fails, searched, it is a bug, so use text instead
        #     'hparam_str',
        #     str({
        #         "lr": self.lr,
        #         "lr_decay_period": self.lr_decay_period,
        #         # "steepness": self.hps.steepness,
        #         # "optimizer": self.hps.optimizer,
        #         # "field_weight_factor": self.hps.field_weight_factor, "resize_times": self.hps.resize_times
        #     }),
        #     0)
        #

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.9,
            patience=10,
            cooldown=5,
            threshold=0.005,
            threshold_mode='abs',
            min_lr=1e-9,
            verbose=True,

        )
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'validation/MSE'}

    def trace_intersections(self):
        p, k = self.ff_surf()
        p_des = self.target_plane.intersect_ray_p_k(p, k)
        p_des_xy = p_des[..., :-1]
        return p_des_xy

    def render_illumfield(self, p_des_xy):
        prism_stack = targetPlane.triangular_prism_function(p_des_xy, self.field_sample_array, self.steepness)
        illum_field = prism_stack.sum(0).sum(0)
        illum_field = illum_field / illum_field.max()  # standardization
        return illum_field

    def render_non_blur_illumfield(self, p_des_xy, steepness=100):
        prism_stack = targetPlane.triangular_prism_function(p_des_xy, self.field_sample_array, steepness)
        illum_field = prism_stack.sum(0).sum(0)
        illum_field = illum_field / illum_field.max()  # standardization
        return illum_field

    def basin_regulation(self, p_des_xy):
        # this is for energy conservation and efficiency
        z = self.basin(p_des_xy)
        punishment = z.mean()
        return punishment

    def forward(self, *args, **kwargs):
        p_des_xy = self.trace_intersections()
        illum_field = self.render_illumfield(p_des_xy)
        # imgTools.imshow(illum_field) # deb
        return p_des_xy, illum_field

    def training_step(self, batch, *args, **kwargs):

        p_des_xy, illum_field = self.forward()
        self.cached_illum_field = illum_field
        basin_regu = self.basin_regulation(p_des_xy)

        y_illum_field = self.y_illum_field  # when using dummy or none dataloader
        illum_field_mse = torch.nn.functional.mse_loss(illum_field, y_illum_field)
        # smooth_regu = self.ff_surf.local_smooth_thresh_cos_regu()
        smooth_regu = self.ff_surf.local_diff_relu_regu()
        loss = basin_regu + smooth_regu + self.field_weight_factor * illum_field_mse
        # an L2 loss and a#
        self.log('train/basin_regu', basin_regu)
        self.log('train/smooth_regu', smooth_regu)
        self.log('train/illum_field_mse', illum_field_mse)

        self.log('train_loss', loss)  # keep this one, ckpt uses it!
        # return {'p_des_xy': p_des_xy, 'y_illum_field': y_illum_field, 'illum_field': illum_field, 'loss': loss} # use with self.training_step_end()
        return loss

    def on_train_end(self):  # log end status illumfield
        p_des_xy, illum_field = self.forward()
        self.logger.experiment.add_image(f'illum_field', illum_field.unsqueeze(0), self.global_step)

    def validation_step(self, batch, *args, **kwargs):

        # y_illum_field = batch.squeeze()
        y_illum_field = self.y_illum_field  # when using dummy or none dataloader
        illum_field = self.cached_illum_field
        self.logger.experiment.add_image(f'illum_field', illum_field.unsqueeze(0), self.global_step)
        mat_save_path = os.path.join(self.logger.log_dir, '{:08d}_step.mat'.format(self.global_step))
        util.mat_easy_save(illum_field, mat_save_path)
        # ↑ mind that ensure upper level folders exit before this
        mse_val = torch.nn.functional.mse_loss(illum_field, y_illum_field)
        self.log('validation/MSE', mse_val)
        psnr_val = psnr(illum_field.unsqueeze(0).unsqueeze(0),
                        y_illum_field.unsqueeze(0).unsqueeze(0))
        ssim_val = ssim(illum_field.unsqueeze(0).unsqueeze(0),
                        y_illum_field.unsqueeze(0).unsqueeze(0))
        self.log('validation/PSNR', psnr_val)
        self.log('validation/SSIM', ssim_val)

        self.log('lr', self.optimizers().state_dict()['param_groups'][0]['lr'])

        # return {'mse':mse_val,'ssim':ssim_val, 'psnr':psnr_val }

    # def validation_epoch_end(self, outputs): # optional
    #     oout_put = outputs[0]
    #     pass


if __name__ == '__main__':
    pass

    #
    # # num_want_mizi = 32 ** 2  # deb
    # num_want_mizi = 32  # deb
    # z_target_plane = 300
    # sample_half_width = 20.
    # illum_field_resolution = 128
    # steepness = 100
    # a_surf = RefraSurfs.CircleParallelSourcy(25.4, num_want_mizi=num_want_mizi)
    # a_plane = targetPlane.RectangleZ(z_target_plane)
    # p, k = a_surf()
    # p_des = a_plane.intersect_ray_p_k(p, k)
    # p_des_xy = p_des[..., :-1]
    #
    # prism_stack = targetPlane.triangular_prism_function(p_des_xy, field_sample_array, steepness)
    # #  using prism func to make field
    # imgTools.imshow(prism_stack[0, 0, ...])
    #
    # imgTools.imshow(prism_stack.sum(0).sum(0))
    #
    # visTools.scatter_xyz(p_des)  # show intersections
    # print(a_surf.facet_normals())

    # a_surf.plot_vertex_3d()

    # n = a_surf.facet_normals()

    # print(n.shape)
    # print(p.shape)
    # print(k.shape)
