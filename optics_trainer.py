"""
optical model
"""

import argparse


def get_parser():
    # mechanical structure
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dev_phase', dest='dev_phase', default='Train')  # can be 'Export' instead

    parser.add_argument('--aper_r', dest='aper_r', default=12.7,
                        help='circular aperture radius of the freeform lens')
    # parser.add_argument('--num_want_mizi', dest='num_want_mizi', default=100)
    parser.add_argument('--num_want_mizi', dest='num_want_mizi', default=100)
    """ all possible actual num_mizi in for aper_r=12.7   :
    [   1    4    9   16   21   32   45   52   69   88  101  120  145  164
      185  216  241  268  293  332  365  392  437  468  509  556  593  648
      681  732  785  824  885  936  989 1052 1109 1168 1225 1288 1353 1428
     1489 1560 1625 1696 1781 1844 1933 2008 2085 2180 2241 2340 2425 2512
     2609 2700 2793 2876 2973 3080 3173 3276 3381 3480 3585 3704 3801 3908
     4017 4144 4257 4360 4493 4612 4717 4856 4981 5088 5225 5356 5497 5616
     5753 5900 6005 6172 6309 6440]
    """
    parser.add_argument('--allow_xy_move', dest='allow_xy_move', default=False)
    parser.add_argument('--target_plane_z', dest='target_plane_z', default=300,
                        help='z of target plane')
    parser.add_argument('--target_plane_half_width', dest='target_plane_half_width', default=12.7)
    parser.add_argument('--basin_xmin', dest='basin_xmin', default=-12.7)
    parser.add_argument('--basin_xmax', dest='basin_xmax', default=12.7)
    parser.add_argument('--basin_ymin', dest='basin_ymin', default=-12.7)
    parser.add_argument('--basin_ymax', dest='basin_ymax', default=12.7)

    # optimization needed
    parser.add_argument('--optimizer', dest='optimizer',
                        default='sgd',
                        # default='adam', #hard to tune
                        help='optimizer type str')

    # data related
    parser.add_argument('--default_root_dir', dest='default_root_dir', default='lightning_logs', )
    parser.add_argument('--target_img_path', dest='target_img_path',
                        default=r'.\datasets\target_pictures\house_original_4.1.05.tiff')
    # default=r'.\datasets\target_pictures\pepper.bmp')
    # default=r'.\\datasets\target_pictures\pepper.bmp')
    parser.add_argument('--target_img_res', dest='target_img_res', default=32)
    parser.add_argument('--resize_times', dest='resize_times', default=4)
    # steepness = 2.
    steepness = 3.48
    parser.add_argument('--steepness', dest='steepness',
                        default=steepness)
    parser.add_argument('--field_weight_factor', dest='field_weight_factor', default=1.)
    # parser.add_argument('--subloss_weight', dest='subloss_weight', default=1.,
    #                     help='sub loss weight')  # relative to that of 'mse' which is always 1

    # training process control
    parser.add_argument('--auto_tune_lr', dest='auto_tune_lr', default=False)
    parser.add_argument('--train_duration', dest='train_duration', default='00:00:30:00')
    parser.add_argument('--gpus', dest='gpus', default=0)  # 0,1 swaps in task monitor of 51
    parser.add_argument('--version', dest='version', default=0)  # deb 666
    parser.add_argument('--model_name', dest='model_name', default='Parallel_1ffSurf_Picture')
    parser.add_argument('--max_epochs', dest='max_epochs', default=100)
    parser.add_argument('--batch_size', dest='batch_size', default=1)  # this is 1, don't change it
    # recommend_lr = 2e-3 / (steepness / 2.)  #
    # recommend_lr = 2e-3 / (steepness / 2.)  #
    # ↑ TUNED steepness=2 WITH lr=2e-3, this line keeps the theoretical max gradient,
    # SEE doc str of sigmoid_step() to understand, you may derive by hand
    # parser.add_argument('--lr', dest='lr', default=recommend_lr)
    parser.add_argument('--lr', dest='lr', default=4e-3)
    # parser.add_argument('--lr_decay_factor', dest='lr_decay_factor', default=0.90)
    # parser.add_argument('--lr_decay_period', dest='lr_decay_period',
    #                     default=500)
    # parser.add_argument('--image_log_period', dest='image_log_period',
    #                     default=20)
    parser.add_argument('--validation_step_interval', dest='validation_step_interval', default=100)
    parser.add_argument('--num_workers', dest='num_workers', default=0)

    # ---------------------

    return parser


if __name__ == '__main__':

    hps = get_parser().parse_args()
    hps.gpus = [int(float(hps.gpus))]
    hps.target_img_res = int(hps.target_img_res)
    hps.resize_times = int(hps.resize_times)
    # hps.image_log_period = int(hps.image_log_period)
    hps.num_want_mizi = int(hps.num_want_mizi)
    hps.num_workers = int(hps.num_workers)
    print('--------- \n Hyper params of this running:==== \n', hps, '\n---------')


    # """ <traing code
    import os
    import timer
    import optics
    import torch
    from util import mat_easy_save
    import pytorch_lightning as pl
    from data.illum_sys_dummy_dataset import make_train_data_loader, make_dummy0_data_loader
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import EarlyStopping

    illum_sys = optics.Parallel_1ffSurf_Picture(hps)

    logger = TensorBoardLogger(hps.default_root_dir, name=hps.model_name, version=hps.version)
    hps.default_root_dir = logger.log_dir
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
        monitor='train_loss',  # mind that the str must be same as self.log(str, val) in LightningModule
        # monitor=None,
        filepath=os.path.join(logger.log_dir, 'checkpoints',
                              '{epoch}-hello-{train_loss:.4f}'),
        save_top_k=1,
        period=10,  # should be 1 when using dataset_length !=1 , otherwise 200,
        mode='min',
    )
    timer_callback = timer.Timer(duration=hps.train_duration)
    early_stopping_callback = EarlyStopping(monitor='train_loss', min_delta=1e-5, patience=20, verbose=True, mode='min')
    train_dataloader = make_dummy0_data_loader(hps, 100)
    validation_dataloader = make_dummy0_data_loader(hps, 1)
    if hps.auto_tune_lr:
        lr_tuner = pl.Trainer.from_argparse_args(hps, auto_lr_find=True, )
        lr_tuner.tune(model=illum_sys, train_dataloader=train_dataloader)
        # mind that multi-run of code with auto tune can cause mixture access of lr_find_temp.ckpt it use and hence troubles
        print('AUTO TUNED ', illum_sys.lr, 'is the lr in use INSTEAD OF ', hps.lr)

        # illum_sys.lr *= 10
        # print('AUTO TUNED lrx10', illum_sys.lr, 'is the lr in use INSTEAD OF ', hps.lr)
        # illum_sys.lr_decay_period = 1000

    trainer = pl.Trainer.from_argparse_args(
        hps,
        logger=logger,
        # callbacks=[logmanager_callback],
        callbacks=[
            timer_callback,
            early_stopping_callback
        ],
        checkpoint_callback=checkpoint_callback,
        # sync_batchnorm=True,
        benchmark=True,
        gradient_clip_val=0.5,
        val_check_interval=hps.validation_step_interval,
        log_every_n_steps=100
        # resume_from_checkpoint=get_ckpt_path(), # uncomment this line to resume training from certain ckpt
    )
    if hps.dev_phase == "Train":
        trainer.fit(illum_sys, train_dataloader=train_dataloader,
                    val_dataloaders=validation_dataloader,
                    )
    elif hps.dev_phase == "Export":
        ckpt_folder_path = os.path.join(trainer.logger.log_dir, 'checkpoints')
        ckpt_file_name_list = os.listdir(
            ckpt_folder_path)
        for name in ckpt_file_name_list:
            if 'ckpt' in name:
                ckpt_file_name = name
                ckpt_path = os.path.join(ckpt_folder_path, ckpt_file_name)

        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        state_dict = ckpt['state_dict']
        illum_sys.load_state_dict(state_dict)
        mizi_facet_group_vertex = illum_sys.ff_surf.refresh_vertex_arr()
        # ↑ torch.Size([3<facet vertices>, 8<facets around each mizi_center>, num_mizi, 3<xyz>])
        facet_group_vertex_save_path = os.path.join(ckpt_folder_path,
                                                    hps.version + 'export_surface' + ckpt_file_name[:-4] + 'mat')
        mizi_facet_group_vertex_arr = mat_easy_save(mizi_facet_group_vertex, facet_group_vertex_save_path)
        print('SAVED:  ', facet_group_vertex_save_path)

    # """  # >traing code
