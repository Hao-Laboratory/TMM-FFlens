"""
this file defines the dataloaders for training of illum optics sys

"""

from torch.utils.data import Dataset, DataLoader
import imgTools
import filePaths


class DummyZeroDataset(Dataset):
    def __init__(self, length):
        super(DummyZeroDataset, self).__init__()
        self.length = length

    def __len__(self):
        # mind that model.global_step = self.__len__()/batch_size*(printed_Epoch+1)
        return self.length  # we have only one data point, the length is one

    def __getitem__(self, idx):
        # return 0, self.target_field
        return 0  # use no dummy input


class FixedTargetField(Dataset):

    def __init__(self, img_file_path, resize_shape, with_circular_aperture=True, pointy_field_x3=False,
                 nearsest_resize_time=3):
        """
        :param img_file_path file path of the one used as target illum field
        :param resize_shape tuple(m ,n), this will determine the out put image resolution(shape)
        :param with_circular_aperture defualt true, use circle crop
        :param pointy_field_x3 if True return pointy structured illum field as y-true
        """
        super().__init__()
        self.sampling_shape = (resize_shape, resize_shape)

        # read as gray resized and standardized
        self.target_field, self.input_gray_image = imgTools.read_as_gray_resized_and_standardized(
            img_file_path,
            self.sampling_shape
        )  # already standardized pix to range 0~1

        # inscrib circle crop
        if with_circular_aperture:
            self.target_field = imgTools.inscrib_circle_cropped(self.target_field)

        # make target into pointy structured light
        if pointy_field_x3:
            self.target_field = imgTools.pointy_pixel_x3(self.target_field)

        if nearsest_resize_time is not None:
            self.target_field = imgTools.nearsest_resize_times(self.target_field, nearsest_resize_time)

    def __len__(self):
        # mind that model.global_step = self.__len__()/batch_size*(printed_Epoch+1)
        return 1  # we have only one data point, the length is one

    def __getitem__(self, idx):
        # return 0, self.target_field
        return self.target_field  # use no dummy input


def make_train_data_loader(hps):
    dataset = FixedTargetField(hps.target_img_path, hps.target_img_res, nearsest_resize_time=hps.resize_times)
    train_dataloader = DataLoader(dataset,
                                  batch_size=hps.batch_size,
                                  # sampler=sampler,
                                  num_workers=hps.num_workers,
                                  # shuffle=False,
                                  pin_memory=True)
    return train_dataloader


def make_dummy0_data_loader(hps, length):
    dataset = DummyZeroDataset(length)
    train_dataloader = DataLoader(dataset,
                                  batch_size=hps.batch_size,
                                  # sampler=sampler,
                                  num_workers=hps.num_workers,
                                  # shuffle=False,
                                  pin_memory=True)
    return train_dataloader


def y_illum_field_extract(hps):
    dataset = FixedTargetField(hps.target_img_path, hps.target_img_res, nearsest_resize_time=hps.resize_times)
    y_illum_field = dataset.__getitem__(0)
    return y_illum_field


def y_illum_field_default():
    # mind the param values here, are they the same as those in optics_trainer.py
    path = filePaths.defualt_target_path
    dataset = FixedTargetField(path,
                               32,
                               nearsest_resize_time=4)
    y_illum_field = dataset.__getitem__(0)
    return y_illum_field


if __name__ == '__main__':
    # mind that using python console to debug this may run into troubles when import ...trainer.py for hps defination
    # these files can cause loopy import,
    pass
