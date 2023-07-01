'''
# -----------------------------------------
Data Loader
FastMRI d.2.0.Complex.SC
by Jiahao Huang (j.huang21@imperial.ac.uk)
# -----------------------------------------
'''

import os
import random
import h5py
import numpy as np
import torch.utils.data as data
import utils_image as util
from select_mask import define_Mask
import torch
import numpy as np
from scipy.fftpack import *


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        # print(f'create {path}')
    else:
        pass
        # print(f'{path} already exists.')


def read_h5(data_path):
    dict = {}
    with h5py.File(data_path, 'r') as file:
        dict['image_complex'] = file['image_complex'][()]
        dict['data_name'] = file['image_complex'].attrs['data_name']
        dict['slice_idx'] = file['image_complex'].attrs['slice_idx']
        dict['image_rss'] = file['image_rss'][()]
    return dict


def preprocess_normalisation(img, type='complex'):

    if type == 'complex_mag':
        img = img / np.abs(img).max()
    elif type == 'complex':
        """ normalizes the magnitude of complex-valued image to range [0, 1] """
        abs_img = normalize(np.abs(img))
        ang_img = normalize(np.angle(img))
        img = abs_img * np.exp(1j * ang_img)
    elif type == '0_1':
        img = normalize(img)
    else:
        raise NotImplementedError

    return img


def normalize(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= np.min(img)
  img /= np.max(img)
  return img


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return np.sqrt((data ** 2).sum(dim))


def undersample_kspace(x, mask):

    fft = fftn(x, axes=(-2, -1))
    fft = fftshift(fft, axes=(-2, -1))

    fft = fft * mask

    fft = ifftshift(fft, axes=(-2, -1))
    x = ifftn(fft, axes=(-2, -1))

    return x


def generate_gaussian_noise(x, noise_level, noise_var):
    spower = np.sum(x ** 2) / x.size
    npower = noise_level / (1 - noise_level) * spower
    noise = np.random.normal(0, noise_var ** 0.5, x.shape) * np.sqrt(npower)
    return noise


class DatasetFastMRI(data.Dataset):

    def __init__(self, opt):
        super(DatasetFastMRI, self).__init__()
        print('Get L/H for image-to-image mapping. Both "paths_L" and "paths_H" are needed.')
        self.opt = opt
        self.patch_size = self.opt['H_size']
        self.complex_type = self.opt['complex_type']

        # get data path
        self.paths_raw = util.get_image_paths(opt['dataroot_H'])
        assert self.paths_raw, 'Error: Raw path is empty.'

        self.paths_H = []
        for path in self.paths_raw:
            if 'file' in path:
                self.paths_H.append(path)
            else:
                raise ValueError('Error: Unknown filename is in raw path')

        self.data_dict = {}

        # get mask
        if 'fMRI' in self.opt['mask']:
            mask_1d = define_Mask(self.opt)
            mask_1d = mask_1d[:, np.newaxis]
            mask = np.repeat(mask_1d, 320, axis=1).transpose((1, 0))
            self.mask = mask  # (H, W)
        else:
            self.mask = define_Mask(self.opt)  # (H, W)

        # # save mask
        # import cv2
        # mask_save_path = os.path.join('tmp', '{}.png'.format(self.opt['mask']))
        # cv2.imwrite(mask_save_path, self.mask * 255)
        # print('mask saved in {}'.format(mask_save_path))

    def __getitem__(self, index):

        mask = self.mask  # H, W, 1

        # get gt image
        H_path = self.paths_H[index]

        img_dict = read_h5(H_path)

        img_H_RSS = img_dict['image_rss']
        img_H_SC = img_dict['image_complex']

        img_H_RSS = preprocess_normalisation(img_H_RSS, type='0_1')
        img_H_SC = preprocess_normalisation(img_H_SC, type='complex')
        # img_H_SC = preprocess_normalisation(img_H_SC, type='complex_mag')

        # get zf image
        img_L_SC = undersample_kspace(img_H_SC, mask)
        img_L_ABS = abs(img_L_SC)
        img_H_ABS = abs(img_H_SC)

        # expand dim
        img_H_SC = img_H_SC[:, :, np.newaxis]  # H, W, 1
        img_L_SC = img_L_SC[:, :, np.newaxis]  # H, W, 1
        img_H_RSS = img_H_RSS[:, :, np.newaxis]  # H, W, 1
        img_L_ABS = img_L_ABS[:, :, np.newaxis]  # H, W, 1
        img_H_ABS = img_H_ABS[:, :, np.newaxis]  # H, W, 1

        # Complex --> 2CH
        img_H_SC = np.concatenate((np.real(img_H_SC), np.imag(img_H_SC)), axis=-1)  # H, W, 2
        img_L_SC = np.concatenate((np.real(img_L_SC), np.imag(img_L_SC)), axis=-1)  # H, W, 2

        # get image information
        data_name =img_dict['data_name']
        slice_idx = img_dict['slice_idx']
        img_info = '{}_{:03d}'.format(data_name, slice_idx)

        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        img_L_SC = torch.from_numpy(np.ascontiguousarray(img_L_SC)).permute(2, 0, 1).to(torch.float32)
        img_H_SC = torch.from_numpy(np.ascontiguousarray(img_H_SC)).permute(2, 0, 1).to(torch.float32)
        img_H_RSS = torch.from_numpy(np.ascontiguousarray(img_H_RSS)).permute(2, 0, 1).to(torch.float32)
        img_L_ABS = torch.from_numpy(np.ascontiguousarray(img_L_ABS)).permute(2, 0, 1).to(torch.float32)
        img_H_ABS = torch.from_numpy(np.ascontiguousarray(img_H_ABS)).permute(2, 0, 1).to(torch.float32)

        return {'L_SC': img_L_SC,  # (2, H, W)
                'L_ABS': img_L_ABS,  # (1, H, W)
                'H_SC': img_H_SC,  # (2, H, W)
                'H_RSS': img_H_RSS,  # (1, H, W)
                'H_ABS': img_H_ABS,  # (1, H, W)
                'H_path': H_path,
                'mask': mask,  # (H, W)
                'img_info': img_info}

    def __len__(self):
        return len(self.paths_H)






