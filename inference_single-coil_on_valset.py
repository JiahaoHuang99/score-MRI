import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI)
from models import ncsnpp
import time
from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse
import sigpy.mri as mr


from data.select_dataset import define_Dataset
from torch.utils.data import DataLoader
import cv2
import h5py


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


def main(dataset_opt, start_slice):
    ###############################################
    # 1. Configurations
    ###############################################

    # args
    N = 2000  # default: 500
    m = 1  # default: 1

    print('initaializing...')
    configs = importlib.import_module("configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    dataset_name = dataset_opt['dataset_name']
    mask_name = dataset_opt['mask']

    task_name = 'scoreMRI_{}_{}'.format(dataset_name, mask_name)

    test_set = define_Dataset(dataset_opt)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1,
                             drop_last=False,
                             pin_memory=True)


    ckpt_filename = "./weights/checkpoint_95.pth"
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    config.training.batch_size = batch_size
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16

    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
    ema.copy_to(score_model.parameters())

    # Specify save directory for saving generated samples
    save_root = Path('./results/single-coil/{}'.format(task_name))
    save_root.mkdir(parents=True, exist_ok=True)

    is_skip = True
    for idx, test_data in enumerate(test_loader):

        # load from break point
        img_info = test_data['img_info'][0]
        # start the loop when img_info == file1000052_009
        # before that, skip the loop
        # after that, keep the loop
        if img_info == start_slice:
            is_skip = False

        if is_skip:
            print('SKIP: Slice Idx: {}; Img Info: {}'.format(idx, img_info))
            continue
        else:
            print('PROCESS: Slice Idx: {}; Img Info: {}'.format(idx, img_info))

        ###############################################
        # 2. Inference
        ###############################################

        L_ABS = test_data['L_ABS'].to(config.device)
        H_ABS = test_data['H_ABS'].to(config.device)

        L_SC = test_data['L_SC'].to(config.device)
        L_SC = L_SC[:, 0, :, :] + 1j * L_SC[:, 1, :, :]
        H_SC = test_data['H_SC'].to(config.device)
        H_SC = H_SC[:, 0, :, :] + 1j * H_SC[:, 1, :, :]
        img_info = test_data['img_info'][0]
        mask = test_data['mask'].unsqueeze(0).to(config.device)

        img = H_SC.clone()
        under_img = L_SC.clone()
        kspace = fft2(img)
        under_kspace = kspace * mask

        pc_fouriercs = get_pc_fouriercs_RI(sde,
                                           predictor, corrector,
                                           inverse_scaler,
                                           snr=snr,
                                           n_steps=m,
                                           probability_flow=probability_flow,
                                           continuous=config.training.continuous,
                                           denoise=True)

        print('Beginning inference')
        tic = time.time()
        x = pc_fouriercs(score_model, scaler(under_img), mask, Fy=under_kspace)
        toc = time.time() - tic
        print('Time took for recon: {} secs.'.format(toc))

        ###############################################
        # 3. Saving recon
        ###############################################
        H_ABS = H_ABS.squeeze().cpu().detach().numpy()
        L_ABS = L_ABS.squeeze().cpu().detach().numpy()
        mask_sv = mask[0, 0, :, :].squeeze().cpu().detach().numpy()

        x_abs = torch.abs(x)
        x_abs = x_abs.squeeze().cpu().detach().numpy()

        print('GT: Max {} Min {}'.format(np.max(H_ABS), np.min(H_ABS)))
        print('ZF: Max {} Min {}'.format(np.max(L_ABS), np.min(L_ABS)))
        print('Recon: Max {} Min {}'.format(np.max(x_abs), np.min(x_abs)))

        mkdir(os.path.join(save_root, 'png', 'GT'))
        mkdir(os.path.join(save_root, 'png', 'ZF'))
        mkdir(os.path.join(save_root, 'png', 'Recon'))
        mkdir(os.path.join(save_root, 'h5'))

        plt.imsave(os.path.join(save_root, 'png', 'GT', 'GT_{}.png'.format(img_info)), H_ABS, cmap='gray')
        plt.imsave(os.path.join(save_root, 'png', 'ZF', 'ZF_{}.png'.format(img_info)), L_ABS, cmap='gray')
        plt.imsave(os.path.join(save_root, 'png', 'Recon', 'Recon_{}.png'.format(img_info)), x_abs, cmap='gray')
        plt.imsave(os.path.join(save_root, 'png', 'mask.png'), mask_sv, cmap='gray')

        with h5py.File(os.path.join(save_root, 'h5', '{}.h5'.format(img_info)), "w") as file:
            file['gt'] = H_ABS
            file['recon'] = x_abs
            file['zf'] = L_ABS
            file.attrs['img_info'] = img_info



if __name__ == "__main__":

    print(torch.cuda.is_available())

    start_slice = 'file1000264_007'

    dataset_opt = {
        "name": "test_dataset",
        "dataset_name": "fastmri.d.2.1.complex.sc_val",
        "dataset_type": "fastmri.d.2.1.complex.sc",
        "dataroot_H": "/media/ssd/data_temp/fastMRI/knee/d.2.0.complex.sc/val/PD",
        "mask": "fMRI_Ran_AF8_CF0.04_PE320",
        "H_size": 320,
        "complex_type": "1ch",
    }

    main(dataset_opt, start_slice)