import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_RI_coil_SENSE)
from models import ncsnpp
import time
from utils import fft2_m, ifft2_m, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint, \
    normalize_complex, root_sum_of_squares, lambda_schedule_const, lambda_schedule_linear
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
    N = 500  # default: 500
    m = 1  # default: 1

    print('initaializing...')
    configs = importlib.import_module("configs.ve.fastmri_knee_320_ncsnpp_continuous")
    config = configs.get_config()
    img_size = config.data.image_size
    batch_size = 1

    dataset_name = dataset_opt['dataset_name']
    mask_name = dataset_opt['mask']

    task_name = 'scoreMRI_{}_{}'.format(dataset_name, mask_name)

    schedule = 'linear'
    start_lamb = 1.0
    end_lamb = 0.2
    m_steps = 50

    if schedule == 'const':
        lamb_schedule = lambda_schedule_const(lamb=start_lamb)
    elif schedule == 'linear':
        lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)
    else:
        NotImplementedError("Given schedule {schedule} not implemented yet!")


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
    save_root = Path('./results/multi-coil/hybrid/{}'.format(task_name))
    save_root.mkdir(parents=True, exist_ok=True)

    for idx, test_data in enumerate(test_loader):

        if idx < start_slice:
            print('Slice Idx: {}'.format(idx))
            continue

        ###############################################
        # 2. Inference
        ###############################################

        L_RSS = test_data['L_RSS'].to(config.device)
        H_RSS = test_data['H_RSS'].to(config.device)

        L_MC = test_data['L_MC'].to(config.device)
        L_MC = L_MC[:, :, 0, :, :] + 1j * L_MC[:, :, 1, :, :]
        H_MC = test_data['H_MC'].to(config.device)
        H_MC = H_MC[:, :, 0, :, :] + 1j * H_MC[:, :, 1, :, :]
        img_info = test_data['img_info'][0]
        mask = test_data['mask'].expand(15, -1, -1).unsqueeze(0).to(config.device)

        img = H_MC.clone()
        under_img = L_MC.clone()
        kspace = fft2_m(img)
        under_kspace = kspace * mask

        mkdir(os.path.join(save_root, 'sens'))
        mps_dir = os.path.join(save_root, 'sens', 'sens_{}.npy'.format(img_info))
        # ESPiRiT
        if os.path.exists(mps_dir):
            print(' EspiritCalib Pass! Load sens from {}.'.format(mps_dir))
            mps = np.load(mps_dir)
        else:
            mps = mr.app.EspiritCalib(kspace.cpu().detach().squeeze().numpy()).run()
            np.save(mps_dir, mps)
        # mps = mr.app.EspiritCalib(kspace.cpu().detach().squeeze().numpy()).run()

        mps = torch.from_numpy(mps).view(1, 15, 320, 320).to(kspace.device)



        pc_fouriercs = get_pc_fouriercs_RI_coil_SENSE(sde,
                                                      predictor, corrector,
                                                      inverse_scaler,
                                                      snr=snr,
                                                      n_steps=m,
                                                      m_steps=50,
                                                      mask=mask,
                                                      sens=mps,
                                                      lamb_schedule=lamb_schedule,
                                                      probability_flow=probability_flow,
                                                      continuous=config.training.continuous,
                                                      denoise=True)

        print('Beginning inference')
        tic = time.time()
        x = pc_fouriercs(score_model, scaler(under_img), y=under_kspace)
        toc = time.time() - tic
        print('Time took for recon: {} secs.'.format(toc))

        ###############################################
        # 3. Saving recon
        ###############################################
        L_RSS = L_RSS.squeeze().cpu().detach().numpy()
        H_RSS = H_RSS.squeeze().cpu().detach().numpy()
        mask_sv = mask[0, 0, :, :].squeeze().cpu().detach().numpy()

        x_abs = torch.abs(x)
        x_abs_rss = root_sum_of_squares(x_abs, dim=1)
        x_abs_rss = x_abs_rss.squeeze().cpu().detach().numpy()

        print('GT: Max {} Min {}'.format(np.max(H_RSS), np.min(H_RSS)))
        print('ZF: Max {} Min {}'.format(np.max(L_RSS), np.min(L_RSS)))
        print('Recon: Max {} Min {}'.format(np.max(x_abs_rss), np.min(x_abs_rss)))

        mkdir(os.path.join(save_root, 'png', 'GT'))
        mkdir(os.path.join(save_root, 'png', 'ZF'))
        mkdir(os.path.join(save_root, 'png', 'Recon'))
        mkdir(os.path.join(save_root, 'h5'))

        plt.imsave(os.path.join(save_root, 'png', 'GT', 'GT_{}.png'.format(img_info)), H_RSS, cmap='gray')
        plt.imsave(os.path.join(save_root, 'png', 'ZF', 'ZF_{}.png'.format(img_info)), L_RSS, cmap='gray')
        plt.imsave(os.path.join(save_root, 'png', 'Recon', 'Recon_{}.png'.format(img_info)), x_abs_rss, cmap='gray')
        plt.imsave(os.path.join(save_root, 'png', 'mask.png'), mask_sv, cmap='gray')

        with h5py.File(os.path.join(save_root, 'h5', '{}.h5'.format(img_info)), "w") as file:
            file['gt'] = H_RSS
            file['recon'] = x_abs_rss
            file['zf'] = L_RSS
            file.attrs['img_info'] = img_info



if __name__ == "__main__":

    print(torch.cuda.is_available())

    start_slice = 0

    dataset_opt = {
        "name": "test_dataset",
        "dataset_name": "fastmri.d.2.0.complex.mc_val_mini",
        "dataset_type": "fastmri.d.2.0.complex.mc",
        "dataroot_H": "/media/ssd/data_temp/fastMRI/knee/d.2.0.complex.mc/val_mini/PD",
        "mask": "fMRI_Ran_AF8_CF0.04_PE320",
        "H_size": 320,
        "complex_type": "1ch",
    }

    main(dataset_opt, start_slice)