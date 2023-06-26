import os
import numpy as np
import skimage.metrics

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def calculate_psnr_single(img1, img2, border=0):
    # img1 =  np.clip(img1.squeeze(), -1, 1)
    # img2 = np.clip(img2.squeeze(), -1, 1)
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    # img1 = img1.astype(np.float64)
    # img2 = img2.astype(np.float64)
    # gt recon
    return skimage.metrics.peak_signal_noise_ratio(img1, img2)

def calculate_ssim_single(img1, img2, border=0):
    # img1 = img1.squeeze()
    # img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    return skimage.metrics.structural_similarity(img1, img2)



if __name__ == '__main__':

    # '/home/jh/score-MRI/results/single-coil/recon/001.npy'
    # '/home/jh/score-MRI/results/single-coil/label/001.npy'
    # '/home/jh/score-MRI/results/single-coil/input/001.npy'

    # load recon label input
    recon = np.load('/home/jh/score-MRI/results/single-coil/recon/001.npy')
    label = np.load('/home/jh/score-MRI/results/single-coil/label/001.npy')
    input = np.load('/home/jh/score-MRI/results/single-coil/input/001.npy')

    # abs
    recon = np.abs(recon)
    label = np.abs(label)
    input = np.abs(input)

    # calculate psnr
    psnr_recon = calculate_psnr_single(recon, label)
    psnr_input = calculate_psnr_single(input, label)
    print('psnr_recon: ', psnr_recon)
    print('psnr_input: ', psnr_input)

    # calculate ssim
    ssim_recon = calculate_ssim_single(recon, label)
    ssim_input = calculate_ssim_single(input, label)
    print('ssim_recon: ', ssim_recon)
    print('ssim_input: ', ssim_input)
