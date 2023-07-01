import torch
import numpy as np
import cv2
import h5py


def preprocess_normalisation(img, type='complex'):

    if type == 'complex':
        img = img / abs(img).max()
    elif type == 'complex_phase':
        """ normalizes the magnitude of complex-valued image to range [0, 1] """
        abs_img = normalize(np.abs(img))
        ang_img = normalize(np.angle(img))
        img = abs_img * np.exp(1j * ang_img)
    elif type == '0_1':
        img = (img - img.min()) / (img.max() - img.min())
    else:
        raise NotImplementedError

    return img


def normalize(img):
  """ Normalize img in arbitrary range to [0, 1] """
  img -= np.min(img)
  img /= np.max(img)
  return img



if __name__ == '__main__':

    # single coil
    # npy This fil
    filename = f'../samples/single-coil/001.npy'
    img_a = np.load(filename).astype(np.complex64)
    img_a = np.abs(img_a)
    cv2.imwrite('../tmp/tmp_img_a.png', img_a * 255)

    # h5
    filename = '/media/ssd/data_temp/fastMRI/knee/d.2.0.complex.sc/val/PD/h5/file1000033_009.h5'
    with h5py.File(filename, 'r') as file:
        img_b = file['image_complex'][()]

    # img_b = preprocess_normalisation(img_b, type='complex')
    img_b = preprocess_normalisation(img_b, type='complex_phase')  # using this!

    img_b = np.abs(img_b)
    cv2.imwrite('../tmp/tmp_img_b.png', img_b * 255)

    err = np.abs(img_a - img_b)
    print(err.max())






    # # multi coil
    # filename = f'../samples/multi-coil/001.npy'
    # img = np.load(filename).astype(np.complex64)
    # # rss on dim 0
    # img = np.sqrt(np.sum(np.abs(img) ** 2, axis=0))
    # img = np.abs(img)







