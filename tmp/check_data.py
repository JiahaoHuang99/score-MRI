import torch
import numpy as np
import cv2



if __name__ == '__main__':

    filename = f'../samples/single-coil/001.npy'

    img =np.load(filename).astype(np.complex64)

    cv2.imwrite('../tmp/tmp_img.png', abs(img) * 255)

    print(img.shape)
