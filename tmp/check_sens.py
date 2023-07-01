import numpy as np
import matplotlib.pyplot as plt

sens = np.load('/home/jh/score-MRI/results/multi-coil/hybrid/scoreMRI_fastmri.d.2.0.complex.mc_val_mini_fMRI_Ran_AF8_CF0.04_PE320_w_phase_normalisation/sens/sens_file1000031_001.npy')
C, H, W = sens.shape

for i in range(C):
    sen = sens[i, :, :]
    sen = np.abs(sen)
    plt.imsave('./tmp/sens_file1000031_001_{}.png'.format(i), sen, cmap='gray')
