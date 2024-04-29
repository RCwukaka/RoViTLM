import numpy as np
import pywt


def waveProcess(all_sets, level=4):
    x = []
    for index, all_set in enumerate(all_sets):
        x_in = [all_set[:, 0]]
        wp = pywt.WaveletPacket(data=x_in, wavelet="db1", mode='symmetric', maxlevel=level)
        wp_coeffs = [np.squeeze(n.data) for n in wp.get_level(level, "freq")]
        x.append(wp_coeffs)
    return x
