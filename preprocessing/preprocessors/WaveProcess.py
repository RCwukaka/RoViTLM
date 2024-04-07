import pywt


def waveProcess(all_sets, level=4):
    x = []
    for index, all_set in enumerate(all_sets):
        x_in = [all_set[:, 0]]
        for _ in range(level):
            x_mid = []
            while len(x_in) != 0:
                x1 = x_in.pop()
                x_ca, x_cd = pywt.wavedec(x1, "db5", level=1)
                x_mid.append(x_ca)
                x_mid.append(x_cd)
            x_in = x_mid.copy()
        x.append(x_in)
    return x
