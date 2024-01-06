import math
from collections import Counter

import numpy as np
from scipy import stats
from scipy.signal import hilbert


def getFeatureAll(x_waves):
    feature = []
    for index in range(len(x_waves)):
        x_o_wave_feature = getOriginalFeatureDim(x_waves[index])
        x_e_wave_feature = getEnvelopeFeatureDim(x_waves[index])
        mid = np.vstack((x_o_wave_feature, x_e_wave_feature))
        feature.append(mid)
    return np.array(feature)


def getOriginalFeatureDim(x_wave):
    mus = np.mean(x_wave, axis=1)
    stds = np.std(x_wave, axis=1)
    var = np.var(x_wave, axis=1)
    skews = stats.skew(x_wave, axis=1)
    maxmins = np.max(x_wave, axis=1) - np.min(x_wave, axis=1)
    kurtosis = stats.kurtosis(x_wave, axis=1)
    entropy = Entropy(x_wave)
    energy = np.sum(x_wave, axis=1)
    feature = np.vstack((mus, stds, var, maxmins, skews, entropy, kurtosis, energy))
    return feature


def Entropy(dataList):
    entropys = []
    for data in dataList:
        counts = len(data)  # 总数量
        counter = Counter(data)  # 每个变量出现的次数
        prob = {i[0]: i[1] / counts for i in counter.items()}  # 计算每个变量的 p*log(p)
        H = - sum([i[1] * math.log2(i[1]) for i in prob.items()])  # 计算熵
        entropys.append(H)
    return entropys


def getEnvelopeFeatureDim(x_wave):
    x_wave = np.abs(hilbert(x_wave))
    return getOriginalFeatureDim(x_wave)