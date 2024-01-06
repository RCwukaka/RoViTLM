import numpy as np

def data_WPT(waves):
    images = []
    for index, all_set in enumerate(waves):
        image = []
        for i in range(0, 16, 4):
            for j in range(0, 256, 16):
                mind = []
                mind.extend(waves[index][i][j:j + 16])
                mind.extend(waves[index][i + 1][j:j + 16])
                mind.extend(waves[index][i + 2][j:j + 16])
                mind.extend(waves[index][i + 3][j:j + 16])
                image.append(mind)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        images.append(image)
    return images