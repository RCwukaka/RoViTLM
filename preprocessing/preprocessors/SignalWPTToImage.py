import numpy as np

def data_WPT1(waves):
    images = []
    for index, all_set in enumerate(waves):
        image = []
        # for i in range(0, 16, 4):
        #     for j in range(0, 256, 16):
        for i in range(0, 64, 8):
            for j in range(0, 64, 8):
                mind = []
                mind.extend(all_set[i][j:j + 8])
                mind.extend(all_set[i + 1][j:j + 8])
                mind.extend(all_set[i + 2][j:j + 8])
                mind.extend(all_set[i + 3][j:j + 8])
                mind.extend(all_set[i + 4][j:j + 8])
                mind.extend(all_set[i + 5][j:j + 8])
                mind.extend(all_set[i + 6][j:j + 8])
                mind.extend(all_set[i + 7][j:j + 8])
                image.append(mind)
        images.append(np.array(image))
    return images

def data_WPT(waves):
    images = []
    for index, all_set in enumerate(waves):
        image = []
        for j in range(0, 256, 16):
            mind = []
            mind.extend(all_set[0][j:j + 16])
            mind.extend(all_set[1][j:j + 16])
            mind.extend(all_set[5][j:j + 16])
            mind.extend(all_set[6][j:j + 16])
            image.append(mind)

        for j in range(0, 256, 16):
            mind = []
            mind.extend(all_set[2][j:j + 16])
            mind.extend(all_set[4][j:j + 16])
            mind.extend(all_set[7][j:j + 16])
            mind.extend(all_set[12][j:j + 16])
            image.append(mind)

        for j in range(0, 256, 16):
            mind = []
            mind.extend(all_set[3][j:j + 16])
            mind.extend(all_set[8][j:j + 16])
            mind.extend(all_set[11][j:j + 16])
            mind.extend(all_set[13][j:j + 16])
            image.append(mind)

        for j in range(0, 256, 16):
            mind = []
            mind.extend(all_set[9][j:j + 16])
            mind.extend(all_set[10][j:j + 16])
            mind.extend(all_set[14][j:j + 16])
            mind.extend(all_set[15][j:j + 16])
            image.append(mind)
        images.append(np.array(image))
    return images