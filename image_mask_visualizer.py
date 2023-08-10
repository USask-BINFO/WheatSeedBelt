"""
Visualizing overlapped versions of pairs of images and their masks.
"""
import glob

import matplotlib.pyplot as plt
import numpy as np
from skimage import io

if __name__ == '__main__':
    image_paths = sorted(glob.glob('data/batch2/transparent/images/*.tif'))
    mask_paths = sorted(glob.glob('data/batch2/transparent/masks/*.nrrd'))
    assert len(image_paths) == len(mask_paths)

    for img_pth, msk_pth in zip(image_paths, mask_paths):
        image = io.imread(img_pth).astype(np.uint8)
        mask = io.imread(msk_pth).astype(np.uint8).squeeze()

        plt.imshow(image)
        plt.imshow(mask, cmap='jet', alpha=0.3)
        plt.show()