"""
This script facilitates the conversion of images from the .npy format with original
non-conventional intensities to the same .npy format, but with standardized intensities
within the conventional [0, 255] range.
"""
import glob
from typing import List

import numpy as np
import argparse
import os
from segmentation import utils


def converter(image_path, out_pth): 
    image = np.load(image_path)                                                                                                                                                                                                                                                                                                
    image = (image - image.min()) / (image.max() - image.min())
    image = (image * 255).astype(np.uint8)
    np.save(out_pth, image)

def preprocessing(collection: List, out_dir: str, out_ext: str):
    assert out_ext.startswith('.')
    os.makedirs(out_dir, exist_ok=True)
    not_processed = []
    for image_path in collection:
        print('processed: ', image_path)
        os.makedirs(os.path.join(out_dir, *os.path.dirname(image_path).split('/')[-1:]), exist_ok=True)
        try: 
            converter(image_path, os.path.join(out_dir, *image_path.split('/')[-2:]))
        except: 
            print(image_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation Configs Params.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        help='The string path of the config file.')
    args = parser.parse_args()
    configs = utils.config_loader(args.config_path)

    dirs = sorted(glob.glob(configs['source_dir'], recursive=True))
    print(len(dirs))

    for item in dirs: 
        if os.path.isdir(item): 
            collection = sorted(glob.glob(os.path.join(item, '*.npy'), recursive=True))
            if len(collection) > 0: 
                preprocessing(
                    collection=collection,
                    out_dir = configs['destination_dir'],
                    out_ext='.npy'
                )