"""
Segment input images.
"""

import argparse
import os

import numpy as np
import pandas as pd
from skimage import color, filters
from skimage import io

from segmentation import Morphology
from segmentation import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation Configs Params.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        help='The string path of the config file.')
    args = parser.parse_args()
    configs = utils.config_loader(args.config_path)

    metadata = pd.read_csv(configs['metadata_path'])
    os.makedirs(configs['out_dir'], exist_ok=True)

    # Define a morphology object.
    morphology = Morphology(
        disk_radius=12,
        area_threshold=1024,
        min_size=2048,
        connectivity=2,
        threshold_method=filters.threshold_otsu,
        color_space_converter=color.rgb2lab,
        channel_number=2,
        num_kmean_clusters=None,
        object_mode='bright'
    )
    region_crop = utils.RegionCrop(
        height=(200, 1400), 
        width=(200, 2000)
    )
    composer = utils.Compose([
        morphology.apply_thresholding,
        morphology.apply_opening,
        morphology.apply_rm_small_objects,
        morphology.apply_rm_small_holes
    ])

    os.makedirs(os.path.join(configs['out_dir'], '../images'), exist_ok=True)
    os.makedirs(os.path.join(configs['out_dir'], 'masks'), exist_ok=True)

    segmented_paths = {
        'PackageID': [], 
        'ImageID': [],
        'Image': [],
        'Mask': [], 
        'Label': []
    }
    for i, row in metadata.iterrows():
        if row['Image'].endswith('.npy'): 
            image = np.load(row['Image'])
        else: 
            image = io.imread(row['Image'])
        image = region_crop(image)
        
        mask = composer(image)

        if configs['visualize'] == True:
            utils.masked_image_visualizer(image, mask)
        
        seg_img_pth = os.path.join(
                configs['out_dir'],
            '../images',
                f"{configs['name_prefix']}-{i:0>6}{configs['img_ext']}"
        )
        seg_msk_pth = os.path.join(
                configs['out_dir'], 
                'masks',
                f"{configs['name_prefix']}-{i:0>6}{configs['msk_ext']}"
        )
        
        io.imsave(seg_img_pth, image, check_contrast=False)
        io.imsave(seg_msk_pth, mask.astype(np.uint8), check_contrast=False)
        
        segmented_paths['PackageID'].append(row['PackageID'])
        segmented_paths['ImageID'].append(row['ImageID'])
        segmented_paths['Image'].append(seg_img_pth)
        segmented_paths['Mask'].append(seg_msk_pth)
        if 'Label' in row.keys(): 
            segmented_paths['Label'].append(row['Label'])
        
        print('Processed: ', row['Image'])
    if len(segmented_paths['Label']) == 0: 
        del segmented_paths['Label']
    df = pd.DataFrame(segmented_paths)
    df.to_csv(configs['segmented_metadata_path'], index=False)