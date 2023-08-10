"""
Extract the color intensity for each contoured image, separately.
The results would be saved inside a json file, group-wise and in a daily bases.
"""
import argparse
import os

import numpy as np
import pandas as pd
from skimage import color
from skimage import io

from segmentation import RoIExtraction, FeatureExtraction
from segmentation import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation Configs Params.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        help='The string path of the config file.')
    args = parser.parse_args()
    configs = utils.config_loader(args.config_path)
    
    # RoI Extractor.
    roi_extractor = RoIExtraction(
        metadata_path=configs['metadata_path'],
        channel_number=configs['channel_number'],
        include_mirror_obj=configs['include_mirror_obj']
    )
    print('Extracting Regions of Interests...')
    os.makedirs(os.path.join(configs['out_dir'], 'RoIs', 'images'), exist_ok=True)
    os.makedirs(os.path.join(configs['out_dir'], 'RoIs', 'masks'), exist_ok=True)
    if configs['include_mirror_obj'] == True: 
        os.makedirs(os.path.join(configs['out_dir'], 'RoIs', 'mirrors'), exist_ok=True)

    rois_meta_df = None 
    for item in range(len(roi_extractor)): 
        sample = {}
        meta, roi_regions = roi_extractor[item]
        sample['PackageID'] = meta['PackageID']
        sample['ImageID'] = meta['ImageID']
        extnsion_len = len(os.path.basename(meta['Image']).split('.')[-1]) 
        sample['Image'] = os.path.join(
            configs['out_dir'], 
            'RoIs', 'images', 
            f"{os.path.basename(meta['Image'])[:-(extnsion_len+1)]}{configs['img_ext']}"
        )
        # The format of the `roi_regions`` list: `mirror object`, `image`, `mask`
        io.imsave(sample['Image'], 
                  roi_regions[1].astype(np.uint8), 
                  check_contrast=False
        )
        extnsion_len = len(os.path.basename(meta['Mask']).split('.')[-1]) 
        sample['Mask'] = os.path.join(
            configs['out_dir'], 
            'RoIs', 'masks',
            f"{os.path.basename(meta['Mask'])[:-(extnsion_len+1)]}{configs['msk_ext']}"
        )
        io.imsave(sample['Mask'], 
                  roi_regions[2].astype(np.uint8), 
                  check_contrast=False
        )
        if 'Label' in meta.index:
            sample['Label'] = meta['Label']
        if configs['include_mirror_obj'] == True:
            extnsion_len = len(os.path.basename(meta['Image']).split('.')[-1]) 
            sample['Mirror'] = os.path.join(
                configs['out_dir'], 
                'RoIs', 'mirrors', 
                f"{os.path.basename(meta['Image'])[:-(extnsion_len+1)]}{configs['mir_ext']}"
            )
            io.imsave(sample['Mirror'], 
                      roi_regions[0].astype(np.uint8), 
                      check_contrast=False
            )
        # Feature Extractor.
        feature_extractor = FeatureExtraction()
        roi_gray = (255 * color.rgb2gray(roi_regions[1])).astype(np.uint8)
        sample['Color'] = feature_extractor.histogram_extractor(
                                                    features=roi_gray).tolist()
        glcm_feat = feature_extractor.glcm(
                image=roi_gray,
                distances=[2, 5, 7, 11, 15],
                angles=[0, np.pi/2, 2*np.pi/3, np.pi, 3*np.pi/2],
                levels=256
        )
        sample['Glcm'] = glcm_feat.mean(axis=1).flatten().tolist()
        sample['Gccrop'] = feature_extractor.gccrop(features=glcm_feat,
                                                    prop='contrast').tolist()
        sample['Lbp'] = feature_extractor.lbp(image=roi_gray, p=8, r=1).tolist()
        if rois_meta_df is None: 
            rois_meta_df = pd.DataFrame([sample])
        else:
           rois_meta_df = rois_meta_df.append(sample, ignore_index=True)
    # Write down the dataframe. 
    rois_meta_df.reset_index(drop=True, inplace=True)
    rois_meta_df.to_csv(configs['out_metadata_path'], index=False)