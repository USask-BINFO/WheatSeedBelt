""" Utility classes and function. """
import copy
import glob
import json
import os
import warnings
from typing import Union, List, Tuple, Callable

import SimpleITK as sitk
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import yaml
from skimage import feature as feat
from skimage import io


class Compose:
    """Apply a list of Callable operations in the input order to the input
        image.

    Args:
        operations (sequence): A sequence of operations to be applied to the
            input image in order.
    Returns:
        image (np.ndarray): The resulting image after applying the operations.
    """
    def __init__(self,
                 operations: Union[List, Tuple]) -> None:
        self.operations = operations
        if len(self.operations) == 0:
            raise Warning('The input operations list is empty.')

    def __call__(self,
                 image: np.ndarray) -> np.ndarray:
        image = image.copy()
        for op in self.operations:
            image = op(image)
        return image


class CenterCrop:
    """Crop the central part of the input image.
        Cropping is applied based on the shape of the expected image
        size in pixels.

    Args:
        height (int): The expected image height of the cropped image.
        width (int): The expected image width of the cropped image.
    Return:
        image (np.ndarray): Cropped image.
    """
    def __init__(self,
                 height: int,
                 width: int
    ) -> None:
        self.height = height
        self.width = width
        self.crop = A.CenterCrop(height=self.height,
                                 width=self.width,
                                 always_apply=True)

    def __call__(self,
                 image: np.ndarray
    ) -> np.ndarray:
        return self.crop(image=image)['image']


class RegionCrop:
    """Crop the central part of the input image.
        Cropping is applied based on the shape of the expected image
        size in pixels.

    Args:
        height (int): The expected image height of the cropped image.
        width (int): The expected image width of the cropped image.
    Return:
        image (np.ndarray): Cropped image.
    """
    def __init__(self,
                 height: Tuple,
                 width: Tuple
    ) -> None:
        self.height = height
        self.width = width
        assert isinstance(self.height, Tuple)
        assert isinstance(self.width, Tuple)
        assert len(self.height) == 2
        assert len(self.width) == 2

    def __call__(self,
                 image: np.ndarray
    ) -> np.ndarray:
        return image[
               self.height[0]: self.height[1],
               self.width[0]: self.width[1],
               :
        ]


class Smoothness:
    """Smooth the input image using a predefined filter (which is gaussian here)

    Args:
        sigma (float) or sequence of floating points. Default to `1.0`.
        mode (str): The mode parameter determines how the array borders are handled,
            where cval is the value when mode is equal to `constant`.
            Default is `nearest`.
            Options: {`reflect`, `constant`, `nearest`, `mirror`, `wrap`}
        cval (float): Value to fill past edges of input if mode is
            `constant`. Default is `0.0`.
        multichannel (boolean): A grayscale or multichannel input image.

    Return:
        image (np.ndarray):
    """
    def __init__(self,
                 sigma: float=1.0,
                 mode: str='nearest',
                 cval: float=0.0,
                 multichannel: bool=True) -> np.ndarray:
        self.sigma = sigma
        self.mode = mode
        self.cval = cval
        self.multichannel = multichannel
        self.filter = ndimage.gaussian_filter

    def __call__(self,
                 image: np.ndarray) -> np.ndarray:
        image = self.filter(
                    input=image,
                    sigma=self.sigma,
                    mode=self.mode,
                    cval=self.cval)
        return image


class ImageMaskOverlapping:
    """Mask all the images inside the input directory using their corresponding
        contour masks.
    Args:
        image_dir (str): the directory path of the images to be masked.
        mask_dir (str): the directory path of the masks corresponding to the
            images inside the image_dir.
        out_dir (str): the path of the output directory to same processed images
            in.
        image_extension (str): the format of the images within image_dir
            (e.g. `.tif, .tiff, .png, .jpg, .bmp`).
        mask_extension (str): the format of the masks within mask_dir
            (e.g. `.nrrd, .png, .tif, .tiff, .jpg, .bmp`)
    Return:
    """
    def __init__(self,
                image_dir: str,
                mask_dir: str,
                out_dir: str,
                image_extension: str='.tiff',
                mask_extension: str='.nrrd') -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_extension = image_extension
        self.mask_extension = mask_extension
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

        self.images = sorted(glob.glob(os.path.join(self.image_dir,
                                                    '*'+self.image_extension)))
        self.masks = sorted(glob.glob(os.path.join(self.mask_dir,
                                                    '*'+self.mask_extension)))
        self.image_reader = self.create_reader(self.image_extension)
        self.mask_reader = self.create_reader(self.mask_extension)

    def create_reader(self,
                      extension: str) -> Callable:
        """
        Create a function for reading an image with the provided extension.
        """
        if not extension.startswith('.'):
            extension = '.' + extension
        if extension == '.nrrd':
            return lambda path: np.squeeze(sitk.GetArrayFromImage(
                                                        sitk.ReadImage(path)))
        elif extension in ['.tif', '.tiff']:
            return lambda path: io.imread(path, plugin='tifffile')
        else:
            return lambda path: io.imread(path)

    def __len__(self) -> int:
        assert len(self.images) == len(self.masks), 'Number of images and ' \
                                                    'masks must be the same.'
        return len(self.images)

    def __call__(self) -> None:
        """Save the segmented images into the output directory."""
        for img_pth, msk_pth in zip(self.images, self.masks):
            print('Process: ', img_pth)
            img = self.image_reader(img_pth)
            msk = self.mask_reader(msk_pth)
            img[msk == 0] = 0
            out_path = os.path.join(self.out_dir, os.path.basename(img_pth))
            io.imsave(out_path, img, check_contrast=False)


class RoIExtraction:
    def __init__(self,
                 metadata_path: str,
                 channel_number: None,
                 include_mirror_obj: bool=False):
        self.metadata = pd.read_csv(metadata_path)
        self.channel_number = channel_number
        self.include_mirror_obj = include_mirror_obj

    def extract_single_object(self,
                              image: np.ndarray,
                              mask: np.ndarray,
                              obj_slice: List
    ) -> np.ndarray:
        mask = mask.copy()
        left_side_id = np.unique(mask[obj_slice]).tolist()
        if 0 in left_side_id:
            left_side_id.remove(0)
        assert len(left_side_id) == 1
        mask[mask != left_side_id] = 0

        return image[np.nonzero(mask)]

    def object_cleaner(self,
                       objects: List
    ) -> List:
        for object in objects:
            height = object[0].stop - object[0].start
            width = object[1].stop - object[1].start
            if height < 200 and width < 200:
                objects.remove(object)
        return objects

    def find_mirror_objects(self,
                            original_objects: List
    ) -> List:
        objects = copy.deepcopy(original_objects)
        left_index = np.argmin([ob[1].start for ob in objects])
        mirror_objects = [objects[left_index]]
        objects.remove(objects[left_index])
        if len(objects) == 1:
            return mirror_objects
        while len(objects) > 0:
            left_index = np.argmin([ob[1].start for ob in objects])
            real_obj = objects[left_index]
            objects.remove(real_obj)
            for mirr_obj in mirror_objects:
                if real_obj[1].start < mirr_obj[1].stop:
                    mirror_objects.append(real_obj)
                    break
            else:
                return mirror_objects
        # If mirror object includes all the objects, we leave the right most object as a real object. 
        if len(original_objects) == len(mirror_objects): 
            warnings.warn(f'Mirror region, included all the objects. We left the right most object as a real object.')
            _ = mirror_objects.pop(-1)
        return mirror_objects

    def process_single_image(self,
                             image_path: str,
                             mask_path: str
    ) -> Tuple[Union[None, np.ndarray], np.ndarray, np.ndarray]:
        image = io.imread(image_path)
        if self.channel_number is not None:
            image = image[:, :, self.channel_number]
        mask = io.imread(mask_path)
        labeled, _ = ndimage.label(mask)
        objects = ndimage.find_objects(labeled)
        # Object Cleaner.
        objects = self.object_cleaner(objects)
        if len(objects) == 0:
            raise ValueError(f'There is no object in {mask_path} mask.')
            return None
        elif len(objects) == 1:
            warnings.warn(f'There is only one object in {mask_path} mask.')
            return None, image[objects[0]], mask[objects[0]]
        else:
            mirror_objects = self.find_mirror_objects(objects)
            if self.include_mirror_obj == True:
                min_row = min([ob[0].start for ob in mirror_objects])
                max_row = max([ob[0].stop for ob in mirror_objects])
                min_col = min([ob[1].start for ob in mirror_objects])
                max_col = max([ob[1].stop for ob in mirror_objects])
                mirro_image = image[min_row:max_row, min_col:max_col, :]
            else:
                mirro_image = None

            # Remove mirror objects from the list of objects if there is any object with no overlap with mirror objects. 
            for object in mirror_objects:
                objects.remove(object)
            min_row = min([ob[0].start for ob in objects])
            max_row = max([ob[0].stop for ob in objects])
            min_col = min([ob[1].start for ob in objects])
            max_col = max([ob[1].stop for ob in objects])
            region_image = image[min_row:max_row, min_col:max_col, :]
            # Some regions include two objects which are separated. Only keep 
            #   the largest one. 
            region_mask = mask[min_row:max_row, min_col:max_col]
            region_mask, _ = ndimage.label(region_mask)
            unique_values, unique_counts = np.unique(region_mask, 
                                                     return_counts=True)
            # Find the pixel value of the major region 
            #   excluding background region (pixel value = 0). 
            major_region_pixel_value = unique_values[1:][
                np.argmax(unique_counts[1:])
            ] 
            rows, cols = np.where(region_mask == major_region_pixel_value)
            region_image = region_image[np.min(rows): np.max(rows), 
                                        np.min(cols): np.max(cols), 
                                        :]
            region_mask = region_mask[np.min(rows): np.max(rows), 
                                      np.min(cols): np.max(cols)]
            return mirro_image, region_image, region_mask

    def __len__(self) -> int: 
        return len(self.metadata)

    def __getitem__(self, 
                    item
    ) -> Tuple[pd.Series, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        row = self.metadata.iloc[item, :]
        print(f"Processing {row['Image']}")
        extracted_regions = self.process_single_image(row['Image'], row['Mask'])
        return row, extracted_regions


class FeatureExtraction:
    def histogram_extractor(self, 
                            features: np.ndarray
    ) -> np.ndarray:
        color_features, _ = np.histogram(features.flatten(), bins=range(0, 256))
        return color_features.flatten()

    def glcm(self,
             image: np.ndarray,
             distances: List,
             angles: List,
             levels: int,
             symmetric: bool=False,
             normed: bool=False):
        """Calculate the gray-level co-occurrence matrix.
        """
        return feat.greycomatrix(image, distances, angles, levels, symmetric,
                                 normed)

    def gccrop(self,
               features: np.ndarray,
               prop: str='contrast'):
        """
        Calculate texture properties of a GLCM.
        """
        return feat.greycoprops(features, prop).flatten()

    def lbp(self,
            image: np.ndarray,
            p: int=8,
            r: int=1):
        """
        Gray scale and rotation invariant Local Binary Pattern that is used for text description.
        """
        uniform_features = feat.local_binary_pattern(image, p, r, 'uniform')
        var_features = feat.local_binary_pattern(image, p, r, 'var')
        uniform_features = self.histogram_extractor(uniform_features)
        var_features = self.histogram_extractor(var_features)
        return np.concatenate((uniform_features, var_features))    


def masked_image_visualizer(image: np.ndarray,
                            mask:  np.ndarray=None) -> None:
    """Visualize a pair of image and its mask on top (overlaid), if provided.
    """
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    if mask is not None:
        plt.imshow(mask, cmap='spring', alpha=0.3)
    plt.show()

def config_loader(config_path):
    assert (config_path.endswith('.yaml') or
            config_path.endswith('.yml') or
            config_path.endswith('.json'))

    with open(config_path, 'r') as fin:
        "Reading config file which can be either `Json` or `Yaml` file."
        if config_path.endswith('.json'):
            configs = json.load(fin)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        else:
            raise ValueError('Only `Json` or `Yaml` configs are acceptable.')
    return configs
