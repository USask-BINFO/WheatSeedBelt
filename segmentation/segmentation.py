"""Segmentation Modules"""
import warnings
from typing import Callable, Union

# Imports
import numpy as np
import skimage.color as color
import skimage.filters as filt
import skimage.morphology as morph
import skimage.segmentation as seg
from sklearn.cluster import KMeans
warnings.simplefilter('ignore')

class Morphology:
    """Applying morphology based operations.
    Args:
         disk_radius (int): The radius of the disk-shaped footprint used for
            other morphological operations. The default value is `2`.
         area_threshold (int): The maximum area, in pixels, of a contiguous hole
            that will be filled. Default value is `64`.
         min_size (int): The smallest allowable object size. Default value is `64`.
         connectivity (int): The connectivity defining the neighborhood of a pixel.
            Used during labelling if image array is bool. The default value
            is `2`.
         threshold_method (Callable, None): thresholding method.
            e.g. `skimage.filters.threshold_otsu`. Default to None.
         color_space_converter (Callable, None): Color space converter function.
            e.g. `skimage.color.rgb2gray`. Default to None.
         channel_number (None, int): The channel number to use for
            segmentation, Or use the original input image. Default is None.
         num_kmean_clusters (int, None): Number of clusters in case of using
            Kmeans clustering for segmentation.
        object_mode (str): If the RoI is `dark` or `bright` in the case of
            binary segmentation. Default to 'dark'.
    """
    def __init__(self,
                 disk_radius: int=2,
                 area_threshold: int = 64,
                 min_size: int=64,
                 connectivity: int = 2,
                 threshold_method: Union[None, Callable]=None,
                 color_space_converter: Union[None, Callable]=None,
                 channel_number: Union[None, int]=2,
                 num_kmean_clusters: Union[None, int]=None,
                 object_mode: str= 'dark'):

        # Define attributes.
        self.area_threshold = area_threshold
        self.min_size = min_size
        self.connectivity = connectivity
        self.threshold_method = threshold_method
        self.color_space_converter = color_space_converter
        self.channel_number = channel_number
        self.num_kmean_clusters = num_kmean_clusters
        self.object_mode = object_mode

        self.set_disk(disk_radius)
        self.dilation = morph.binary_dilation
        self.erosion = morph.binary_erosion
        self.closing = morph.binary_closing
        self.opening = morph.binary_opening
        self.rm_small_holes = morph.remove_small_holes
        self.rm_small_objs = morph.remove_small_objects

    def preprocess(self, image):
        if (self.color_space_converter is not None and
            isinstance(self.color_space_converter, Callable)):
            image = self.color_space_converter(image)
        if self.channel_number is not None:
            if image.ndim == 2:
                raise ValueError(f'Input image is a 2 dimentinoal GrayScale image.')
            image = image[:, :, self.channel_number]
        return image

    def set_disk(self, disk_radius):
        self.disk_radius = disk_radius
        self.disk = morph.disk(radius=self.disk_radius, dtype=bool)

    def apply_kmeans(self, image: np.ndarray):
        image = self.preprocess(image)
        image_dim = 3 if image.ndim == 3 else 1
        image_vector = image.reshape((-1, image_dim))
        kmeans = KMeans(n_clusters=self.num_kmean_clusters).fit(image_vector)
        mask = kmeans.labels_.flatten().reshape(image.shape[:2])
        return mask

    def apply_thresholding(self, image: np.ndarray):
        """Apply thresholding operation and return the resulted boolean
            image.
       Args:
           image (numpy.ndarray): An numpy array image, that can be binary,
               grayscale, or a color image.
       Returns:
           image (numpy.ndarry): A binary image of type numpy boolean ndarray.
        """
        image = self.preprocess(image)
        thresholds = self.threshold_method(image)
        if isinstance(thresholds, (int, float, np.int64, np.float32)):
            if self.object_mode == 'bright':
                image[image < thresholds] = 0
            else:
                image[image > thresholds] = 0
            image[image != 0] = 1
        else:
            image = np.digitize(image, thresholds)
        return image

    def binarize(self, image: np.ndarray, objects2keep: int=1):
        image[image != objects2keep] = 0
        image[image == objects2keep] = 1
        return image

    def apply_dilation(self, image: np.ndarray):
        """Apply dilation operation and return the resulted boolean image.
        Args:
            image (numpy.ndarray): A binary numpy array.
        Returns:
            image (numpy.ndarry): A dilated binary image.
        """
        dilated = self.dilation(image, self.disk).astype(np.uint8)
        return dilated

    def apply_erosion(self, image: np.ndarray):
        """Apply erosion operation and return the resulted boolean image.
        Args:
            image (numpy.ndarray): A binary numpy array image.
        Returns:
            image (numpy.ndarry): An eroded binary image.
        """
        eroded = self.erosion(image, self.disk).astype(np.uint8)
        return eroded

    def apply_closing(self, image: np.ndarray):
        """Apply closing operation and return the resulted boolean image.
        Args:
            image (numpy.ndarray): A binary numpy array image.
        Returns:
            image (numpy.ndarry): A binary image.
        """
        closed = self.closing(image, self.disk).astype(np.uint8)
        return closed

    def apply_opening(self, image: np.ndarray):
        """Apply closing operation and return the resulted boolean image.
        Args:
            image (numpy.ndarray): A binary numpy array image.
        Returns:
            image (numpy.ndarry): A binary image.
        """
        opened = self.opening(image, self.disk).astype(np.uint8)
        return opened

    def apply_rm_small_holes(self, image: np.ndarray):
        """The input array with small holes within connected components
            removed.
        """
        if len(np.unique(image)) <= 2:
            if image.dtype != np.bool:
                image = image.astype(np.bool_)
        else:
            raise ValueError('Input image must be binary.')
        cleaned = self.rm_small_holes(ar=image,
                                      area_threshold=self.area_threshold,
                                      connectivity=self.connectivity)
        return cleaned

    def apply_rm_small_objects(self, image: np.ndarray):
        """The input array with small connected components removed.
        """
        if len(np.unique(image)) <= 2:
            if image.dtype != np.bool:
                image = image.astype(np.bool_)
        else:
            raise ValueError('Input image must be binary.')
        cleaned = self.rm_small_objs(ar=image,
                                     min_size=self.min_size,
                                     connectivity=self.connectivity)
        return cleaned


class RandomWalker:
    """Using RandomWalker to segment an image from a set of markers.
    Args:
        beta (int): Penalization coefficient for the random walker motion
            (the greater beta, the more difficult the diffusion).
        mode (str): Mode for solving the linear system in the random walker algorithm.
            Optoions are: `cg`, `cg_j`, `cg_mg`, and `bf`. Default to `bf`.
        connectivity (int): The connectivity defining the neighborhood of a pixel.
            The default value is `2`.
        multichannel (boolean): 
    """
    def __init__(self,
                 beta: int=10,
                 mode: str='bf',
                 connectivity: int=2, 
                 multichannel: bool=False):
        # Define attributes.
        self.beta = beta
        self.mode = mode
        self.connectivity = connectivity
        self.multichannel = multichannel
        self.random_walker = seg.random_walker

    def __call__(self, image: np.ndarray, markers: np.ndarray=None):
        if markers is None:
            markers = morph.label(image, connectivity=self.connectivity)
        labeled = self.random_walker(data=image,
                                     labels=markers,
                                     beta=self.beta,
                                     mode=self.mode,
                                     multichannel=self.multichannel)
        return labeled
