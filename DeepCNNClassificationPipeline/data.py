import warnings
from collections import Counter
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io 

warnings.filterwarnings("ignore")


class BeltData(Dataset):
    """Belt Wheat Dataset.
    Args:
        metadata_paths (sequence): 
        root_dir (str):  
        transform (Callable):
        supervised (bolean): 
    """
    def __init__(self, 
                 metadata_paths: Union[List, Tuple],
                 root_dir: str='./', 
                 transform: Union[None, Callable]=None,
                 supervised: bool=True
    ) -> None:   
        self.metadata_paths = metadata_paths
        self.root_dir = root_dir
        self.transform = transform
        self.supervised = supervised

        self.metadata = None 
        for path in self.metadata_paths:
            df = pd.read_csv(path)
            if self.metadata is None: 
                if 'Label' in df.columns.tolist() and self.supervised: 
                    self.metadata = pd.DataFrame({
                        'ImageID': [],
                        'Image': [], 
                        'Label': []
                    })
                else: 
                    self.metadata = pd.DataFrame({
                        'ImageID': [],
                        'Image': []
                    })
                    
            self.metadata = pd.concat(
                    [self.metadata, df[self.metadata.columns.tolist()]], 
                    axis=0, 
                    ignore_index=True
            )
        if 'Label' in self.metadata.columns.tolist(): 
            self.metadata = self.metadata.astype({
                'Label': np.int32
            })
            # Calcualte class weights. 
            labels_count = Counter(self.metadata.loc[:, 'Label'])
            self.class_weights = {l: 1.0 - labels_count[l] / sum(labels_count.values()) for l in labels_count.keys()}
            self.samples_weight = torch.FloatTensor([
                self.class_weights[l] 
                for l in self.metadata.loc[:, 'Label'].tolist()
            ])
        else: 
            self.class_weights = None
            self.samples_weight = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, 
                    item: int
    ) -> Tuple[torch.Tensor, Union[np.number, str]]:
        if self.metadata.loc[item, 'Image'].endswith('.npy'): 
            image = np.load(self.metadata.loc[item, 'Image'])        
        else: 
            image = io.imread(self.metadata.loc[item, 'Image'])
        img_min_val, img_max_val = image.min(), image.max()
        assert img_min_val >= 0 and img_max_val <= 255, f"Image: {self.metadata.loc[item, 'Image']}\
                                                          value range error. Image must be of type uint8 or float32."
        if self.transform is not None: 
            image = self.transform(image=image)['image']
        else: 
            if img_min_val < 0 or img_max_val > 255: 
                image = (image - img_min_val) / (img_max_val - img_min_val)
            image = torch.FloatTensor(image)
        if self.supervised == True: 
            label = self.metadata.loc[item, 'Label'].astype(np.int64)
        else: 
            if 'ImageID' in self.metadata.columns: 
                label = self.metadata.loc[item, 'ImageID']
            else:
                label = self.metadata.loc[item, 'Image'] # In this case label serves the prediction pipeline as the unique id. 
        return image, label
    
    @staticmethod
    def collate(batch: List
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        images, labels = list(zip(*batch))
        images = torch.stack(images, dim=0)
        if isinstance(labels, (int, float)): 
            labels = torch.tensor(labels, dtype=torch.long)
        return images, labels
            
        
        
        
