import gc 
import json
import os
import random 
from typing import Dict, List, Tuple, Callable, Optional, Union

import numpy as np
import pandas as pd

import torch
import torchmetrics
import yaml
from torchmetrics import functional
import seaborn as sns
import matplotlib.pyplot as plt 
from enum import Enum

    
class Loss(torch.nn.Module): 
    """Calculate loss by combining different losses for a batch of data.
    Args: 
        task (str): the task of the classification mode. 
            `binary` or `multi` mode is used to preprocess the model output for 
            caluclating the loss. 
    Return: 
        loss (Tensor): the torch.Tensor calculated loss. 
    """
    def __init__(self,
                 task: str='multiclass'
    ) -> None:
        super(Loss, self).__init__()
        
        self.task = task
        if self.task == 'multiclass': 
            self.ce = torch.nn.CrossEntropyLoss()
        elif self.task == 'binary':
            self.ce = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError('Only `binary` or `multiclass` classification task modes are supported.')
    
    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        preds = preds.squeeze()
        if self.task == 'binary': 
            labels = labels.float()
        loss = self.ce(preds, labels)
        return loss

    
class BinaryMetrics(Enum): 
    ACCURACY = torchmetrics.classification.BinaryAccuracy
    PRECISION = torchmetrics.classification.BinaryPrecision
    RECALL = torchmetrics.classification.BinaryRecall
    F1SCORE = torchmetrics.classification.BinaryF1Score
    
    
class MulticlassMetrics(Enum): 
    ACCURACY = torchmetrics.classification.MulticlassAccuracy
    PRECISION = torchmetrics.classification.MulticlassPrecision
    RECALL = torchmetrics.classification.MulticlassRecall
    F1SCORE = torchmetrics.classification.MulticlassF1Score
     

class Score: 
    """Calculate the classification scores such as `accuracy`, `precision`, 
        `recall`, and `f1-score`.
    Args: 
        task (str): The binary or non-binary classification task. 
            `binary` or `multiclass` mode is used to preprocess the model output for 
            caluclating the loss. 
        num_classes (int): The number of classes in the classification task. 
        classes (sequence): The name of the classess to be used in visualizing the 
            confusion matrix. 
        average (str): This parameter is required for multiclass/multilabel tasks. 
            The options are ‘micro’, ‘macro’, ‘weighted’, ‘none’. 
            More information about each option in `TorchMetrics`. 
        sigmoid (bool): set it to `True` for binary classifiers. 
        save_confmatrix (bool): set it to `True` if you want to calculate 
            and save confusion matrix.
        conf_out_dir (str): The directory address for saving the confusion matrices.
        confmat_normalize (str): The normalization method for the confusion matrix. Options are 'true', 'none'.
    Returns: 
        scores (Dict): a dictionary contains the scores. Each key is the score name and its value 
            would the flaoting point score value.
    """
    
    def __init__(self, 
                 task: str='multiclass',
                 num_classes:int=3,
                 classes: Union[Tuple[str], List[str]]=('class0', 'class1', 'class2'),
                 average: str='micro', 
                 sigmoid: bool=False, 
                 binary_threshold: float=0.5,
                 save_confmatrix: bool=False,
                 conf_out_dir: str='ConfusionMatrices/',
                 confmat_normalize: str = 'true'
    ) -> None:
        self.task = task
        self.num_classes = num_classes
        self.classes = classes
        self.average = average
        self.sigmoid = sigmoid
        self.binary_threshold=binary_threshold
        self.save_confmatrix = save_confmatrix
        self.conf_out_dir = conf_out_dir
        self.confmat_normalize = confmat_normalize

        if self.task == 'binary': 
            self.metric_functions = {
                        met.name: met.value(threshold=self.binary_threshold)
                        for met in BinaryMetrics
            }
            self.confmat = torchmetrics.classification.BinaryConfusionMatrix(
                    threshold=self.binary_threshold, 
                    normalize=self.confmat_normalize
            )
        elif self.task == 'multiclass': 
            self.metric_functions = {
                        met.name: met.value(num_classes=self.num_classes, average=self.average)
                        for met in MulticlassMetrics
            }
            self.confmat = torchmetrics.classification.MulticlassConfusionMatrix(
                    num_classes=self.num_classes, 
                    normalize=self.confmat_normalize
            )
        else: 
            raise ValueError('Only `binary` or `multiclass` classification task modes are supported.')

        # Define the collection to save the input data. 
        self.reset()
        
    def update(self, 
               preds: torch.Tensor, 
               targets: torch.Tensor
    ) -> Dict:
        """Update the prediction and targest collection by the new batch data."""
        preds, targets = self.format_labels(preds, targets)
        self.preds_collection = torch.hstack([self.preds_collection, preds])
        self.targets_collection = torch.hstack([self.targets_collection, targets])
    
    def reset(self
    ) -> None: 
        """Reset the collections for the new collection process of the new epoch."""
        self.preds_collection = torch.tensor([], dtype=torch.short)
        self.targets_collection = torch.tensor([], dtype=torch.short)
    
    def close(self
    ) -> None: 
        """Remove the collections and free the memory."""
        del self.preds_collection
        del self.targets_collection
        gc.collect()
    
    def compute_scores(self
    ) -> Dict: 
        """Compute the defined scores for the collected predictisons/targets."""
        assert len(self.preds_collection) > 0, 'The prediction collection is empty.'
        assert len(self.targets_collection) > 0, 'The target collection is empty.'
        scores = {}
        for key, func in self.metric_functions.items(): 
            scores[key] = func(self.preds_collection, 
                               self.targets_collection
            ).item()
        return scores
    
    def compute_confmatrix(self, 
                           conf_file_name: str
    ) -> None: 
        """Build and save the confusion matrix as an image in the provided output path."""
        assert len(self.preds_collection) > 0, 'The prediction collection is empty.'
        assert len(self.targets_collection) > 0, 'The target collection is empty.'
        if self.save_confmatrix is True:
            cf_matrix = self.confmat(
                    self.preds_collection, 
                    self.targets_collection                                     
            ).cpu().numpy()
            cf_df = pd.DataFrame(
                    cf_matrix, 
                    index = [i for i in self.classes],
                    columns = [i for i in self.classes]
            )
            # Map [0, 1] percentage to a [0, 100] percentage.
            cf_df = cf_df.mul(100)
            plt.figure(figsize = (10, 10))
            heatmap = sns.heatmap(cf_df, annot=True, fmt='.1f', square=1, linewidth=1., cbar=False, annot_kws={"fontsize": 24})
            for t in heatmap.texts: 
                t.set_text(t.get_text() + ' %')
            heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=24)
            heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=24)
            plt.xlabel('Predicted', fontsize=32, color='tab:red')
            plt.ylabel('Actual', fontsize=32, color='tab:cyan')
            sns.set(font_scale=2.0)
            plt.savefig(os.path.join(self.conf_out_dir, conf_file_name))
        else: 
            print('Warning: Confusion matrix calculator has not been activated.')
    
    def format_labels(self, 
                   preds: torch.Tensor,
                   targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        """Convert the predicted probablities to labels and correct the prediction and targets 
            formats based on the `binary` or `multiclass` tasks.
        """
        preds = preds.clone().cpu()
        targets = targets.clone().cpu()
        if self.sigmoid is True and self.task == 'binary':
            preds = torch.ge(torch.sigmoid(preds), self.binary_threshold)
        elif self.sigmoid is False and self.task == 'binary': 
            preds = torch.ge(preds, self.binary_threshold)
        else:
            _, preds = torch.max(preds, dim=1)
        preds = preds.flatten().type(torch.short)
        targets = targets.flatten().type(torch.short)
        return preds, targets
    
    def format_metrics(self, 
                        scores: Dict
    ) -> str: 
        """Format the calucalted metrics into a printable string message."""
        formatted = []
        for key, val in scores.items(): 
            formatted.append(f'{key}: {val: 0.4f}')
        return ', '.join(formatted)   


class Pad: 
    """Pad the input image and mask with a constant value if their size is less the the 
        requested sizes, else no padding would be applied.
    Args: 
        height (int): height of th image and mask after padding. Default to `512`. 
        width (int): width of th image and mask after padding. Default to `512`.
        depth (int): depth of th image after padding. Default to `3`.
        image_constant (int): a constant value to pad the image. Default to `0`.
        mask_constant (int): a constant value to pad the mask. Default to `0`.
        always_apply (bool): always apply the transformation or not. 
        p (float): the probability of applying the transformation on the input image and mask. 
    Returns: 
        {'image': np.ndarray, 'mask': np.ndarray}
        
    -- Note: last channel image is accepted. 
    """
    def __init__(self, 
                 height: int=512,
                 width: int=512,
                 depth: int=3,
                 image_constant: int=0,
                 mask_constant: int=0, 
                 always_apply: bool=False, 
                 p: float=1.0
    ) -> None: 
        self.height = height
        self.width = width
        self.depth = depth
        self.image_constant = image_constant
        self.mask_constant = mask_constant
        self.always_apply = always_apply
        self.p = p
        if self.always_apply is True: 
            self.p = 1.0
    
    def __call__(self, 
                 image: Optional[np.ndarray]=None, 
                 mask: Optional[np.ndarray]=None
    ) -> Dict:
        if random.random() <= self.p: 
            if image is not None: 
                hight_pad = (self.height - image.shape[0]) if image.shape[0] < self.height else 0
                width_pad = (self.width - image.shape[1]) if image.shape[1] < self.width else 0
                depth_pad = (self.depth - image.shape[2]) if image.shape[2] < self.depth else 0
                image = np.pad(
                    image, 
                    ((hight_pad // 2, hight_pad - hight_pad // 2), 
                     (width_pad // 2, width_pad - width_pad // 2), 
                     (depth_pad // 2, depth_pad - depth_pad // 2)
                    ),
                    mode='constant', 
                    constant_values=self.image_constant
                )
            if mask is not None: 
                hight_pad = (self.height - mask.shape[0]) if mask.shape[0] < self.height else 0
                width_pad = (self.width - mask.shape[1]) if mask.shape[1] < self.width else 0
                mask = np.pad(
                    mask, 
                    ((hight_pad // 2, hight_pad - hight_pad // 2), 
                     (width_pad // 2, width_pad - width_pad // 2), 
                     (0, 0)
                    ),
                    mode='constant', 
                    constant_values=self.mask_constant
                )
        return {'image': image, 'mask': mask}
    

class MinMaxScaler: 
    """Scale the image intensity values into the range of zero and one, `[0, 1]`. 
    Args: 
        always_apply (boolean): Whether always apply this transformation to the inputs or not. 
        p (float): The probability of applying the transformation to the inputs. 
    Returns: 
        A dictionary with the following items: 
            `image`: the resulting ndarray image
            `mask`: the resulting ndarray image
    """
    def __init__(self, 
                 always_apply: bool=True, 
                 p: float=True
    ) -> None: 
        self.always_apply = always_apply
        self.p = p
        if self.always_apply is True: 
            self.p = 1.0
            
    def __call__(self, 
                 image: Optional[np.ndarray]=None, 
                 mask: Optional[np.ndarray]=None
    ) -> Dict:
        if random.random() <= self.p: 
            image = (image - image.min()) / (image.max() - image.min())
        if image is not None: 
            image = image.astype(np.float32)
        if mask is not None: 
            mask = mask.astype(np.int64)
        return {'image': image, 'mask': mask}
    
    
class Wrapper: 
    """A transformation composer wrapper that applies manual transformations, 
        in order, on the input images.
    Args: 
        transformations (sequence): A list of augmentation transformations to 
            be applied in order on the input images. 
    Returns: 
        A dictionary with the following items: 
            `image`: the resulting ndarray image after applying all the transformations.
    """
    def __init__(self, 
                 transformations: List
    ) -> None: 
        self.transformations = transformations
    
    def __call__(self, 
                 image: Optional[np.ndarray]=None
    ) -> Dict:
        for tr in self.transformations:
            image = tr(image=image)['image']
        return {'image': image}
    

def config_loader(config_path: str) -> Dict:
    """Load configuration either from a `YAML` or `JSON` file.
    Args: 
        config_path (str): The path to the config file. 
    Returns: 
        configs (Dict): a dictionary contains all the configuration parameters. 
    """
    assert (config_path.endswith('.yaml') or
            config_path.endswith('.yml') or
            config_path.endswith('.json'))

    with open(config_path, 'r') as fin:
        if config_path.endswith('.json'):
            configs = json.load(fin)
        elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
            configs = yaml.load(fin, Loader=yaml.FullLoader)
        else:
            raise ValueError('Only `Json` or `Yaml` configs are acceptable.')
    return configs
    
