import copy
import logging
import os
from enum import Enum
from tkinter.tix import InputOnly
from typing import Optional

import torch
import torchvision
from torch import nn


class ModelFeatureMapDim(Enum): 
    BASEMODEL = 512
    RESNET18 = 512
    RESNET34 = 512
    RESNET50 = 2048
    RESNET101 = 2048
    INCEPTIONV3 = 2048
    EFFICIENTNET_B0 = 1280
    EFFICIENTNET_B1 = 1280
    EFFICIENTNET_B2 = 1408
    EFFICIENTNET_B3 = 1536
    EFFICIENTNET_B4 = 1792
    EFFICIENTNET_B5 = 2048


class ModelFCTitle(Enum): 
    BASEMODEL = None
    RESNET18 = 'fc'
    RESNET34 = 'fc'
    RESNET50 = 'fc'
    RESNET101 = 'fc'
    INCEPTIONV3 = 'fc'
    EFFICIENTNET_B0 = 'classifier'
    EFFICIENTNET_B1 = 'classifier'
    EFFICIENTNET_B2 = 'classifier'
    EFFICIENTNET_B3 = 'classifier'
    EFFICIENTNET_B4 = 'classifier'
    EFFICIENTNET_B5 = 'classifier'


class ModelFunctions: 
    BASEMODEL = None
    RESNET18 = torchvision.models.resnet18
    RESNET34 = torchvision.models.resnet34
    RESNET50 = torchvision.models.resnet50
    RESNET101 = torchvision.models.resnet101
    INCEPTIONV3 = torchvision.models.inception_v3
    EFFICIENTNET_B0 = torchvision.models.efficientnet_b0
    EFFICIENTNET_B1 = torchvision.models.efficientnet_b1
    EFFICIENTNET_B2 = torchvision.models.efficientnet_b2
    EFFICIENTNET_B3 = torchvision.models.efficientnet_b3
    EFFICIENTNET_B4 = torchvision.models.efficientnet_b4
    EFFICIENTNET_B5 = torchvision.models.efficientnet_b5


class BaseModel(nn.Module):
    """A cusotm CNN model for the classification task.
    Args:
        num_classes (int): The number of classes to be predicted.
        dropout (float): The dropout rate.
    """
    def __init__(self,
                 num_classes: int = 1,
                 dropout: float = 0.5,
    ) -> None:
        super(BaseModel, self).__init__()

        self.num_classes = num_classes
        self.dropout = dropout
        self.feature_map_dim = ModelFeatureMapDim['BASEMODEL'].value

        self.basemodel = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(num_features=self.feature_map_dim),
            nn.Dropout(self.dropout),
            nn.Linear(in_features=self.feature_map_dim,
                      out_features=self.num_classes)
        )

    def forward(self,
                data: torch.Tensor
    ) -> torch.Tensor:
        return self.basemodel(data)


class Model(nn.Module):
    """Deep Convolutional Neural Network for Image Classification
    Args:
        model_name (str): name of the model to use defined in the ModelFunctions Enum
        pretrained (bool): whether to use pretrained ImageNet weights
        num_classes (int): number of classes to predict
        dropout (float): dropout rate of the last dropout layer in the classifier model
        device (torch.device): device to use for training
    """
    def __init__(self, 
                 model_name: str='efficientnet_b4',
                 pretrained: bool=True, 
                 num_classes: int=1,
                 dropout: float=0.5,
                 device: str='cpu'
    ) -> None:
        super(Model, self).__init__()
        
        self.model_name = model_name
        if self.model_name not in ModelFeatureMapDim.__members__: 
            print('Availabel models are: ', list(ModelFeatureMapDim.__members__))
            raise ValueError('the requested model is not available.')
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device 
        
        # Define the model.
        if ModelFCTitle[self.model_name].value is not None:
            self.model: nn.Module = getattr(ModelFunctions, self.model_name)(pretrained=self.pretrained)
        self.feature_map_dim: int = ModelFeatureMapDim[self.model_name].value
        
        # Define the classifier. 
        if ModelFCTitle[self.model_name].value == 'fc':
            print(f'Model `{self.model_name}` is used!')
            self.model.fc = nn.Sequential(
                nn.BatchNorm1d(num_features=self.feature_map_dim),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.feature_map_dim, 
                          out_features=self.num_classes)
            )
        elif ModelFCTitle[self.model_name].value == 'classifier': 
            print(f'Model `{self.model_name}` is used!')
            self.model.classifier =  nn.Sequential(
                    nn.Flatten(),
                    nn.BatchNorm1d(num_features=self.feature_map_dim),
                    nn.Dropout(self.dropout),
                    nn.Linear(in_features=self.feature_map_dim, 
                              out_features=self.num_classes)
            )
        elif ModelFCTitle[self.model_name].value is None: 
            print(f'Model `{self.model_name}` is used!')
            self.model = BaseModel(num_classes=self.num_classes,
                                   dropout=self.dropout)
        else: 
            raise ValueError('The classifier layer type is not recognized.')
            
        # Send the model on the device.  
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

    def forward(self, 
                data: torch.Tensor
    ) -> torch.Tensor:
        return self.model(data)
    
    def save(self, 
             path: str='model.pt', 
             logger: Optional[logging.RootLogger]=None
    ) -> None: 
        if torch.cuda.device_count() > 1:
            torch.save(copy.deepcopy(self.module.state_dict()),  path)
        else: 
            torch.save(copy.deepcopy(self.state_dict()), path)
        if logger is not None: 
            logger.info(f'Model saved into {path}.')

    def load(self, path='model.pth', logger=None): 
        if os.path.isfile(path) is True: 
            if logger is not None:
                logger.info(f'Pretrained model loaded from {path}.')
            else:
                print(f'Pretrained model loaded from {path}.')
        self.load_state_dict(torch.load(path))
