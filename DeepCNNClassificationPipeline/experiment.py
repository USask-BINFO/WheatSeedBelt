import argparse
import importlib
import logging
import os
import shutil
import sys
import random 
import warnings
import wandb
from asyncio.log import logger
from collections import Counter
from random import shuffle
from typing import Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from data import BeltData
from development import Development
from model import Model
from utils import Loss, Score, config_loader

warnings.filterwarnings("ignore")

# Set Seeds. 
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)


if __name__ == '__main__': 
    # Define input arguments.
    parser = argparse.ArgumentParser(description='Trainer Params.')
    parser.add_argument('-c', '--config', dest='config_path', type=str, default='configs/experiment.yaml',
                    help='The string path of the config file.')
    args = parser.parse_args()
    configs = config_loader(args.config_path)

    # Define output directory. 
    output_directory = os.path.join(configs['out_dir'], configs['experiment']) 
    os.makedirs(os.path.join(output_directory, configs['model']['development_phase']), exist_ok=True)
    os.makedirs(os.path.join(output_directory, 'ConfusionMatrices'), exist_ok=True)
    shutil.copy(args.config_path, os.path.join(output_directory, configs['model']['development_phase']))

    # Define Logger. 
    log_file = os.path.join(output_directory, 'experiment.log')   
    logging.basicConfig(
        format='',                                                                                               
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file, 'a'),
            logging.StreamHandler(sys.stdout) 
        ]   
    )  
    logger = logging.getLogger('')
    logger.info('-' * 50)
    logger.info('-' * 50)
    
    # Set WandB
    if configs['model']['development_phase'] == 'train' and configs['wandb']['apply'] is True:
        wandb.init(                                                                                                                                             
            project=configs['wandb']['project'], 
            name=configs['experiment'], 
            entity=configs['wandb']['entity'], 
            config=configs
        ) 
    else: 
        wandb = None

    # Define Transformations. 
    phases = configs['phases']
    transformers = importlib.import_module(configs['transforms']).transforms
    transformers = {
        phase: transformers[phase] 
        for phase in phases
    }

    # Define datasets. 
    for phase in phases: 
        logger.info(f'Used metatdatas for {phase}:')
        for item in configs['metadata_paths'][phase]:
            logger.info(f'  - {item}')
    datasets = {
        phase: BeltData(
            metadata_paths=configs['metadata_paths'][phase], 
            root_dir=configs['root_dir'][phase], 
            transform=transformers[phase]
        )   
        for phase in phases
    }

    # Define data samplers. 
    samplers = {
        phase: None
        for phase in phases
    }
    if configs['model']['development_phase'] == 'train' and configs['use_sampler'] == True:
        samplers['train'] = WeightedRandomSampler(
                datasets['train'].samples_weight, 
                len(datasets['train']), 
                replacement=True, 
                generator=None
        )

    # Define data loaders. 
    loaders = {
        phase: DataLoader(
            dataset=datasets[phase], 
            batch_size=configs['batch_size'],
            num_workers=configs['num_workers'],
            shuffle=(configs['model']['development_phase'] == 'train' and configs['use_sampler'] == False),
            sampler=samplers[phase]
        )
        for phase in phases
    }
    
    # Define the training model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(
        model_name=configs['model']['model_name'], 
        pretrained=configs['model']['imagenet_pretrained'], 
        num_classes=configs['model']['num_classes'],
        device=device
    )
    if configs['model']['pretrained'] == True and configs['model']['pretrained_path'] != None:
        model.load(path=configs['model']['pretrained_path'], 
                   logger=logger)
    elif configs['model']['imagenet_pretrained'] is True: 
        logger.info('Used ImageNet pretrained model.')
    
    
    # Watch the model. 
    if wandb is not None: 
        wandb.watch(model)
    
    # Define optimzer
    optimizer = {
        'SGD': torch.optim.SGD(
            model.parameters(), 
            lr=configs['optimizer']['lr'], 
            momentum=configs['optimizer']['momentum'], 
            weight_decay=configs['optimizer']['weight_decay']
        ),
        'Adam': torch.optim.Adam(
            model.parameters(), 
            lr=configs['optimizer']['lr'], 
            betas=(0.9, 0.999), 
            eps=1e-08, 
            weight_decay=configs['optimizer']['weight_decay'], 
            amsgrad=False
        )
    }[configs['optimizer']['name']]
    if configs['model']['development_phase'] == 'train':
        logger.info(optimizer)
    
    # Define scheduler 
    scheduler = {
        'ExponentialLR': torch.optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=configs['scheduler']['gamma']
        ),
        'StepLR': torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=configs['scheduler']['step_size'], 
            gamma=configs['scheduler']['gamma']
        )
    }[configs['scheduler']['name']]
    if configs['model']['development_phase'] == 'train':
        logger.info(scheduler)

    # Define Criteria and Metrics. 
    criterion = Loss(
        task=configs['loss']['task']
    )
    metrics = Score(
        task=configs['loss']['task'],
        num_classes=configs['scoring']['num_classes'],
        classes=configs['scoring']['classes'],
        average=configs['scoring']['average'], 
        sigmoid=configs['scoring']['sigmoid'],
        save_confmatrix=(configs['scoring']['save_confmatrix'] == True or 
                         configs['model']['development_phase'] == 'test'),
        conf_out_dir=os.path.join(output_directory, 'ConfusionMatrices'),
        confmat_normalize=configs['scoring']['confmat_normalize']
    )

    # Define devemopment object. 
    developer = Development(
        model=model, 
        loaders=loaders,
        optimizer=optimizer, 
        scheduler=scheduler, 
        num_epochs=configs['num_epochs'],
        criterion=criterion,
        metrics=metrics,
        device=device,
        logger=logger, 
        wandb=wandb,
        out_dir=output_directory,
        save_valid_confmatrix=configs['scoring']['save_confmatrix']
    )
    # Train 
    if configs['model']['development_phase'] == 'train': 
        logger.info('-' * 10)
        logger.info('Start Training ...')
        developer.train()
    # Test
    if configs['model']['development_phase'] == 'test': 
        logger.info('-' * 10)
        if configs['model']['pretrained'] == False or configs['model']['pretrained_path'] is None: 
            logger.info('Pretrained model did not use.')
        logger.info('Start Testing ...')
        test_loss, test_scores = developer.run_phase(phase='test')
        logger.info(f'Test Loss: {test_loss:0.4f}, Scores: {metrics.format_metrics(test_scores)}')
        # Generate the confusion matrix.
        metrics.compute_confmatrix(
            conf_file_name=f"{configs['scoring']['conf_prefix']}_testing_confmat.png"
        )
    # Remvoe the collected data in the metrics object. 
    metrics.close()
