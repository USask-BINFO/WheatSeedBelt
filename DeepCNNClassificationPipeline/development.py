import logging
import os
import sys
import time
from typing import Callable, Tuple, Optional
from collections import Counter

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt 


class Development:
    """A class for the development of the model.
    Args:
        model (Callable): A model to be trained.
        loaders (Dict): A dictionary of data loaders including the defined phases such as train, valid, test, predict.
        optimizer (Callable): An optimizer for the model.
        scheduler (Callabel): A scheduler for the optimizer.
        num_epochs (int): The number of epochs to train the model.
        criterion (Callable): A loss function.
        metrics (Callable): A dictionary of metrics.
        device (torch.device): A device to train the model.
        logger (Callable): A logger.
        wandb (Callable): A wandb object.
        out_dir (str): A directory to save the model.
        save_valid_confmatrix (boolean): A flag to save the confusion matrix of the validation set.
    """
    def __init__(
            self, 
            model: Callable, 
            loaders: Callable,
            optimizer: Callable=None, 
            scheduler: Callable=None, 
            num_epochs: int=None,
            criterion: Callable=None,
            metrics: Callable=None,
            device: torch.device=torch.device('cpu'),
            logger: Callable=None, 
            wandb: Optional[Callable]=None,
            out_dir: str='out', 
            save_valid_confmatrix: bool=False
    ) -> None:
        self.model = model
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler 
        self.num_epochs=num_epochs
        self.criterion = criterion
        self.metrics = metrics
        self.device = device
        self.wandb = wandb
        self.out_dir = out_dir
        self.save_valid_confmatrix = save_valid_confmatrix
        if logger is not None: 
            self.logger = logger 
        else: 
            self.logger = logging.basicConfig(
                format='',                                                                                                                                                                                                                                                             
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(os.path.join(self.out_dir, 'experimnet.log'), 'a'),
                    logging.StreamHandler(sys.stdout)
            ])

    def train(self) -> None:
        # Start training.
        since = time.time()
        # Save the initial model. 
        os.makedirs(os.path.join(self.out_dir, 'weights'), exist_ok=True)
        self.model.save(path=os.path.join(self.out_dir, 'weights', f'epoch{0:0>3}.pt'), 
                        logger=self.logger)
        best_score = 0.0
        for epoch in range(1, self.num_epochs + 1):
            self.logger.info(f'Epoch {epoch:0>3} / {self.num_epochs}')
            
            # Train
            train_loss, train_scores = self.run_phase(phase='train')
                
            # Validate
            valid_loss, valid_scores = self.run_phase(phase='valid')
            if self.save_valid_confmatrix is True: 
                self.metrics.compute_confmatrix(
                    conf_file_name=f"{epoch:0>3}_validation_confmat.png"
                )

             # Run scheduler if is not None.
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Print out the results
            self.logger.info(f'Train Loss: {train_loss:0.4f}, Scores: {self.metrics.format_metrics(train_scores)}')
            self.logger.info(f'Valid Loss: {valid_loss:0.4f}, Scores: {self.metrics.format_metrics(valid_scores)}')
            
            # Check for the best model
            if best_score < valid_scores['F1SCORE']: 
                best_score = valid_scores['F1SCORE']
                self.model.save(path=os.path.join(self.out_dir, 'weights', f'epoch{epoch:0>3}.pt'), 
                                logger=self.logger)
            self.logger.info('')
            
            # Track changes in losses and accuracy. 
            if self.wandb:
                info2track = {
                    "Epoch": epoch,
                    "Train Loss".title(): train_loss,
                    "Valid Loss".title(): valid_loss
                }
                for key in self.metrics.metric_functions.keys():
                    info2track[f'Train {key}'.title()] = train_scores[key]
                    info2track[f'Valid {key}'.title()] = valid_scores[key]
                self.wandb.log(info2track)

        time_elapsed = time.time() - since
        print(f'Training completed in {time_elapsed // 60:0.0f}m {time_elapsed % 60:.0f}s.')
        print(f'Best Validation Score: {best_score:0.4f}')

    def run_phase(self, 
                  phase: str
    ) -> Tuple[float, float]: 
        if phase == 'train': 
            self.model.train()
        else:
            self.model.eval()
        # Reset the metrics collections.
        self.metrics.reset()
        with torch.set_grad_enabled(phase == 'train'):
            epoch_loss = 0.0
            epoch_counter = 0
            for inputs, targets in tqdm(self.loaders[phase]):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if phase == 'train':
                    self.optimizer.zero_grad()
                
                outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, targets)
                
                if phase == 'train': 
                    batch_loss.backward()
                    self.optimizer.step()
                
                self.metrics.update(outputs, targets)

                epoch_loss += batch_loss.item()
                epoch_counter += 1
            
            epoch_loss = epoch_loss / epoch_counter
            epoch_scores = self.metrics.compute_scores()
        return epoch_loss, epoch_scores
