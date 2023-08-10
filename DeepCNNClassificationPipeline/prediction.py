import argparse
import importlib
import os
import warnings
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import BeltData
from model import Model
from utils import config_loader

warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def predict(model: Callable, 
            loader: Callable, 
            device
) -> None:
    model.eval()
    pred_values = []
    id_values = []
    with torch.no_grad():
        for minibatch in tqdm(loader):
            inputs, ids = minibatch
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Post processing. 
            if model.num_classes == 1: 
                preds = torch.ge(torch.sigmoid(outputs), 0.5).long()
            else: 
                _, preds = torch.max(outputs, dim=1)
                preds = preds.long()
            pred_values.extend(
                preds.cpu().numpy().flatten().tolist()
            )
            id_values.extend(
                ids
            )
    return pred_values, id_values


if __name__ == '__main__': 
    # Define input arguments.
    parser = argparse.ArgumentParser(description='Trainer Params.')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                    help='The string path of the config file.')
    args = parser.parse_args()
    configs = config_loader(args.config_path)

    # Define output directory. 
    output_directory = os.path.join(configs['prediction']['dir'], configs['experiment']) 
    os.makedirs(output_directory, exist_ok=True)

    # Define Transformations. 
    transformers = importlib.import_module(configs['transforms']).transforms

    # Define datasets. 
    for item in configs['metadata_paths']['predict']:
        print(f'  - {item}')
    dataset = BeltData(
            metadata_paths=configs['metadata_paths']['predict'], 
            root_dir=configs['root_dir']['predict'], 
            transform=transformers['predict'], 
            supervised=False
    )

    # Define data loaders. 
    loader = DataLoader(
            dataset=dataset, 
            batch_size=configs['batch_size'],
            num_workers=configs['num_workers'],
            shuffle=False, 
            collate_fn=BeltData.collate
    )
    
    # Define model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(
        model_name=configs['model']['model_name'], 
        pretrained=configs['model']['imagenet_pretrained'], 
        num_classes=configs['model']['num_classes'],
        device=device
    )
    if configs['model']['pretrained'] == True and configs['model']['pretrained_path'] != None: 
        model.load(path=configs['model']['pretrained_path'])
    else: 
        raise ValueError('There is no pretrained model.')
    
    pred_values, id_values = predict(model, loader, device)
    
    # Writing results.
    metadata = dataset.metadata
        
    image_paths = [
        str(metadata.loc[metadata.ImageID == item, 'Image'].tolist()[0])
        for item in id_values
    ]
    df = pd.DataFrame({
            'Image': image_paths,
            'IDs': id_values, 
            'PredictedLabel': pred_values
    })
    df.to_csv(
        os.path.join(output_directory, configs['prediction']['pred_file_name']), 
        index=False
    )



