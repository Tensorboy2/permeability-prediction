'''
main.py

The program runs training of PyTorch models defined in in the "configs" folder.
'''
import torch
import torch.nn as nn
torch.manual_seed(0)
import torch.optim as optim
import time
import os
path = os.path.dirname(__file__)


from train import train
from models.resnet import ResNet50, ResNet101
from models.convnext import ConvNeXtTiny, ConvNeXtSmall
from models.vit import ViT_B16, ViT_S16, ViT_T16, ViT_T8, ViT_S8
from data_loader import get_data

# Available models:
model_registry = {
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ConvNeXtTiny": ConvNeXtTiny,
    "ConvNeXtSmall": ConvNeXtSmall,
    "ViT_B16": ViT_B16,
    "ViT_S16": ViT_S16,
    "ViT_T16": ViT_T16,
    "ViT_T8": ViT_T8,
    "ViT_S8": ViT_S8,
}

def get_model_size(model):
    '''For logging the memory usage of the model.'''
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def main(model, hyperparameters, data, save_path="metrics.csv", save_model_path=None):
    '''
    Main function of Project 2. 

    Executes model instance, data loading, training and saving of metrics.
    '''
    num_epochs = hyperparameters["num_epochs"]
    lr = hyperparameters["lr"]
    warmup_steps = hyperparameters["warmup_steps"]
    weight_decay = hyperparameters["weight_decay"]
    batch_size = hyperparameters["batch_size"]
    decay = hyperparameters["decay"]

    torch.cuda.empty_cache() # Make available space.
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Check if gpu is available.

    print(f"Device: {device}")
    optimizer = optim.AdamW(params = model.parameters(),
                             lr = lr, 
                             weight_decay=weight_decay) # AdamW for decoupled weight decay.
    model.to(device)
    print(model.name) # Print model to log current model in training.
    train_data_loader, val_data_loader = get_data(batch_size=batch_size,
                                                   val_size=data["val_size"],
                                                   rotate=data["rotate"],
                                                   hflip=data["hflip"],
                                                   vflip=data["vflip"],
                                                   group=data["group"],
                                                   num_samples=data["num_samples"],
                                                   num_workers=data.get("num_workers", 2))
    start = time.time()
    train(model,
            optimizer,
            train_data_loader,
            val_data_loader, 
            num_epochs=num_epochs,
            warmup_steps=warmup_steps,
            decay=decay,
            save_path=save_path,
            save_model_path=save_model_path,
            clip_grad=hyperparameters.get("clip_grad", False),
            device=device)
    stop = time.time()
    print(f'Total training time: {stop-start} seconds')


import yaml
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load a YAML config file.")
    parser.add_argument("config_file", type=str, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    for exp in config["experiments"]:
        model_name = exp["model"]
        model_class = model_registry[model_name]
        model = model_class()
        print(f"Model size: {get_model_size(model):.2f} MB")

        main(model, exp["hyperparameters"], exp["data"], save_path=exp["save_path"], save_model_path=exp.get("save_model_path", None))

        # Memory management for gpu:
        del model
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
