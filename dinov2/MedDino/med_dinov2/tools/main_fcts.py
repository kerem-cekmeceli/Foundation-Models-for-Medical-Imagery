from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import torch.nn.functional as F
# from model import Segmentor
from scipy.ndimage import zoom
import matplotlib.pyplot as plt


def train_all_batches(model, train_loader, loss_fn, optimizer, device):
    model.train()
    batches = tqdm(train_loader, desc='Train Batches', leave=False)
    running_loss = 0
    
    if len(batches)<=0:
        raise Exception('No data')
    
    for x_batch, y_batch in batches:
        x_batch = x_batch.to(device=device)
        y_batch = y_batch.to(device=device)
        
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return running_loss / len(batches)
        
def validate_all_batches(model, val_loader, loss_fn, device):
    model.eval()
    batches = tqdm(val_loader, desc='Eval Batches', leave=False)
    running_loss = 0
    
    if len(batches)<=0:
        raise Exception('No data')
    
    for x_batch, y_batch in batches:
        x_batch = x_batch.to(device=device)
        y_batch = y_batch.to(device=device)
        
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        running_loss += loss.item()
        
    return running_loss / len(batches)

# @TODO add metrics: dict:{metric_name:callable}

def train(model, train_loader, loss_fn, optimizer, n_epochs, device,
          scheduler=None, val_loader=None, print_epoch_info=True):
    epochs = tqdm(range(n_epochs), desc='Epochs')
    
    train_loss = []
    if val_loader is not None:
        val_loss = []
    
    for epoch in epochs:
        # Train
        train_loss_e = train_all_batches(model, train_loader, loss_fn, optimizer, device)
        train_loss.append(train_loss_e)
        epoch_str = f'Epoch:{epoch+1}/{n_epochs}, lr={optimizer.param_groups[0]["lr"]}\
                        , train_loss={train_loss_e}'
        
        # Update LR
        if scheduler is not None:
            scheduler.step()
        
        # Validate    
        if val_loader is not None:
            val_loss_e = validate_all_batches(model, val_loader, loss_fn, device)
            val_loss.append(val_loss_e)
            epoch_str += f', val_loss={val_loss_e}'
            
        if print_epoch_info:
            tqdm.write(epoch_str)
            
    if val_loader is not None:   
        return tuple(train_loss, val_loss)
    else:
        return tuple(train_loss)
    

###########################################################################

# def train_(model, train_loader, criterion, optimizer, epoch, device):
#     model.train()
#     loop = tqdm(train_loader, total=len(train_loader))
#     running_loss = 0
#     correct = 0

#     for batch_idx, (data, target) in enumerate(loop):
#         # print(batch_idx) 
#         data, target = data.to(device), target.to(device)

#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(output.data, 1)
#         loop.set_description(f"Epoch {epoch+1}")
#         loop.set_postfix(loss = loss.item())

#     print(f'\nTrain set: Average loss: {running_loss/len(train_loader):.4f}')

# def validation(model, criterion, valid_loader):
#     model.eval()
#     running_loss = 0
#     correct = 0

#     with torch.no_grad():
#         loop = tqdm(valid_loader, total=len(valid_loader))
#         for data, target in loop:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             loss = criterion(output, target)
#             running_loss += loss.item()
#             _, predicted = torch.max(output.data, 1)

#     print(f'\nValidation set: Average loss: {running_loss/len(valid_loader):.4f}')


def infer(image_path, model, device, img_transform):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    transformed_image = img_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make sure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Make prediction
        output = model(transformed_image)

        # Get the predicted class for each pixel
        _, predicted = torch.max(output, 1)
    
    # Move prediction to cpu and convert to numpy array
    predicted = predicted.squeeze().cpu().numpy()

    return transformed_image.cpu().squeeze().permute(1, 2, 0).numpy(), predicted