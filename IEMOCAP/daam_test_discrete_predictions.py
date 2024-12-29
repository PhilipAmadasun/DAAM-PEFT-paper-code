import torch
import torch.nn as nn

class DensityAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis, num_heads, num_gaussians, padding_value, mean_offset_init=0, eps=1e-8):
        super().__init__()
        if not isinstance(norm_axis, int):
            raise ValueError("norm_axis must be an integer.")
        if num_heads <= 0 or not isinstance(num_heads, int):
            raise ValueError("num_heads must be a positive integer.")
        if num_gaussians <= 0 or not isinstance(num_gaussians, int):
            raise ValueError("num_gaussians must be a positive integer.")

        self.norm_axis = norm_axis
        self.eps = eps
        self.num_heads = num_heads
        self.padding_value = padding_value
        self.num_gaussians = num_gaussians

        self.mean_offsets = nn.Parameter(torch.zeros(num_gaussians, dtype=torch.float))
        self.c = nn.Parameter(torch.exp(torch.randn(num_gaussians, dtype=torch.float)))

    def forward(self, x, return_attention_details=False):
        if x.dim() < 2:
            raise ValueError(f"Input tensor must have at least 2 dimensions, got {x.dim()}.")
        if self.norm_axis >= x.dim() or self.norm_axis < -x.dim():
            raise ValueError(f"norm_axis {self.norm_axis} is out of bounds for input tensor with {x.dim()} dimensions.")

        mask = x != self.padding_value if self.padding_value is not None else None
        x_masked = torch.where(mask, x, torch.zeros_like(x)) if mask is not None else x

        mean = x_masked.mean(dim=self.norm_axis, keepdim=True)
        var = x_masked.var(dim=self.norm_axis, keepdim=True) + self.eps

        mixture = 1
        for i in range(self.num_gaussians):
            adjusted_mean = mean + self.mean_offsets[i]
            y_norm = (x - adjusted_mean) / torch.sqrt(var)
            gaussian = torch.exp(-((y_norm ** 2) / (2.0 * (self.c[i] ** 2)))) / torch.sqrt(2 * torch.pi * (self.c[i] ** 2))
            mixture *= gaussian

        mixture /= mixture.sum(dim=self.norm_axis, keepdim=True).clamp(min=self.eps)

        if return_attention_details:
            return torch.where(mask, x * mixture, x) if mask is not None else x * mixture, mixture.detach()
        else:
            return torch.where(mask, x * mixture, x) if mask is not None else x * mixture
            
            
class MultiHeadDensityAdaptiveAttention(nn.Module):
    def __init__(self, norm_axis, num_heads, num_gaussians, padding_value=None, eps=1e-8):
        super().__init__()
        self.norm_axis = norm_axis
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            DensityAdaptiveAttention(norm_axis, num_heads, num_gaussians, padding_value, eps)
            for _ in range(num_heads)
        ])

    def forward(self, x, return_attention_details=False):
        chunk_size = x.shape[self.norm_axis] // self.num_heads
        if chunk_size == 0:
            raise ValueError(f"Input tensor size along norm_axis ({self.norm_axis}) must be larger than the number of heads ({self.num_heads}).")

        outputs, attention_details_ = [], []
        for i in range(self.num_heads):
            start_index = i * chunk_size
            print(f"Processing head {i}")
            end_index = start_index + chunk_size if i < self.num_heads - 1 else x.shape[self.norm_axis]
            chunk = x.narrow(self.norm_axis, start_index, end_index - start_index)
            if return_attention_details:
                out, mixture = self.attention_heads[i](chunk, return_attention_details=True)
                outputs.append(out)
                attention_details_.append(mixture)
            else:
                outputs.append(self.attention_heads[i](chunk))

        if return_attention_details:
            print("Exiting MultiHeadDensityAdaptiveAttention with attention details")
            return torch.cat(outputs, dim=self.norm_axis), torch.cat(attention_details_, dim=self.norm_axis)
        else:
            print("Exiting MultiHeadDensityAdaptiveAttention without attention details")
            return torch.cat(outputs, dim=self.norm_axis)
            
            

class DensityBlock(nn.Module):
    def __init__(self, norm_axes, num_heads, num_gaussians, num_layers, padding_value=None, eps=1e-8):
        super().__init__()
        if len(norm_axes) != num_layers or len(num_heads) != num_layers or len(num_gaussians) != num_layers:
            raise ValueError("Lengths of norm_axes, num_heads, and num_gaussians must match num_layers.")

        self.layers = nn.ModuleList([
            MultiHeadDensityAdaptiveAttention(norm_axes[i], num_heads[i], num_gaussians[i], padding_value, eps)
            for i in range(num_layers)
        ])

    def forward(self, x, return_attention_details=False):
        attention_details_ = {}
        for idx, layer in enumerate(self.layers):
            if return_attention_details:
                x_, attention_details = layer(x, return_attention_details=True)
                attention_details_['layer_'+str(idx)] = attention_details
                x = x_ + x
            else:
                x = layer(x) + x

        if return_attention_details:
            return x, attention_details_
        return x


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch.nn import MultiheadAttention
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm  # Import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from transformers import AutoProcessor, AutoModel
import torchaudio
import torch.nn.init as init
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
from torch.distributions import Normal

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(1)  # You can choose any seed value

class CustomDataset(Dataset):
    def __init__(self, directory):
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pt')]
        self.label_mapping = {'sadness': 0, 'neutral': 1, 'happiness_excited': 2, 'angry': 3}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        features = data['features'].squeeze(0)
        label = self.label_mapping[data['label']]
        return features, label


# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.5, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        eps = 1e-6  # Small epsilon to avoid underflow

        # Add epsilon inside the exponentiation step
        F_loss = self.alpha * (1-pt + eps)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
        
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F



class DAAMModel(nn.Module):
    def __init__(self, num_classes, num_heads=8):
        super(DAAMModel, self).__init__()

        #self.daam = DensityBlock(norm_axes = -2, num_heads = 4, num_gaussians = 4, num_layers = 1,              #padding_value=None, eps=1e-8) 
        self.daam = DensityBlock(norm_axes=[-2], num_heads=[4], num_gaussians=[4], num_layers=1, padding_value=None, eps=1e-8)

        #You will be applying DAAM on the number of layers, which means we believe the most important information is on the number of layers,
        #You pass a wav file accross all the layers of WavLM (The tested SER model), you are aspplying accorss the layer dimension and you can normalize across different dimensions.
        #You can add another layer after DAAm

        self.conv1 = nn.Conv2d(in_channels=24, out_channels=512, kernel_size=(3,3), stride=1, padding=1) # when 1 layers --> 24
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(3,3), stride=1, padding=1)
        self.fc = nn.Linear(24 * 1024, num_classes)  # Fully connected layer

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x, save_attention_weights=False):
        # Apply DAAM
        attention_weights_ = None

        if not x.shape[0] == 1: 
            x = x.squeeze().mean(dim=2)
        else:
            x = x.squeeze()[None, :, :, :].mean(dim=2) #The dimensions will be bath_size,num_layers,num_features=1024


        if save_attention_weights:
            x, attention_weights_ = self.daam(x, save_attention_weights)
            print("DensityBlock with attention details used in DAAMModel")
        else:
            x = self.daam(x, save_attention_weights)
            print("DensityBlock without attention details used in DAAMModel")
        print(f"Output shape after DensityBlock in DAAMModel: {x.shape}")
        x = x.reshape(x.size(0), x.size(1), 32, 32)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        if save_attention_weights:
            return x, attention_weights_
        else:

            return x

    
def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

import os 
def load_checkpoint(filename="checkpoint.pth"):
    if os.path.isfile(filename):
        return torch.load(filename)
    return None

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, device, start_epoch=0, best_val_accuracy=0.0, fold_dir=''):
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    scaler = GradScaler()
    clip_value = 1000.0
    accumulation_steps = 1
    accumulation_counter = 0

    #model.to(device)

    for epoch in range(start_epoch, num_epochs):
        model.to(device)
        model.train()
        total_train_loss = 0.0
        total_train_samples = 0

        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Training]', ncols=100)
        optimizer.zero_grad()

        for batch_data, batch_labels in train_progress_bar:
            if torch.isnan(batch_data).any() or torch.isnan(batch_labels).any():
                print("NaN detected in input data or labels")
                continue

            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)

            with autocast():
                outputs = model(batch_data)
                if torch.isnan(outputs).any():
                    print("NaN detected in model outputs")
                    continue
                data_loss = criterion(outputs, batch_labels)
                
                total_loss = data_loss

                total_train_loss += data_loss.item() * batch_data.size(0)
                total_train_samples += batch_data.size(0)
                #del outputs, batch_data, batch_labels

            
            scaler.scale(total_loss).backward()
            if isinstance(model, nn.DataParallel):
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), clip_value)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Parameter update
            if (accumulation_counter + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                accumulation_counter = 0

            accumulation_counter += 1

            # Update progress bar
            train_progress_bar.set_postfix(loss=(total_train_loss / total_train_samples))

        torch.cuda.empty_cache()

        # Validation Phase
        if isinstance(model, nn.DataParallel):
            model = model.module  # Unwrap the model from DataParallel wrapper
        model.to("cpu").eval()
        correct = 0
        total = 0

        val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]', ncols=100)
        attention_weights_ = []
        with torch.no_grad():
            for batch_data, batch_labels in val_progress_bar:
                #batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                batch_data, batch_labels = batch_data.to("cpu"), batch_labels.to("cpu")
   
                # Save attention weights during validation
                outputs, attention_weights = model(batch_data, save_attention_weights=True)
                attention_weights_.append(attention_weights)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                del outputs, attention_weights, batch_data, batch_labels
                

        val_accuracy = correct / total
        # Step the scheduler with the current epoch's validation accuracy
        scheduler.step(val_accuracy)
        print(f'Validation Accuracy: {val_accuracy}')

        # Save the model if it's the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = model.state_dict()
            # Save a checkpoint in the fold directory
            checkpoint = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'attention_weights': attention_weights_ # each element of the list is a batch
            }
            
            checkpoint_path = os.path.join(fold_dir, 'checkpoint.pth')
            torch.save(checkpoint, checkpoint_path)

            with open(os.path.join(fold_dir,'AdaGCTlog.txt'), 'a') as f:

                string = 'Epoch: '+ str(epoch+1)+ ' best_val_accuracy: '+ str(best_val_accuracy)
                f.write(string)
                f.write("\n")

            f.close()
    

        torch.cuda.empty_cache()


    return None, None #best_model_state, best_val_accuracy


# Load data from CSV
#df = pd.read_csv('./emotion_data.csv')
# Load dataset
df = pd.read_csv('./emotion_data.csv')

# Number of subprocesses to use for data loading
num_workers = 4  # You can adjust this number based on your system's capabilities

# GroupKFold based on session
gkf = GroupKFold(n_splits=5)
df['fold'] = -1
for fold, (_, test_idx) in enumerate(gkf.split(df, groups=df['session'])):
    df.loc[test_idx, 'fold'] = fold

# Map labels to integers
label_mapping = {'sadness': 0, 'neutral': 1, 'happiness_excited': 2, 'angry': 3}  # Add all labels here

# Initialize the processor and WavLM / HuBERT model
pretrained_model = AutoModel.from_pretrained("microsoft/wavlm-large") #Wav2Vec2Model.from_pretrained("facebook/hubert-base-ls960")

parent_dir = 'DAAMv1'

if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

base_dir = './processed_features'
sessions = [f'Session{i}' for i in range(1, 6)]

# Prepare datasets for each session
datasets = {f'Session{i}': CustomDataset(os.path.join(base_dir, f'Session{i}')) for i in range(1, 6)}

# GroupKFold replaced by manual session-based division
for i, session in enumerate(sessions, start=1):
    # Define training and validation datasets based on session

    train_sessions = [datasets[s] for s in sessions if s != session]
    val_dataset = datasets[session]

    # Concatenate all training datasets except the current session
    train_dataset = torch.utils.data.ConcatDataset(train_sessions)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=num_workers)

    # Training and validation logic here
    print(f"Training on sessions excluding {session}, validating on {session}")

    # Initialize your model
    # params: 336,412
    model = DAAMModel(num_classes=len(label_mapping))

    # Use DataParallel if multiple GPUs are available
    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        print(f"Using {n_gpu} GPUs!")
        model = nn.DataParallel(model)
    elif n_gpu == 1:
        print(f"Using {n_gpu} GPU!")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=10, verbose=True)
    criterion = FocalLoss()

    fold_dir = "fold_"+str(i-1)
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)

    # Check for a saved checkpoint in the fold directory
    checkpoint_path = os.path.join(parent_dir + fold_dir, 'checkpoint.pth')
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        del checkpoint['attention_weights']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_val_accuracy = checkpoint['best_val_accuracy']

        # Ask user whether to continue training or not
        resume_training = input(f"Checkpoint for fold {fold_dir} found at epoch {start_epoch} and Validation Accuracy {best_val_accuracy}. Do you want to resume training? (yes/no): ").strip().lower()
        if resume_training == "yes":
            max_epochs = int(input("Enter the maximum number of epochs to train: "))
        else:
            continue
    else:
        start_epoch = 0
        best_val_accuracy = 0.0
        max_epochs = 35

    # Train and Validate the Model
    train_and_validate(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        num_epochs=max_epochs, device=device,
        start_epoch=start_epoch, best_val_accuracy=best_val_accuracy,
        fold_dir=fold_dir  # Pass the fold directory
    )
