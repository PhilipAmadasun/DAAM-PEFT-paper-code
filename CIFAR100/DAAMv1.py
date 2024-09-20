import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch.nn import MultiheadAttention
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm  # Import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModel
import torch.nn.init as init
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random, pickle
from torch.distributions import Normal
from torch.utils.data import DataLoader

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def set_seed(seed_value=1):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # You can choose any seed value

class DAAM(nn.Module):
    def __init__(self, feature_size, num_layers, time_steps, c=2, eps=1e-5):
        super().__init__()
        
        self.downsample = nn.AdaptiveAvgPool1d(1)
        self.eps = eps
        self.c = nn.Parameter(torch.full((1, feature_size), c, dtype=torch.float))
        # Learnable offset for the mean
        self.mean_offset = nn.Parameter(torch.zeros((1, feature_size), dtype=torch.float))
        self.attention_weights = {
            'var': [],
            'mean': [],
            'c': [],
            'attention_maps': []
        }

    def forward(self, x, save_attention_weights=False):
        batch_size, num_layers, time_steps, feature_size = x.size()
        # Reshape x while keeping the feature_size as the channel dimension
        x_reshaped = x.view(batch_size, num_layers * feature_size, time_steps)

         # Apply the downsampling convolutions
        y = self.downsample(x_reshaped)

        # Reshape back to the original format
        y = y.view(batch_size, num_layers, feature_size)

        # Calculate mean and variance
        mean = y.mean(dim=1, keepdim=True)
        mean_x2 = (y ** 2).mean(dim=1, keepdim=True)
        var = mean_x2 - mean ** 2

        var = abs(var) + 1e-8

        # Adjust mean with the learnable offset
        adjusted_mean = mean + self.mean_offset

        # Gaussian normalization
        y_norm = (y - adjusted_mean) / torch.sqrt(var + self.eps)
        self.y_transform = torch.exp(-(y_norm ** 2 / (2 * self.c)))

        # Add the time steps dimension to y_transform
        self.y_transform = self.y_transform.unsqueeze(2)  # New shape: [batch_size, num_layers, 1, feature_size]

        # Now expand y_transform to match the time steps dimension of x
        self.y_transform = self.y_transform.expand(batch_size, num_layers, time_steps, feature_size)

        # Append current attention weights to the history
        if save_attention_weights:
            #self.attention_weights['var'].append(var.detach().squeeze().mean(dim=0).mean(dim=0).item())
            #self.attention_weights['mean'].append(self.mean.detach().squeeze().mean(dim=0).mean(dim=0).item())
            self.attention_weights['attention_maps'].append(self.y_transform.mean(dim=2).mean(dim=0).detach())

        return x * self.y_transform


class MultiHeadDAAM(nn.Module):
    def __init__(self, num_layers, time_steps, feature_size, num_heads=8, c=2, eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = num_layers // num_heads
        assert self.head_dim * num_heads == num_layers, "num_layers must be divisible by num_heads"

        self.daams = nn.ModuleList([DAAM(feature_size, self.head_dim, time_steps, c, eps) for _ in range(num_heads)])

    def forward(self, x, save_attention_weights=False):
        # Handle different tensor dimensions

        if x.shape[0] == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()
        
        batch_size, num_layers, time_steps, feature_size = x[:,:,None,:].size()
    

        # Reshape x for multihead processing
        x = x.view(batch_size, self.num_heads, self.head_dim, time_steps, feature_size)

        outputs, attention_weights_ = [], []
        for i in range(self.num_heads):
            head_output = self.daams[i](x[:, i, :, :, :], save_attention_weights).mean(dim=2)
            outputs.append(head_output)

            if save_attention_weights:

                attention_weights_.append(self.gcts[i].attention_weights)


        output = torch.cat(outputs, dim=1)
        output = output.view(batch_size, num_layers, feature_size)

        if save_attention_weights:

            return output, attention_weights_
        
        else:

            return output



import torch
import torch.nn as nn
import torch.nn.functional as F

fold_dir = 'DAAMModel'

class DAAMModel(nn.Module):
    def __init__(self, num_classes, num_heads=8):
        super(DAAMModel, self).__init__()

        self.daam = MultiHeadDAAM(num_layers=24, time_steps=1, feature_size=1024, num_heads=num_heads)

        self.conv1 = nn.Conv2d(in_channels=24, out_channels=512, kernel_size=(3,3), stride=1, padding=1) # --> if 1 layer: 24 -> 24
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(3,3), stride=1, padding=1)

        self.fc = nn.Linear(24 * 1024, num_classes)  # Fully connected layer

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x, save_attention_weights=False):
        # Apply DAAM
        attention_weights_ = None

        if save_attention_weights:

            x = self.daam(x)
        
        else:

            x = self.daam(x)
        

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

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)

import os 
def load_checkpoint(filename="checkpoint.pth"):
    if os.path.isfile(filename):
        return torch.load(filename)
    return None

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, device, start_epoch=0, best_val_accuracy=0.0, fold_dir=''):
    #torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection
    scaler = GradScaler()
    clip_value = 1000.0
    accumulation_steps = 1
    accumulation_counter = 0

    for epoch in range(start_epoch, num_epochs):
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
        model.eval()
        correct = 0
        total = 0

        val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]', ncols=100)
        attention_weights_ = []
        with torch.no_grad():
            for batch_data, batch_labels in val_progress_bar:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                # Save attention weights during validation
                outputs, attention_weights = model(batch_data, save_attention_weights=True)
                attention_weights_.append(attention_weights)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

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

            with open(os.path.join(fold_dir,'DAAMlog.txt'), 'a') as f:

                string = 'Epoch: '+ str(epoch+1)+ ' best_val_accuracy: '+ str(best_val_accuracy)
                f.write(string)

            f.close()

        torch.cuda.empty_cache()


    return best_model_state, best_val_accuracy


# Custom function to load dataset from pickle file
def load_dataset_from_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
class CIFAR100Dataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as file:
            self.data = pickle.load(file)

    def __getitem__(self, idx):
        # Assuming the data is saved as a tuple (features, label)
        features, label = self.data[idx]
        return features, label

    def __len__(self):
        return len(self.data)



# Define the path to the pickle files
train_pickle_path = './train_cifar100.pkl'  # Update with the actual path
val_pickle_path = './val_cifar100.pkl'      # Update with the actual path

# Load datasets
train_dataset = CIFAR100Dataset(train_pickle_path)
val_dataset = CIFAR100Dataset(val_pickle_path)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #32
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) #32

# Initialize your model
num_classes = 100  # CIFAR100 has 100 classes
# params: 238,104
model = DAAMModel(num_classes=num_classes)  # Adjust the number of classes as needed

print("Number of params:", count_trainable_parameters(model))
input()

# Use DataParallel if multiple GPUs are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    print(f"Using {n_gpu} GPUs!")
    model = nn.DataParallel(model)
elif n_gpu == 1:
    print(f"Using {n_gpu} GPU!")

model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=10, verbose=True)
criterion = nn.CrossEntropyLoss()


# Training loop
num_epochs = 35  # Set the number of epochs
best_model_state, best_val_accuracy = train_and_validate(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs, device)