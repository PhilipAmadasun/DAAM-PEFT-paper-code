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
import torchaudio
import torch.nn.init as init
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random, pickle
from torch.distributions import Normal
from torch.utils.data import DataLoader

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

set_seed(333)  # You can choose any seed value

import torch
import torch.nn as nn
import torch.nn.functional as F

fold_dir = 'MHA'

class MHAModel(nn.Module):
    def __init__(self, num_classes, num_heads=8):
        super(MHAModel, self).__init__()

        self.mha = MultiheadAttention(embed_dim=1024, num_heads=num_heads)

        self.conv1 = nn.Conv2d(in_channels=24, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=24, kernel_size=(3,3), stride=1, padding=1)

        self.fc = nn.Linear(24 * 1024, num_classes)  # Fully connected layer

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x, save_attention_weights=False):
        attention_weights_ = None

        if x.shape[0] == 1:
            x = x.squeeze().unsqueeze(0)
        else:
            x = x.squeeze()


        # Proceed with applying MultiheadGQA
        out, attention_weights_ = self.mha(x,x,x)

        x = out

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

            with open(os.path.join(fold_dir,'AdaGCTlog.txt'), 'a') as f:

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
train_pickle_path = '../train_cifar100.pkl'  # Update with the actual path
val_pickle_path = '../val_cifar100.pkl'      # Update with the actual path

# Load datasets
train_dataset = CIFAR100Dataset(train_pickle_path)
val_dataset = CIFAR100Dataset(val_pickle_path)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #32
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) #32

# Initialize your model
num_classes = 100  # CIFAR100 has 100 classes
# params: 6,877,820
model = MHAModel(num_classes=num_classes)  # Adjust the number of classes as needed

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