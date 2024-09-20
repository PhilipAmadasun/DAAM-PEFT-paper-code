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


class MHAModel(nn.Module):
    def __init__(self, num_classes, embed_dim=1024, num_heads=8):
        super(MHAModel, self).__init__()

        self.mha = self.mha = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=(3,3), stride=1, padding=1)
        self.fc = nn.Linear(1 * 1024, num_classes)  # Fully connected layer

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
            x = x.squeeze()[None, :, :, :].mean(dim=2)

        x = x[:, 22, :]


        # Proceed with applying MultiheadGQA
        x
        x, attention_weights_ = self.mha(x,x,x)
        x = x[:, None, :]
        
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
        model.eval()
        correct = 0
        total = 0
        ccc = True
        val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]', ncols=100)
        attention_weights_ = []
        with torch.no_grad():
            for batch_data, batch_labels in val_progress_bar:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
   
                # Save attention weights during validation
      
                outputs, attention_weights = model(batch_data, save_attention_weights=True)
                if attention_weights is not None:
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

            f.close()
    

        torch.cuda.empty_cache()


    return None, None #best_model_state, best_val_accuracy


# Load data from CSV
df = pd.read_csv('/home/george/mauven/IEMOCAP/Attention-Info-Extract/emotion_data.csv')#pd.read_csv('./emotion_data.csv')

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

base_dir = "/home/george/mauven/IEMOCAP/Attention-Info-Extract/neurips/AdaGCT/processed_features" #'./processed_features'
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
    model = MHAModel(num_classes=len(label_mapping))

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