import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from transformers import WavLMForSequenceClassification, AutoFeatureExtractor
import numpy as np
from sklearn.model_selection import GroupKFold
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn import functional as F
import random
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignores specific categories of warnings

# To ignore all warnings:
warnings.filterwarnings("ignore")
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This will block all messages except fatal errors (3).

tf.get_logger().setLevel('ERROR')  # This will allow ERROR messages to show, but suppress INFO and WARNING messages.

def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # You can choose any seed value

# Define label mapping
label2idx = {'sadness': 0, 'neutral': 1, 'happiness_excited': 2, 'angry': 3}

class AudioDataset(Dataset):
    def __init__(self, dataframe, feature_extractor):
        self.dataframe = dataframe
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        audio_path = '/home/george/IEMOCAP/' + row['path'][2:]
        label = label2idx[row['emotion']]
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        inputs = self.feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
        return {
            "input_values": inputs.input_values.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "label": torch.tensor(label)
        }
    
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

def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True)
    
    return {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'labels': labels
    }

class LoRALayerPlus(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        
        # Initialize A with random normal scaled by std_dev
        self.A = nn.Parameter(torch.randn(in_features, rank) * std_dev)
        
        # Initialize B with zeros
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        
        self.alpha = alpha

    def forward(self, x):
        # Apply the transformation with alpha scaling after the matrix multiplication
        x = self.alpha * (x @ self.A @ self.B)
        return x

def add_lora_layers(model, target_modules, rank=4, alpha=16, lambda_=2.828):
    lora_params_A = []
    lora_params_B = []
    for name, module in model.named_modules():
        if any(target_module in name for target_module in target_modules):
            if isinstance(module, nn.Linear):
                in_features, out_features = module.in_features, module.out_features
                lora_layer = LoRALayerPlus(in_features, out_features, rank, alpha).to(module.weight.device)
                setattr(module, 'lora', lora_layer)
                lora_params_A.append(lora_layer.A)
                lora_params_B.append(lora_layer.B)

    def forward_hook(module, input, output):
        return output + module.lora(input[0])
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            module.register_forward_hook(forward_hook)
    
    # Freeze all parameters except LoRA parameters and output classifier
    for name, param in model.named_parameters():
        if 'classifier' in name or 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    return model, lora_params_A, lora_params_B

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_values=input_values, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_values = batch['input_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_values=input_values, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load dataset
    df = pd.read_csv('/home/george/mauven/IEMOCAP/Attention-Info-Extract/emotion_data.csv')
    
    # GroupKFold based on session
    gkf = GroupKFold(n_splits=5)
    df['fold'] = -1
    for fold, (_, test_idx) in enumerate(gkf.split(df, groups=df['session'])):
        df.loc[test_idx, 'fold'] = fold

    num_workers = 4  # You can adjust this number based on your system's capabilities

    focal_loss = FocalLoss(alpha=0.25, gamma=2.5, reduction='mean').to(device)

    for fold in range(5):

        if fold <= 1:
            continue

        # Load feature extractor and model
        feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
        model = WavLMForSequenceClassification.from_pretrained("microsoft/wavlm-large", num_labels=4)
        
        # Specify target modules for LoRA+
        target_modules = ['q_proj', 'v_proj']
        model, lora_params_A, lora_params_B = add_lora_layers(model, target_modules, rank=4, alpha=16, lambda_=2.828)
        model = model.to(device)

        print(f"Training fold {fold + 1}")

        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]

        train_dataset = AudioDataset(train_df, feature_extractor)
        val_dataset = AudioDataset(val_df, feature_extractor)

        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=num_workers)

        eta = 1e-4
        lambda_ = 2.828
        optimizer = AdamW([
            {'params': lora_params_A, 'lr': eta},
            {'params': lora_params_B, 'lr': lambda_ * eta}
        ])

        # Training loop
        num_epochs = 35
        best_acc = 0.0
        for epoch in range(num_epochs):
            train_loss = train_epoch(model, train_dataloader, optimizer, focal_loss, device)
            val_accuracy = evaluate(model, val_dataloader, device)
            if val_accuracy >= best_acc:
                best_acc = val_accuracy
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

            number = count_trainable_parameters(model)
        
        with open("final_results_loraplus_r4alpha16.txt", 'a') as f:
            string = f"Fold {fold + 1}, Best validation accuracy rate: {best_acc}, Number of trainable parameters: {number}"
            f.write(string + "\n")

        # Save the model
        torch.save(model.state_dict(), f"model_fold_{fold+1}.pt")

        del model

if __name__ == '__main__':
    main()
