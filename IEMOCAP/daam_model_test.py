import os
import torch
import torchaudio
from transformers import AutoModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import pandas as pd
from sklearn.model_selection import GroupKFold
from tqdm import tqdm  # Import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor, AutoProcessor, AutoModel
import torch.nn.init as init
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
from torch.distributions import Normal
import torch.nn as nn

# Define constants
AUDIO_FOLDER_PATH = './test_wav_files'  # Directory for your audio files
BATCH_SIZE = 8  # Batch size for inference

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

# Step 1: Load WavLM and DAAM models
class DAAMModelWrapper(torch.nn.Module):
    def __init__(self, num_classes=4, checkpoint_path='path_to_daam_checkpoint'):
        super(DAAMModelWrapper, self).__init__()
        self.daam_model = DAAMModel(num_classes)
        
        # Load the trained DAAM model checkpoint
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            
            # Check if the checkpoint has a 'state_dict' key
            if "state_dict" in checkpoint:
                self.daam_model.load_state_dict(checkpoint["state_dict"])
                print(f"Loaded DAAM model state_dict from checkpoint at {checkpoint_path}")
            else:
                # If it's just model weights, load directly
                self.daam_model.load_state_dict(checkpoint)
                print(f"Loaded DAAM model weights directly from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    def forward(self, features):
        return self.daam_model(features)

# Step 2: Feature extraction and DAAM inference
def process_audio_and_infer_emotion(audio_folder_path, model_name="microsoft/wavlm-large", target_length=80000, batch_size=8):
    # Load WavLM model for feature extraction
    wavlm_model = AutoModel.from_pretrained(model_name)
    wavlm_model.eval()

    # Load DAAM model
    daam_model = DAAMModelWrapper(checkpoint_path='./daam_test_checkpoints/fold_0/checkpoint.pth')
    daam_model.eval()

    # Process each audio file in batches
    audio_files = [f for f in os.listdir(audio_folder_path) if f.endswith('.wav')]
    results = []
    label_mapping = {0: 'sadness', 1: 'neutral', 2: 'happiness_excited', 3: 'angry'}

    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i + batch_size]
        features_batch = []

        # Step 3: Extract features for the current batch
        for audio_file in batch_files:
            audio_path = os.path.join(audio_folder_path, audio_file)
            waveform, _ = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(axis=0, keepdim=True)
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(1)), mode="constant")

            with torch.no_grad():
                outputs = wavlm_model(waveform.squeeze()[None, :], output_hidden_states=True)
            hidden_states = torch.stack(outputs.hidden_states, dim=0)[1:]  # Adjust based on needs

            features_batch.append(hidden_states)

        # Convert the list of features to a tensor
        features_batch = torch.stack(features_batch)

        # Step 4: Perform emotion recognition with DAAM model
        with torch.no_grad():
            output = daam_model(features_batch)
            probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
            predictions = torch.argmax(probabilities, dim=1)  # Get the index of the highest probability

            # Map the predicted index to emotion labels and add to results
            results.extend([label_mapping[pred.item()] for pred in predictions])

    # Print or process results
    for idx, (audio_file, result) in enumerate(zip(audio_files, results)):
        print(f"Emotion recognition result for {audio_file}: {result}")

# Run the pipeline
if __name__ == '__main__':
    process_audio_and_infer_emotion(AUDIO_FOLDER_PATH, batch_size=BATCH_SIZE)
