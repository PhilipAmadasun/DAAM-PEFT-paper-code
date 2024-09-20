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


def set_seed(seed_value=1):
    """Set seed for reproducibility."""
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(333)  # You can choose any seed value

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


def scaled_dot_product_gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    mask: Optional[Tensor] = None,
    is_causal: Optional[bool] = None,
    need_weights: bool = False,
    average_attn_weights: bool = False,
    force_grouped: bool = False,
):
    """Scaled dot product attention with support for grouped queries.

    Einstein notation:
    - b: batch size
    - n / s: sequence length
    - h: number of heads
    - g: number of groups
    - d: dimension of query/key/value

    Args:
        query: Query tensor of shape (b, n, h, d)
        key: Key tensor of shape (b, s, h, d)
        value: Value tensor of shape (b, s, h, d)
        dropout: Dropout probability (default: 0.0)
        scale: Scale factor for query (default: d_query ** 0.5)
        mask: Mask tensor of shape (b, n, s) or (b, s). If 'ndim == 2', the mask is
            applied to all 'n' rows of the attention matrix. (default: None)
        force_grouped: If True, apply grouped-query attention even if the number of
            heads is equal for query, key, and value. (default: False)

    Returns:
        2-tuple of:
        - Attention output with shape (b, n, h, d)
        - (Optional) Attention weights with shape (b, h, n, s). Only returned if
          'need_weights' is True.
    """
    if (mask is not None) and (is_causal is not None):
        raise ValueError(
            "Only one of 'mask' and 'is_causal' should be provided, but got both."
        )
    elif not query.ndim == key.ndim == value.ndim == 4:
        raise ValueError(
            f"Expected query, key, and value to be 4-dimensional, but got shapes "
            f"{query.shape}, {key.shape}, and {value.shape}."
        )

    # Move sequence length dimension to axis 2.
    # This makes the attention operations below *much* faster.
    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape
    bv, hv, nv, dv = value.shape
    if not (bq == bk == bv and dq == dk == dv):
        raise ValueError(
            "Expected query, key, and value to have the same batch size (dim=0) and "
            f"embedding dimension (dim=3), but got query: {query.shape}, "
            f"key: {key.shape}, and value: {value.shape}."
        )
    elif (hk != hv) or (nk != nv):
        raise ValueError(
            "Expected key and value to have the same size in dimensions 1 and 2, but "
            f"got key: {key.shape} and value: {value.shape}."
        )
    elif hq % hk != 0:
        raise ValueError(
            "Expected query heads to be a multiple of key/value heads, but got "
            f"query: {query.shape} and key/value: {key.shape}."
        )

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    if num_head_groups > 1 or force_grouped:
        # Separate the query heads into 'num_head_groups' chunks, and fold the group
        # dimension into the batch dimension.  This allows us to compute the attention
        # for each head in parallel, then sum over all of the groups at the end.
        query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
        similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
        #similarity = torch.matmul(query, key.transpose(-2, -1))
    else:
        # If the number of query/key heads is equal, we can skip grouping the queries,
        # and just use the standard sdot product attention.
        similarity = einsum(query, key, "b h n d, b h s d -> b h n s")
        #similarity = torch.matmul(query, key.transpose(-2, -1))

    if is_causal:
        # Mask out the upper triangular portion of the attention matrix. This prevents
        # the model from attending to tokens in the future.
        mask = torch.ones(
            (bq, nq, nk),
            device=query.device,
            dtype=torch.bool,
        ).tril_()

    if mask is not None:
        # Expand mask to match the shape of the attention matrix.
        # If mask is 2D, assume that it is applied to the key/value sequence dimension.
        # Else if mask is 3D, assume that it is applied to the query/key/value sequence
        # dimension for all attention heads.
        #
        # Users could also provide a 4D mask, which is applied to the query/key/value
        # sequence dimension for each attention head (though I don't have a particular
        # use case in mind for that).
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () n s")
        # Mask similarity values by setting them to negative infinity.  This guarantees
        # that they will not contribute to the softmax computation below.
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    # Apply attention matrix to the value Tensor.
    out = einsum(attention, value, "b h n s, b h s d -> b h n d")
    # Move head dimension back to axis 2
    out = rearrange(out, "b h n d -> b n h d")

    attn_weights: Optional[Tensor] = None
    if need_weights:
        # Move the sequence dimensions back to positions 1, 2.  Move the head dimension
        # to position 3.  This more closely matches the return shape of the attention
        # output: (b, n, h, d).
        attn_weights = rearrange(attention, "b h n s -> b n s h")
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

    return out, attn_weights


class MultiheadGQA(nn.Module):
    """Multi-head grouped query attention (GQA) layer.

    Reference:
        "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
        https://arxiv.org/pdf/2305.13245v1.pdf

    GQA is a variant of multihead attention (MHA) that uses fewer write heads
    (key / value) than query heads.  GQA can be viewed as a generalization of
    multi-query attention (MQA), which uses a single write head. GQA and MQA give
    significant speedups over standard MHA in decoder layers, with minimal loss in
    accuracy. In the paper, GQA is shown to be more accurate than MQA, while still
    having a significant speedup over MHA.

    NOTE: The original authors only benchmark GQA by adapting the T5 (XL or XXL) model
    from MHA to GQA.  As a result, they do not mention parameter initialization or
    layer normalization strategies.  I follow the best practices laid out in the
    MAGNETO paper, which improves Transformer performance through better parameter
    initialization and layer norm placement.  See:
        https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
    """

    def __init__(
        self,
        embed_dim: int,
        query_heads: int,
        kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        layer_norm: bool = True,
        layer_norm_eps: float = 1e-5,
        gamma_init: float = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.query_heads = query_heads
        self.kv_heads = kv_heads
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.gamma_init = gamma_init

        if self.query_heads % self.kv_heads != 0:
            raise ValueError(
                f"query_heads ({query_heads}) must be divisible by "
                f"kv_heads ({kv_heads})"
            )
        elif (embed_dim % self.query_heads != 0) or (embed_dim % self.kv_heads != 0):
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by "
                f"query_heads ({query_heads}) and kv_heads ({kv_heads})"
            )

        head_dim = embed_dim // query_heads
        if not head_dim % 8 == 0:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be divisible by 8"
            )
        if not head_dim <= 128:
            raise ValueError(
                f"head_dim (embed_dim / num_heads = {head_dim}) must be <= 128"
            )

        # Query projection layer is the same as in vanilla MHA.
        self.q_proj = nn.Linear(
            embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )
        # Key/value projection layers have a smaller output dimension, so that
        # the we have fewer key/value attention heads after reshaping.
        kv_embed_dim = embed_dim // query_heads * kv_heads
        self.k_proj = nn.Linear(
            embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.v_proj = nn.Linear(
            embed_dim, kv_embed_dim, bias=bias, device=device, dtype=dtype
        )
        self.norm: Optional[nn.LayerNorm] = None
        if layer_norm:
            self.norm = nn.LayerNorm(
                kv_embed_dim, eps=layer_norm_eps, device=device, dtype=dtype
            )
        # Grouped attention output will have the same embedding dimension as the
        # key/value Tensors.  So the output projection layer needs to accept the
        # same dimension (kv_embed_dim).
        self.out_proj = nn.Linear(
            kv_embed_dim, embed_dim, bias=bias, device=device, dtype=dtype
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0)
        nn.init.xavier_normal_(self.k_proj.weight)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0)

        # NOTE: We follow the initialization strategy from MAGNETO.  See:
        # https://arxiv.org/pdf/2210.06423.pdf, Fig. 2
        # Gain (self.gamma_init) should be provided as a keyword argument when
        # initializing the larger Transformer model, since it requires knowledge
        # of the number of encoder/decoder layers in the model.

        nn.init.xavier_normal_(self.v_proj.weight, gain=self.gamma_init)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0)
        nn.init.xavier_normal_(self.out_proj.weight, gain=self.gamma_init)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        need_weights: bool = False,
        # TODO
        # attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        average_attn_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Notation:
        #   b - batch size
        #   n - sequence length
        #   h - number of heads
        #   d - embedding dimension
        #
        # Input shape: (b, n, d)
        q: Tensor = self.q_proj(query)
        k: Tensor = self.k_proj(key)
        v: Tensor = self.v_proj(value)

        # Unfold 'd' dimension into 'h' separate attention heads.
        q = rearrange(q, "b n (h d) -> b n h d", h=self.query_heads)
        k = rearrange(k, "b n (h d) -> b n h d", h=self.kv_heads)
        v = rearrange(v, "b n (h d) -> b n h d", h=self.kv_heads)
        # Apply attention, then fold 'h' attention heads back into 'd'.
        x, attn = scaled_dot_product_gqa(
            query=q,
            key=k,
            value=v,
            # TODO
            # mask=attn_mask,
            is_causal=is_causal,
            need_weights=need_weights,
            average_attn_weights=average_attn_weights,
            force_grouped=False,
        )
        x = rearrange(x, "b n h d -> b n (h d)")

        # NOTE: This is different from 'nn.MultiheadAttention'!  We follow the MAGNETO
        # architecture (https://arxiv.org/pdf/2210.06423.pdf), which applies an extra
        # layer norm before the linear output projection.  The cross-attention layer in
        # the MAGNETO decoder does not include this layer norm, so users have the
        # option to disable it (layer_norm=False).
        if self.layer_norm:
            assert self.norm is not None
            x = self.norm(x)
        # Linear projection on attention outputs.
        x = self.out_proj(x)

        return x, attn


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

        self.gcts = nn.ModuleList([DAAM(feature_size, self.head_dim, time_steps, c, eps) for _ in range(num_heads)])

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
            head_output = self.gcts[i](x[:, i, :, :, :], save_attention_weights).mean(dim=2)
            # 4, 3, 1024
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
    def __init__(self, num_classes, num_heads=1):
        super(DAAMModel, self).__init__()

        self.daam = MultiHeadDAAM(num_layers=24, time_steps=1, feature_size=1024, num_heads=num_heads)
        self.gqa = MultiheadGQA(embed_dim=512, query_heads=8, kv_heads=2, device="cuda", dtype=torch.float32)
        self.embed_transform = nn.Linear(1024, 512)


        self.conv1 = nn.Conv2d(in_channels=24, out_channels=512, kernel_size=(3,3), stride=1, padding=1)
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

        reshaped_tensor = self.embed_transform(x)

        # Now, use reshaped_tensor as query, key, and value for MultiheadGQA
        # Assuming your model is using float32
        # Ensure the tensors passed to mha are of the same dtype
        query = reshaped_tensor.float()  # 
        key = reshaped_tensor.float()    # 
        value = reshaped_tensor.float()  # 


        # Proceed with applying MultiheadGQA
        x, attn_weights = self.gqa(query, key, value, need_weights=True)
        

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
train_pickle_path = '../../train_cifar100.pkl'  # Update with the actual path
val_pickle_path = '../../val_cifar100.pkl'      # Update with the actual path

# Load datasets
train_dataset = CIFAR100Dataset(train_pickle_path)
val_dataset = CIFAR100Dataset(val_pickle_path)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #32
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) #32

# Initialize your model
num_classes = 100  # CIFAR100 has 100 classes
# params: 1,208,856
model = DAAMModel(num_classes=num_classes)  # Adjust the number of classes as needed

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