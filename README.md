```markdown
# Density Adaptive Attention

DensityAdaptiveAttention is a PyTorch library providing modules for applying density adaptive attention mechanisms that can approximate any Probability Distribution to derive attention weights. This library offers a novel approach to enhance neural network models with adaptive attention capabilities, inspired by density distribution principles.

## Features

- **Customizable Density Attention**: Tailor the attention mechanism to suit various neural network architectures.
- **Multiple Attention Heads**: Support for multiple attention heads for complex tasks.
- **PyTorch Integration**: Seamlessly integrates with existing PyTorch models.


## Requirements

- Python 3.x
- PyTorch (latest version recommended)

## Usage

Import the `DensityAdaptiveAttention` and `MultiHeadDensityAdaptiveAttention` modules and integrate them into your PyTorch models (found in the appendix folder):

```python
import torch
from density_adaptive_attention import DensityAdaptiveAttention, MultiHeadDensityAdaptiveAttention, DensityBlock
```

## Example Usage

This example demonstrates the use of the `DensityBlock` class, which encapsulates multiple layers of Density Adaptive Attention.

```python
import torch
import torch.nn as nn
from density_adaptive_attention import DensityBlock

norm_axes = [1, 1, 1]  # Axes for each layer in the DensityBlock.
num_heads = [4, 4, 4]  # Number of attention heads for each layer.
num_densities = [5, 5, 5]  # Number of densities per head for each layer.
num_layers = 3  # Total number of layers in the DensityBlock.
padding_value = None  # Padding value for sequences in the input tensor.
eps = 1e-8  # Small epsilon value for numerical stability.

# Initialize the DensityBlock
attention_block = DensityBlock(norm_axes, num_heads, num_densities, num_layers, padding_value, eps)

# Example neural network with DensityBlock
class ExampleNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExampleNetwork, self).__init__()
        # Initialize DensityBlock for attention mechanism
        self.attention_block = attention_block
        # Initialize a linear layer
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Apply DensityBlock for attention
        x = self.attention_block(x)
        # Apply linear layer
        x = self.linear(x)
        return x

# Example usage
input_dim = 128
output_dim = 128
model = ExampleNetwork(input_dim, output_dim)
input_tensor = torch.rand(10, input_dim)  # Example input tensor
output = model(input_tensor)
```

## Code Directories

The repository includes code for reproducing the experiments in our paper across multiple modalities:

- **AGNEWS**:
  - Contains code and scripts for text classification experiments on the AG News dataset using the Density Adaptive Attention Mechanism.
  - Demonstrates the application of DAAM in enhancing text classification performance by dynamically recalibrating feature significance.

- **CIFAR100**:
  - Contains code and scripts for image classification experiments on the CIFAR100 dataset using the Density Adaptive Attention Mechanism.
  - Showcases how DAAM improves image classification tasks through dynamic attention across visual features.

- **IEMOCAP**:
  - Contains code and scripts for speech emotion recognition experiments on the IEMOCAP dataset using the Density Adaptive Attention Mechanism.
  - Illustrates the effectiveness of DAAM in handling non-stationary speech data for emotion recognition.

- **LoRA-experiments/peft**:
  - Contains code for experiments comparing our method with Low-Rank Adaptation (LoRA) techniques for parameter-efficient fine-tuning.
  - Provides insights into the advantages of DAAM over existing parameter-efficient fine-tuning methods.

- **appendix**:
  - Contains supplementary code and experiments that are detailed in the appendix of our paper.
  - Includes additional ablation studies and alternative implementations of DAAM.


## Contributing

Contributions to the DensityAdaptiveAttention library are welcome!

- **Report Issues**: Submit an issue on GitHub if you find bugs or potential improvements.
- **Submit Pull Requests**: Feel free to fork the repository and submit pull requests with your enhancements.

## License

This project is licensed under Apache-2.0 - see the LICENSE file in the repository for details.
```
