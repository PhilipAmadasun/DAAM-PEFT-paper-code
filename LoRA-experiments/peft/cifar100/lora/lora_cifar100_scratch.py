import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from transformers import BeitFeatureExtractor, BeitForImageClassification
from tqdm import tqdm

class Cifar100Dataset(Dataset):
    def __init__(self, data_type='train'):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.dataset = datasets.CIFAR100(root='./data', train=(data_type == 'train'), download=True, transform=transform)
        self.feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        pil_image = transforms.ToPILImage()(image)
        inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
        return {
            "pixel_values": inputs['pixel_values'].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

def custom_data_collator(features):
    pixel_values = torch.stack([item["pixel_values"] for item in features])
    labels = torch.tensor([item["labels"] for item in features])
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_features, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)

def add_lora_layers(model, target_modules, rank=4, alpha=16):
    lora_params = []
    for name, module in model.named_modules():
        if any(target_module in name for target_module in target_modules):
            if isinstance(module, nn.Linear):
                in_features, out_features = module.in_features, module.out_features
                lora_layer = LoRALayer(in_features, out_features, rank, alpha).to(module.weight.device)
                setattr(module, 'lora', lora_layer)
                lora_params.extend(lora_layer.parameters())

    def forward_hook(module, input, output):
        return output + module.lora(input[0])
    
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            module.register_forward_hook(forward_hook)
    
    for name, param in model.named_parameters():
        if 'classifier' in name or 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    return model, lora_params

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=input_values)
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
            input_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=input_values)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_dataset = Cifar100Dataset(data_type='train')
    val_dataset = Cifar100Dataset(data_type='val')

    model = BeitForImageClassification.from_pretrained("microsoft/beit-large-patch16-224", num_labels=100, ignore_mismatched_sizes=True)
    model.to(device)

    # Print all the parameters of the model to choose names
    #for name, param in model.named_parameters():
    #    print(name)
    #input()

    target_modules = ['query', 'value']
    model, lora_params = add_lora_layers(model, target_modules, rank=8, alpha=16)

    # Print number of trainable parameters just before training starts
    number = count_trainable_parameters(model)
    print(f"Number of trainable parameters before training: {number}")


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_data_collator)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=custom_data_collator)

    num_epochs = 35
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device)
        val_accuracy = evaluate(model, val_dataloader, device)
        if val_accuracy >= best_acc:
            best_acc = val_accuracy
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        number = count_trainable_parameters(model)
        
    with open("run5_final_results_r8alpha16.txt", 'a') as f:
        string = f"Best validation accuracy rate: {best_acc}, Number of trainable parameters: {number}"
        f.write(string + "\n")

    torch.save(model.state_dict(), "run5_r4alpha16model_final.pt")

if __name__ == "__main__":
    main()
