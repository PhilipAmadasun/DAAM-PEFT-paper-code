import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import LlamaTokenizer, LlamaForSequenceClassification, DataCollatorWithPadding
from datasets import load_dataset
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Check if CUDA is available and set the default device accordingly
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the AG News dataset for news categorization
dataset = load_dataset("ag_news")

# Load model and tokenizer
model_checkpoint = "TheBloke/Llama-2-13B-fp16"
tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint)

# Check if the tokenizer has a pad token, if not, set it to the EOS token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = LlamaForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=4, id2label={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
)

# Use DataParallel to utilize multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to(device)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Tokenize the datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Define the LoRALayer class
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=16):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_features, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        self.alpha = alpha

    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)

# Function to add LoRA layers to the model
def add_lora_layers(model, target_modules, rank=4, alpha=16):
    lora_params_A = []
    lora_params_B = []
    for name, module in model.named_modules():
        if any(target_module in name for target_module in target_modules):
            if isinstance(module, nn.Linear):
                in_features, out_features = module.in_features, module.out_features
                lora_layer = LoRALayer(in_features, out_features, rank, alpha).to(module.weight.device)
                setattr(module, 'lora', lora_layer)
                lora_params_A.append(lora_layer.A)
                lora_params_B.append(lora_layer.B)

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
    
    return model, lora_params_A, lora_params_B

# Add LoRA layers to the model
target_modules = ['q_proj', 'v_proj']
model, lora_params_A, lora_params_B = add_lora_layers(model.module if isinstance(model, nn.DataParallel) else model, target_modules, rank=8, alpha=16)

# Print the number of trainable parameters
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_params = count_trainable_parameters(model)
print(f"Number of trainable parameters: {num_params}")

# Define training and evaluation functions
def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_values = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(input_ids=input_values, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        # Print loss for each batch
        pbar.set_description(f"Training - Batch Loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    pbar = tqdm(dataloader, desc="Evaluating")
    with torch.no_grad():
        for batch in pbar:
            input_values = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_values, attention_mask=attention_mask)
            
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()
            # Print loss for each batch
            pbar.set_description(f"Evaluating - Batch Loss: {loss.item():.4f}")

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def main():
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=data_collator)  # Reduced batch size
    val_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=data_collator)  # Reduced batch size

    criterion = nn.CrossEntropyLoss()
    eta = 1e-6
    lambda_ = 2.828  # Example value for lambda, you can set it to any value > 1
    optimizer = torch.optim.AdamW([
        {'params': lora_params_A, 'lr': eta},
        {'params': lora_params_B, 'lr': lambda_ * eta}
    ])
    scaler = GradScaler()

    # Print number of trainable parameters just before training starts
    number = count_trainable_parameters(model)
    print(f"Number of trainable parameters before training: {number}")

    num_epochs = 1
    best_acc = 0.0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, scaler)
        val_accuracy = evaluate(model, val_dataloader, device, criterion)
        if val_accuracy >= best_acc:
            best_acc = val_accuracy
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save the model and tokenizer
    with open("run3_loraplus_r8alpha16.txt", 'a') as f:
        string = f"Best validation accuracy rate: {best_acc}, Number of trainable parameters: {number}"
        f.write(string + "\n")

    model_save_path = "./model_saver8alpha16"
    tokenizer_save_path = "./tokenizer_saver8alpha16"
    model.module.save_pretrained(model_save_path) if isinstance(model, nn.DataParallel) else model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)
    print(f"Model and tokenizer have been saved to {model_save_path} and {tokenizer_save_path} respectively.")

if __name__ == "__main__":
    main()