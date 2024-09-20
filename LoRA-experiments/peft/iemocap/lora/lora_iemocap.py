import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
from transformers import WavLMForSequenceClassification, Trainer, TrainingArguments, AutoProcessor
from datasets import load_metric
import numpy as np
from sklearn.model_selection import KFold
from peft import get_peft_model, LoraConfig
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

label2idx = {'angry': 0, 'happiness_excited': 1, 'neutral': 2, 'sadness': 3}

class AudioDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

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
        return {"input_values": waveform.squeeze().numpy(), "label": label}

def compute_metrics(pred):
    metric = load_metric("accuracy")
    accuracy = metric.compute(predictions=np.argmax(pred.predictions, axis=1), references=pred.label_ids)
    return {"accuracy": accuracy}

def data_collator(batch):
    input_values = [item['input_values'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Find the maximum length in the batch
    max_length = max([len(x) for x in input_values])
    
    # Pad sequences to the maximum length
    padded_inputs = np.zeros((len(input_values), max_length))
    for i, inp in enumerate(input_values):
        padded_inputs[i, :len(inp)] = inp

    input_values = torch.tensor(padded_inputs, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    return {'input_values': input_values, 'labels': labels}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load processor and model
    model = WavLMForSequenceClassification.from_pretrained("microsoft/wavlm-large", num_labels=4).to(device)

    peft_config = LoraConfig(task_type="AUDIO_CLS", r=1, lora_alpha=16, lora_dropout=0.01, target_modules=['q_proj'])
    model = get_peft_model(model, peft_config)

    # Load dataset
    df = pd.read_csv('/home/george/mauven/IEMOCAP/Attention-Info-Extract/emotion_data.csv')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"Training fold {fold + 1}")

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_dataset = AudioDataset(train_df)
        val_dataset = AudioDataset(val_df)

        # Debug: print the size of the datasets
        print(f"Train dataset size: {len(train_dataset)}, Validation dataset size: {len(val_dataset)}")

        # Print a sample from the dataset to ensure it's correct
        print("Sample from train dataset:", train_dataset[0])

        train_args = TrainingArguments(
            output_dir=f'./results_fold_{fold + 1}',
            evaluation_strategy='epoch',
            save_strategy='epoch',
            learning_rate=1e-5,
            per_device_train_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir=f'./logs_fold_{fold + 1}',
            fp16=True
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        # Print a batch to debug
        train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator)
        for batch in train_dataloader:
            print("Batch from dataloader:", batch)
            break  # Only print the first batch

        trainer.train()
        trainer.save_model(f"./model_save_fold_{fold + 1}")
        results = trainer.evaluate()
        print(f"Test set accuracy for fold {fold + 1}: {results}")

if __name__ == '__main__':
    main()
