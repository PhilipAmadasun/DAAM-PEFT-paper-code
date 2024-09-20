import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchtext.datasets import AG_NEWS
import pickle

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model for LLaMA
tokenizer_llama = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-fp16")
model_llama = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-13B-fp16", output_hidden_states=True)
tokenizer_llama.pad_token = tokenizer_llama.eos_token
model_llama.to(device)

# Function to get layerwise hidden states for a batch
def get_batch_layerwise_hidden_states(input_ids, attention_mask):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model_llama(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

    layer_hidden_states = [hidden_states[i].mean(dim=-2) for i in range(1, len(hidden_states))]
    return torch.stack(layer_hidden_states).cpu().numpy()

# Function to process and save dataset
def process_and_save_dataset(dataset, file_name):
    processed_data = []

    for idx in tqdm(range(len(dataset)), desc=f"Processing {file_name}"):
        label, text = dataset[idx]

        # Tokenize the text
        encoded_text = tokenizer_llama(
            text,
            return_tensors="pt",
            padding='do_not_pad',
            truncation=True,
            max_length=4096
        )

        # Move input tensors to the device
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']

        # Get hidden states
        hidden_states = get_batch_layerwise_hidden_states(input_ids, attention_mask)

        #print(hidden_states.shape, label - 1)
        #input()

        processed_data.append((hidden_states, label - 1))

    # Save the processed data to a pickle file
    with open(file_name, 'wb') as f:
        pickle.dump(processed_data, f)

def main():
    # Load the AG News dataset
    train_dataset = list(AG_NEWS(split='train'))
    val_dataset = list(AG_NEWS(split='test'))

    # Process and save datasets
    process_and_save_dataset(train_dataset, 'train_dataset.pkl')
    process_and_save_dataset(val_dataset, 'val_dataset.pkl')

if __name__ == "__main__":
    main()
