import os
import pandas as pd
import torchaudio
import torch
from transformers import AutoModel
from tqdm import tqdm

import argparse
import os
import shutil
import csv
from pydub import AudioSegment

# Additional function to handle copying the original .wav file if no processing is needed
def copy_original_wav_file(original_path, adapted_path, filename):
    os.makedirs(adapted_path, exist_ok=True)
    new_path = os.path.join(adapted_path, filename)
    shutil.copy2(original_path, new_path)
    return new_path

# Function to parse the emotion label files
def parse_label_file(file_path, target_classes):
    emotions_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('['):
                parts = line.strip().split('\t')
                turn_name = parts[1].strip()
                emotion = parts[2].strip()
                vad = parts[3].strip('[]').split(',')
                if emotion in target_classes:
                    emotions_data[turn_name] = {
                        'emotion': target_classes[emotion],
                        'valence': vad[0],
                        'activation': vad[1],
                        'dominance': vad[2]
                    }
    return emotions_data

# Function to create folders for each emotion class
def create_folders_for_classes(base_path, classes):
    for class_name in set(classes.values()):
        os.makedirs(os.path.join(base_path, class_name), exist_ok=True)

# New function to process WAV files
def process_wav_file(file_path, target_sr=16000, max_length=5000):
    # Load the audio file
    audio = AudioSegment.from_wav(file_path)
    # Resample if necessary
    if audio.frame_rate != target_sr:
        audio = audio.set_frame_rate(target_sr)
    # Split if necessary
    parts = [audio[i:i+max_length] for i in range(0, len(audio), max_length)]
    return parts

def copy_and_process_wav_files(base_path, session, emotions_data, adapted_path):
    wav_path = os.path.join(base_path, session, 'sentences', 'wav')
    for subdir, dirs, files in os.walk(wav_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                turn_name = file.split('.')[0]
                if turn_name in emotions_data:
                    emotion = emotions_data[turn_name]['emotion']
                    # Process the WAV file
                    audio_parts = process_wav_file(file_path)
                    part_paths = []
                    for idx, part in enumerate(audio_parts):
                        new_filename = f"{turn_name}_part{idx}.wav"
                        new_path = os.path.join(adapted_path, session, emotion, new_filename)
                        part.export(new_path, format='wav')
                        part_paths.append(new_path)
                    # Yield file paths and metadata with session information
                    for part_path in part_paths:
                        yield part_path, session, emotion, emotions_data[turn_name]

def preprocess_and_save_features(csv_file_path, audio_folder_path, output_folder_path, ptm_model_name="microsoft/wavlm-large", target_length=80000):
    # Load the pre-trained model
    ptm = AutoModel.from_pretrained(ptm_model_name)
    ptm.eval()  # Set the model to evaluation mode

    # Load CSV file
    df = pd.read_csv(csv_file_path)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder_path, exist_ok=True)

    # Group data by session to process and save by session
    grouped = df.groupby('session')
    for session_id, group in grouped:
        session_path = os.path.join(output_folder_path, f"{session_id}")
        os.makedirs(session_path, exist_ok=True)

        # Process each audio file in the session
        for index, row in tqdm(group.iterrows(), total=group.shape[0], ncols=100):
            audio_path = os.path.join(audio_folder_path, row['path'])
            waveform, _ = torchaudio.load(audio_path)

            # Ensure the waveform is mono by taking the mean if it's not
            if waveform.shape[0] > 1:
                waveform = waveform.mean(axis=0, keepdim=True)

            # Pad or truncate the waveform to the target length
            if waveform.size(1) > target_length:
                waveform = waveform[:, :target_length]
            elif waveform.size(1) < target_length:
                pad_amount = target_length - waveform.size(1)
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

            # Process waveform with the pre-trained model
            with torch.no_grad():
                waveform = waveform.unsqueeze(0)  # Ensure it has a batch dimension if not present
                outputs = ptm(waveform.squeeze()[None, :], output_hidden_states=True)
            hidden_states = outputs.hidden_states
            stacked_hidden_states = torch.stack(hidden_states, dim=0)[1:]  # Adjust based on your needs

            # Save the processed features and label
            tensor_output_path = os.path.join(session_path, f"features_{index}.pt")
            torch.save({
                'features': stacked_hidden_states,
                'label': row['emotion']
            }, tensor_output_path)

def main(dataset_path, output_csv_path, adapted_path):
    target_classes = {'ang': 'angry', 'neu': 'neutral', 'sad': 'sadness', 'hap': 'happiness_excited', 'exc': 'happiness_excited'}
    sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

    # Create folders for each session and each emotion class within that session
    for session in sessions:
        session_path = os.path.join(adapted_path, session)
        create_folders_for_classes(session_path, target_classes)

    with open(output_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['path', 'session', 'emotion', 'activation', 'valence', 'dominance'])

        for session in sessions:
            label_path = os.path.join(dataset_path, session, 'dialog', 'EmoEvaluation')
            for label_file in os.listdir(label_path):
                if label_file.endswith('.txt'):
                    file_path = os.path.join(label_path, label_file)
                    emotions_data = parse_label_file(file_path, target_classes)
                    for file_info in copy_and_process_wav_files(dataset_path, session, emotions_data, adapted_path):
                        # Write the metadata for each processed part including session info
                        csvwriter.writerow([file_info[0], file_info[1], file_info[2], file_info[3]['activation'], file_info[3]['valence'], file_info[3]['dominance']])
    

    # Example usage
    csv_file_path = args.output_csv_path  # Path to your CSV file
    audio_folder_path = args.adapted_path            # Path to the folder containing the audio files
    output_folder_path = './processed_features'  # Where to save the processed feature tensors

    preprocess_and_save_features(csv_file_path, audio_folder_path, output_folder_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process IEMOCAP dataset")
    parser.add_argument('dataset_path', type=str, help='Path to the IEMOCAP_full_release directory')
    parser.add_argument('output_csv_path', type=str, help='Path to the output CSV file')
    parser.add_argument('adapted_path', type=str, help='Path to the adapted IEMOCAP directory')
    args = parser.parse_args()

    # Ensure the adapted path exists
    os.makedirs(args.adapted_path, exist_ok=True)

    #python3 _preprocess_iemocap.py ./IEMOCAP_full_release ./emotion_data.csv ./adapted_IEMOCAP
    
    main(args.dataset_path, args.output_csv_path, args.adapted_path)
