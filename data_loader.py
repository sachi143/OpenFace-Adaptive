import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import os

class MOSEIDataset(Dataset):
    def __init__(self, data_path, split='train'):
        """
        Loads CMU-MOSEI data from a pickle file.
        Expected structure:
        {
            'train': {'visual': [...], 'audio': [...], 'text': [...], 'labels': [...]},
            'valid': {...},
            'test': {...}
        }
        """
        self.split = split
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}. Please ensure 'mosei_data.pkl' exists.")
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {split} set from {data_path}")

    def __len__(self):
        return len(self.data[self.split]['labels'])

    def __getitem__(self, idx):
        # 1. Visual: OpenFace Features (AUs, Gaze) -> Size 35
        visual = torch.tensor(self.data[self.split]['visual'][idx], dtype=torch.float32)
        
        # 2. Audio: Wav2Vec or COVAREP Features -> Size 74
        audio = torch.tensor(self.data[self.split]['audio'][idx], dtype=torch.float32)
        
        # 3. Text: BERT/RoBERTa Embeddings -> Size 768
        text = torch.tensor(self.data[self.split]['text'][idx], dtype=torch.float32)
        
        # 4. Label: Emotion (Happy, Sad, Angry, etc.)
        label = torch.tensor(self.data[self.split]['labels'][idx], dtype=torch.long)
        
        return visual, audio, text, label
