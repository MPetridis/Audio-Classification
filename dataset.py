import os
import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score





class ESC50Dataset(Dataset):
  def __init__(self, csv_file, root_dir, split="train"):
    df=len(pd.read_csv(csv_file))
    if split == "train":
      self.len = int(df*0.8)
      self.shift = 0
    else:
      self.len = int(df*0.2)
      self.shift = int(df*0.8)
    self.annotations = pd.read_csv(csv_file)
    self.root_dir = root_dir
    self.transform = Compose([
      T.MelSpectrogram(sample_rate=44100, n_fft=1024, hop_length=512, n_mels=64),
      T.AmplitudeToDB()
    ])

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    index = self.shift + index
    audio_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
    label = self.annotations.iloc[index, 2]

    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = self.transform(waveform)
    
    return waveform, label
