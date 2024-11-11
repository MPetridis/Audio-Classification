import os
import pandas as pd
import numpy as np
import torchaudio
from torch.utils.data import Dataset
import torchaudio.transforms as T
from torchvision.transforms import Compose



def partitions(dataset_csv,n_partition):
  directory_path='./partitions'

  if os.path.exists(directory_path):
    print(f"The directory '{directory_path}' exists.")
  else:
    os.mkdir(directory_path)
    df = pd.read_csv(dataset_csv)
    # Set the number of partitions
    N = n_partition  # Adjust this number as needed
    # Split the dataset into N equal partitions
    partitions = np.array_split(df, N)
    # Save each partition to a separate CSV file
    for i, partition in enumerate(partitions):
      partition.to_csv(f"{directory_path}/partition_{i+1}.csv", index=False)



class ESC50Dataset(Dataset):
  def __init__(self, csv_file, root_dir, split="train"):
    df=len(pd.read_csv(csv_file))
    if split == "train":
      self.len = int(df*0.8)
      self.shift = 0
    elif split == "val":
      self.len = int(df*0.2)
      self.shift = int(df*0.8)
    else:
      self.len = df
      self.shift = 0
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
