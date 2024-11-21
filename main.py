from dataset import Dataset_prep,partitions
from model import SoundClassifier
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import classification_report
from frouros.detectors.data_drift import KSTest,EMD


def collate_fn(batch):
    # Separate the inputs and labels
    inputs, labels = zip(*batch)

    # Find the maximum width and height in the batch
    max_height = max([input.size(1) for input in inputs])
    max_width = max([input.size(2) for input in inputs])

    # Pad each tensor to the maximum width and height
    padded_inputs = []
    for input in inputs:
        pad_height = max_height - input.size(1)
        pad_width = max_width - input.size(2)
        padded_input = torch.nn.functional.pad(input, (0, pad_width, 0, pad_height))
        padded_inputs.append(padded_input)

    # Stack the padded inputs and convert labels to tensor
    inputs = torch.stack(padded_inputs)
    labels = torch.tensor(labels)

    return inputs, labels

def train(l_r=0.001,bs=20,n_e=10,device="cuda"):
  
  dataset_csv="train_post_competition.csv"
  csv_file = "partitions/partition_1.csv"
  # root_dir = "C:\\Users\\petri\\Downloads\\ESC-50-master\\audio"
  root_dir = "/home/mlearning/audio/FSDKaggle2018.audio_train"
  partitions(dataset_csv,3)
  dataset = Dataset_prep(csv_file, root_dir, "train")
  # classifier = SoundClassifier().cuda()
  classifier = SoundClassifier()
  criterion = nn.BCEWithLogitsLoss()
  # classifier.load_state_dict(torch.load("./checkpoint"))

  lr = l_r
  batch_size = bs
  optimizer = Adam(classifier.parameters(), lr=lr)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
  dataset_val = Dataset_prep(csv_file, root_dir, split="val")
  dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
  n_epoch = n_e
  # classifier.train()
  train_losses = []
  val_losses = []
  train_accuracies = []
  val_accuracies = []
  classifier = classifier.to(device)
  classifier.train()

  for epoch in tqdm(range(n_epoch)):
    classifier.train()
    for inputs, labels in dataloader:
      
      optimizer.zero_grad()

      labels = nn.functional.one_hot(labels, num_classes=41).float().to(device)
      # labels = nn.functional.one_hot(labels, num_classes=41).float()
      inputs = inputs.to(device)
      # inputs = inputs

      outputs = classifier(inputs)
      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()
    # print(f"EPOCH {epoch+1} | Loss: {loss.detach().item():.4f}")
    
    train_losses.append(loss.detach().item())
    val_losses.append(get_val_loss(dataloader_val,classifier,criterion,device))
    train_accuracies.append(calculate_accuracy(dataloader,classifier,device))
    val_accuracies.append(calculate_accuracy(dataloader_val,classifier,device))

  try:
    print_report(dataloader_val,classifier,device)
  except Exception as e:
     print(e)
  torch.save(classifier.state_dict(), "./checkpoint")
  save_fig("progress",train_losses,val_losses,train_accuracies,val_accuracies)

@torch.no_grad
def print_report(dataloader,classifier,device):
  all_pred=[]
  all_true=[]
  classifier.eval()
  for data in dataloader:
    images, labels = data
    images=images.to(device)
    # calculate outputs by running images through the network
    outputs = classifier(images)
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs, 1)
    all_pred.extend(predicted.cpu().numpy())
    all_true.extend(labels.cpu().numpy())
    
  print(classification_report(all_true,all_pred))
    
def evaluate(device,csv_file):
  # csv_file = "partitions/partition_2.csv"
  # root_dir = "C:\\Users\\petri\\Downloads\\ESC-50-master\\audio"
  root_dir = "E:\\FSDKaggle2018.audio_train"
  classifier = SoundClassifier()
  classifier.load_state_dict(torch.load("checkpoint", weights_only=True,map_location=device))
  classifier=classifier.to(device)
  classifier.eval()
  dataloader = DataLoader(Dataset_prep(csv_file, root_dir, "t"), collate_fn=collate_fn)

  # correct = 0
  # total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  print_report(dataloader,classifier,device)
  # with torch.no_grad():
  #   for data in dataloader:
  #     images, labels = data
  #     # calculate outputs by running images through the network
  #     images=images.to(device)
  #     outputs = classifier(images)
  #     # the class with the highest energy is what we choose as prediction
  #     _, predicted = torch.max(outputs, 1)
  #     total += labels.size(0)
  #     correct += (predicted == labels).sum().item()

  # print(f'Accuracy of the network on the {total} test audio: {100 * correct // total} %')

def save_fig(name,train_losses,val_losses,train_accuracies,val_accuracies):
  fig, ax = plt.subplots(1, 2, figsize=(14, 4))
  ax[0].set_title("Losses")
  ax[0].set_xlabel("Epoch")
  ax[0].set_ylabel("Loss")
  ax[0].plot(train_losses, c="blue", label="Train Loss")
  ax[0].plot(val_losses, c="green", label="Validation Loss")
  ax[0].legend()
  ax[0].grid()

  ax[1].set_title("Accuracies")
  ax[1].set_xlabel("Epoch")
  ax[1].set_ylabel("Accuracy")
  ax[1].plot(train_accuracies, c="blue", label="Train Accuracy")
  ax[1].plot(val_accuracies, c="green", label="Validation Accuracy")
  ax[1].legend()
  ax[1].grid()
  fig.savefig(f"./{name}.png")

@torch.no_grad
def calculate_accuracy(dataloader,classifier,device):
    # Evaluation
    all_preds = []
    all_labels = []
    classifier.eval()
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        # inputs = inputs
        outputs = classifier(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

@torch.no_grad
def get_val_loss(dataloader_val,classifier,criterion,device):
    classifier.eval()
    val_loss = 0
    n = 0
    for inputs, labels in dataloader_val:
        labels = nn.functional.one_hot(labels, num_classes=41).float().to(device)
        # labels = nn.functional.one_hot(labels, num_classes=41).float()
        inputs = inputs.to(device)
        # inputs = inputs

        outputs = classifier(inputs)
        val_loss += criterion(outputs, labels)
        n += 1
    return (val_loss/n).detach().item()

def fn():
  dataset_csv="train_post_competition.csv"
  csv_file = "partitions/partition_1.csv"
  # root_dir = "C:\\Users\\petri\\Downloads\\ESC-50-master\\audio"
  root_dir = "E:\\FSDKaggle2018.audio_train"
  partitions(dataset_csv,3)
  dataset = Dataset_prep(csv_file, root_dir, "train")
  # classifier = SoundClassifier().cuda()
  classifier = SoundClassifier()
  dataloader = DataLoader(dataset, batch_size=180, shuffle=True,collate_fn=collate_fn)
  print(classifier)
  print(dataloader)
  train_features, train_labels = next(iter(dataloader))
  print(f"Feature batch shape: {train_features.shape}")
  print(f"Labels batch shape: {train_labels}")
  img = train_features[0].squeeze()
  label = train_labels[0]
  plt.imshow(img, cmap="viridis")
  plt.show()
  print(f"Label: {label}")
  # for inputs, labels in dataloader:
  #       print(f"Inputs shape: {inputs.shape}")
  #       print(f"Labels shape: {labels.shape}")
  #       break  # Only inspect the first batch

def data_drift_detect_kmt(csv_file1, csv_file2):
  root_dir = "E:\\FSDKaggle2018.audio_train"
  
  # Load datasets
  dataset1 = Dataset_prep(csv_file1, root_dir, "t")
  dataset2 = Dataset_prep(csv_file2, root_dir, "t")
  

  data1 = preprocess_data(dataset1)  # Adjust length as needed
  data2 = preprocess_data(dataset2)
  # Drift detection using KSTest
  # pca = PCA(n_components=50)
  # data1 = pca.fit_transform(data1)
  # data2 = pca.fit_transform(data2)
  detector = KSTest()
  dd=[]
  for index,wf in enumerate(data1):
    _=detector.fit(X=wf)
    drift_score,_ = detector.compare(X=data2[index])[0]
    dd.append(drift_score)
  
  return sum(dd)/len(dd)

def data_drift_detect_emd(csv_file1, csv_file2):
  root_dir = "E:\\FSDKaggle2018.audio_train"
  
  # Load datasets
  dataset1 = Dataset_prep(csv_file1, root_dir, "t")
  dataset2 = Dataset_prep(csv_file2, root_dir, "t")
  

  data1 = preprocess_data(dataset1)  # Adjust length as needed
  data2 = preprocess_data(dataset2)
  # Drift detection using KSTest
  # pca = PCA(n_components=50)
  # data1 = pca.fit_transform(data1)
  # data2 = pca.fit_transform(data2)
  detector = EMD()
  dd=[]
  for index,wf in enumerate(data1):
    _=detector.fit(X=wf)
    drift_score= detector.compare(X=data2[index])[0]
    dd.append(drift_score.distance)
  
  return sum(dd)/len(dd)

def preprocess_data(dataset):
  max_height = max([waveform.size(1) for waveform, _ in dataset])
  max_width = max([waveform.size(2) for waveform, _ in dataset])

  padded_data=[]
  for waveform, _  in dataset:
      pad_height = max_height - waveform.size(1)
      pad_width = max_width - waveform.size(2)
      padded_waveform = torch.nn.functional.pad(waveform, (0, pad_width, 0, pad_height))
      padded_data.append(padded_waveform.flatten().numpy())
      # Flatten the padded waveform and append to processed_data
  pca = PCA(n_components=60)
  reduced_data = pca.fit_transform(padded_data)
  # reduced_data=padded_data
  return reduced_data



if __name__=="__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # train(bs=300,n_e=100,device=device)
  # for i in range(3):
  #   evaluate(device,f"partitions/partition_{i+1}.csv")
  fn()
  # print(data_drift_detect_kmt("partitions/partition_1.csv","partitions/partition_2.csv"))
  # print(data_drift_detect_emd("partitions/partition_1.csv","partitions/partition_2.csv"))