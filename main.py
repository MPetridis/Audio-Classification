from dataset import Dataset_prep,partitions
from model import SoundClassifier
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from sklearn.metrics import classification_report
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

def train(l_r=0.001,bs=20,n_e=10):
  dataset_csv="train_post_competition.csv"
  csv_file = "partitions/partition_1.csv"
  # root_dir = "C:\\Users\\petri\\Downloads\\ESC-50-master\\audio"
  root_dir = "E:\\FSDKaggle2018.audio_train"
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
  classifier.train()
  train_losses = []
  val_losses = []
  train_accuracies = []
  val_accuracies = []

  classifier.train()

  for epoch in tqdm(range(n_epoch)):
    for inputs, labels in tqdm(dataloader,desc=f"EPOCH {epoch+1}"):
      optimizer.zero_grad()

      # labels = nn.functional.one_hot(labels, num_classes=50).float().cuda()
      labels = nn.functional.one_hot(labels, num_classes=41).float()
      # inputs = inputs.cuda()
      inputs = inputs

      outputs = classifier(inputs)
      loss = criterion(outputs, labels)


      loss.backward()
      optimizer.step()
    print(f"EPOCH {epoch+1} | Loss: {loss.detach().item():.4f}")
    
    train_losses.append(loss.detach().item())
    val_losses.append(get_val_loss(dataloader_val,classifier,criterion))
    train_accuracies.append(calculate_accuracy(dataloader,classifier))
    val_accuracies.append(calculate_accuracy(dataloader_val,classifier))

  print_report(dataloader_val,classifier)

  torch.save(classifier.state_dict(), "./checkpoint")
  save_fig(train_losses,val_losses,train_accuracies,val_accuracies)

@torch.no_grad
def print_report(dataloader,classifier):
  all_pred=[]
  all_true=[]
  for data in dataloader:
    images, labels = data
    # calculate outputs by running images through the network
    outputs = classifier(images)
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)
    all_pred.append(predicted)
    all_true.append(labels)
  print(classification_report(all_true,all_pred))
    


def evaluate():
  csv_file = "partitions/partition_2.csv"
  # root_dir = "C:\\Users\\petri\\Downloads\\ESC-50-master\\audio"
  root_dir = "E:\\ESC-50-master\\audio"
  classifier = SoundClassifier()
  classifier.load_state_dict(torch.load("checkpoint", weights_only=True))
  classifier.eval()
  dataloader = DataLoader(Dataset_prep(csv_file, root_dir, "t",collate_fn=collate_fn))
  correct = 0
  total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs

  with torch.no_grad():
    for data in dataloader:
      images, labels = data
      # calculate outputs by running images through the network
      outputs = classifier(images)
      # the class with the highest energy is what we choose as prediction
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print(f'Accuracy of the network on the {total} test audio: {100 * correct // total} %')


def save_fig(train_losses,val_losses,train_accuracies,val_accuracies):
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
  fig.savefig("./progress.png")

@torch.no_grad
def calculate_accuracy(dataloader,classifier):
    # Evaluation
    all_preds = []
    all_labels = []

    for inputs, labels in dataloader:
        # inputs = inputs.cuda()
        inputs = inputs
        outputs = classifier(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy
@torch.no_grad
def get_val_loss(dataloader_val,classifier,criterion):
    val_loss = 0
    n = 0
    for inputs, labels in dataloader_val:
        # labels = nn.functional.one_hot(labels, num_classes=50).float().cuda()
        labels = nn.functional.one_hot(labels, num_classes=41).float()
        # inputs = inputs.cuda()
        inputs = inputs

        outputs = classifier(inputs)
        val_loss += criterion(outputs, labels)
        n += 1
    return (val_loss/n).detach().item()



if __name__=="__main__":
  train(bs=80)