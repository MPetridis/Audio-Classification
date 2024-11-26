from model import MLP
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from dataset import Dataset_prep,partitions
from torch.utils.data import DataLoader
from main import save_fig,print_report

def collate_fn(batch, target_size=(64, 2442)):
    # Separate the inputs and labels
    inputs, labels = zip(*batch)

    # Target size for padding (fixed width and height)
    target_height, target_width = target_size

    # Pad each tensor to the target size
    padded_inputs = []
    for input in inputs:
        # Calculate padding for height and width
        pad_height = target_height - input.size(1)
        pad_width = target_width - input.size(2)

        # Apply padding (pad right and bottom)
        padded_input = torch.nn.functional.pad(input, (0, pad_width, 0, pad_height))
        padded_inputs.append(padded_input)

    # Stack the padded inputs and convert labels to tensor
    inputs = torch.stack(padded_inputs)
    labels = torch.tensor(labels, dtype=torch.long)

    return inputs, labels
def train(epochs,batch_size):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dataset_csv="train_post_competition.csv"
  csv_file = "partitions/partition_1.csv"
  # root_dir = "C:\\Users\\petri\\Downloads\\ESC-50-master\\audio"
  root_dir = "E:\\FSDKaggle2018.audio_train\\FSDKaggle2018.audio_train"

  dataset = Dataset_prep(csv_file, root_dir, "train")
  train_loader = Dataset_prep(csv_file, root_dir, split="val")
  model = MLP().to(device)  # Pass the input_size to the model
  # classifier = SoundClassifier().cuda()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
  # classifier.load_state_dict(torch.load("./checkpoint"))
  
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
  dataloader_val = DataLoader(train_loader, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
  data,label=next(iter(dataloader))
  print(data.shape)
  print(label.shape)
  
  train_losses = []
  val_losses = []
  train_accuracies = []
  val_accuracies = []
  model.train()
  for epoch in tqdm(range(epochs)):
    model.train()
    for inputs, labels in dataloader:

        # labels = nn.functional.one_hot(labels, num_classes=41).float()
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # inputs = inputs
        # print(inputs.shape)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
    train_losses.append(loss.detach().item())
    val_losses.append(get_val_loss(dataloader_val,model,criterion,device))
    train_accuracies.append(calculate_accuracy(dataloader,model,device))
    val_accuracies.append(calculate_accuracy(dataloader_val,model,device))

  torch.save(model.state_dict(), "./checkpoint_MLP")
  save_fig("MPL_progress",train_losses,val_losses,train_accuracies,val_accuracies)

@torch.no_grad()
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
    # accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

@torch.no_grad()
def get_val_loss(dataloader_val,classifier,criterion,device):
    classifier.eval()
    val_loss = 0
    n = 0
    for inputs, labels in dataloader_val:
        labels = labels.to(device)
        # labels = nn.functional.one_hot(labels, num_classes=41).float()
        inputs = inputs.to(device)
        # inputs = inputs

        outputs = classifier(inputs)
        val_loss += criterion(outputs, labels)
        n += 1
    return (val_loss/n).detach().item()

def evaluate(device,csv_file):
  root_dir = "E:\\FSDKaggle2018.audio_train\\FSDKaggle2018.audio_train"
  classifier = MLP()
  classifier.load_state_dict(torch.load("checkpoint_MLP", weights_only=True,map_location=device))
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

if __name__ =="__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

  train(30,180)
  # for i in range(3):
  #   evaluate(device,f"partitions/partition_{i+1}.csv")

