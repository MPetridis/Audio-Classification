import torch
import torch.nn as nn
from model import SelfAttention
from dataset import Dataset_prep,partitions
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
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
  root_dir = "C:\\Users\\Minas Petridis\\Desktop\\FSDKaggle2018.audio_train"
  classifier = SelfAttention()
  classifier.load_state_dict(torch.load("checkpoint_MLP", weights_only=True,map_location=device))
  classifier=classifier.to(device)
  classifier.eval()
  dataloader = DataLoader(Dataset_prep(csv_file, root_dir, "t"), collate_fn=collate_fn)

  # correct = 0
  # total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  print_report(dataloader,classifier,device)


def train():
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_csv="train_post_competition.csv"
    csv_file = "partitions/partition_1.csv"
    # root_dir = "C:\\Users\\petri\\Downloads\\ESC-50-master\\audio"
    root_dir = "E:\\FSDKaggle2018.audio_train\\FSDKaggle2018.audio_train"

    dataset = Dataset_prep(csv_file, root_dir, "train")
    train_loader = Dataset_prep(csv_file, root_dir, split="val")
    sa = SelfAttention(d=2442, d_q=2, d_k=2, d_v=4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(sa.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    dataloader = DataLoader(dataset, batch_size=180, shuffle=True,collate_fn=collate_fn)
    dataloader_val = DataLoader(train_loader, batch_size=180, shuffle=False,collate_fn=collate_fn)
    sa.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in tqdm(range(5)):
        sa.train()
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = sa(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        train_losses.append(loss.detach().item())
        val_losses.append(get_val_loss(dataloader_val,sa,criterion,device))
        train_accuracies.append(calculate_accuracy(dataloader,sa,device))
        val_accuracies.append(calculate_accuracy(dataloader_val,sa,device))

if __name__=="__main__":
    train()