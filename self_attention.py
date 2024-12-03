import torch
import torch.nn as nn
from model import AttnVGG
from dataset import Dataset_prep,partitions
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from main import save_fig,print_report

def collate_fn(batch, target_size=(255, 255)):
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
        inputs = inputs.cuda()
        # inputs = inputs
        outputs = classifier(inputs)
        _, predicted = torch.max(outputs[0], 1)
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
        labels = labels.cuda()
        # labels = nn.functional.one_hot(labels, num_classes=41).float()
        inputs = inputs.cuda()
        # inputs = inputs

        outputs = classifier(inputs)
        val_loss += criterion(outputs[0], labels)
        n += 1
    return (val_loss/n).detach().item()

def evaluate(device,csv_file):
  root_dir = "E:\\FSDKaggle2018.audio_train\\FSDKaggle2018.audio_train"
  classifier = AttnVGG(num_classes=41, normalize_attn=False)
  classifier.load_state_dict(torch.load("checkpoint_attention", weights_only=True,map_location=device))
  classifier=classifier.cuda()
  classifier.eval()
  dataloader = DataLoader(Dataset_prep(csv_file, root_dir, "t"), collate_fn=collate_fn)

  # correct = 0
  # total = 0
  # since we're not training, we don't need to calculate the gradients for our outputs
  print_report(dataloader,classifier,device)

def train(epochs):
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # partitions("train_post_competition.csv",3)

    csv_file = "partitions/partition_1.csv"
    # root_dir = "C:\\Users\\petri\\Downloads\\ESC-50-master\\audio"
    root_dir = "E:\\FSDKaggle2018.audio_train\\FSDKaggle2018.audio_train"

    dataset = Dataset_prep(csv_file, root_dir, "train")
    train_loader = Dataset_prep(csv_file, root_dir, split="val")

    dataloader = DataLoader(dataset, batch_size=35, shuffle=True,collate_fn=collate_fn)
    dataloader_val = DataLoader(train_loader, batch_size=35, shuffle=False,collate_fn=collate_fn)

    model = AttnVGG(num_classes=41, normalize_attn=False)
    # model.load_state_dict(torch.load("checkpoint_attention", weights_only=True,map_location=device))
    model=model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

    data,label=next(iter(dataloader))
    print(data.shape)
    print(label.shape)
    print(model)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    model.train()
    for epoch in tqdm(range(epochs)):
        model.train()
        for inputs, labels in dataloader:

            # labels = nn.functional.one_hot(labels, num_classes=41).float()
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()

            # inputs = inputs
            # print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs[0], labels)

            loss.backward()
            optimizer.step()
        train_losses.append(loss.detach().item())
        val_losses.append(get_val_loss(dataloader_val,model,criterion,device))
        train_accuracies.append(calculate_accuracy(dataloader,model,device))
        val_accuracies.append(calculate_accuracy(dataloader_val,model,device))
        torch.cuda.empty_cache()
        # torch.cuda.ipc_collect()

    torch.save(model.state_dict(), "./checkpoint_attention")
    save_fig("self_attention_progress",train_losses,val_losses,train_accuracies,val_accuracies)

    
if __name__=="__main__":
    # train(8)
    for i in range(3):
      evaluate('cuda',f"partitions/partition_{i+1}.csv")