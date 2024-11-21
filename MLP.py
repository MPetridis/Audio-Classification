from model import MLP
import torch
import torch.nn as nn
from tqdm import tqdm
from dataset import Dataset_prep
from torch.utils.data import DataLoader
from main import get_val_loss ,calculate_accuracy,save_fig

def collate_fn(batch, target_size=(64, 1024)):
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
  root_dir = "E:\\FSDKaggle2018.audio_train"

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


if __name__ =="__main__":

  train(5,32)
