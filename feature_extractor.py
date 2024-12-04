from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import torch
from model import AttnVGG
from model import MLP,AttnVGG,AttentionBlock
from dataset import Dataset_prep
from torch.utils.data import DataLoader
from main import preprocess_data
from frouros.detectors.data_drift import KSTest,EMD

class Extractor(torch.nn.Module):
    def __init__(self, mlp, ant, mlp_nodes, attn_nodes, attn_features, up_factor, normalize_attn=True):
        super(Extractor, self).__init__()
        # Create feature extractors for the specified nodes
        self.mlp_extractor = create_feature_extractor(mlp, mlp_nodes)
        self.attn_extractor = create_feature_extractor(ant, attn_nodes)  # Directly use AttnVGG model

    def forward(self, x):
        # Forward pass through the MLP feature extractor
        x_flat = x.view(x.size(0), -1)  # Flatten input for MLP
        mlp_features = self.mlp_extractor(x_flat)
        mlp_fc1 = mlp_features["fc1"]
        # Reshape MLP output to match AttnVGG input (assume spatial dimensions of 8x8)
        batch_size = mlp_fc1.size(0)

        reshaped_input = mlp_fc1.view(batch_size,1,16,-1)  # Reshape to (B, C, H, W)

        # Forward pass through the AttnVGG feature extractor
        attn_features = self.attn_extractor(reshaped_input)
        out = attn_features["attn1"].view(attn_features["attn1"].size(0), -1)  # Directly use the model
        return out


def pad_data(data):
  max_len = max(tensor.size(0) for tensor in data)

  # Pad each tensor
  padded_data = [torch.nn.functional.pad(tensor, (0, 0, 0, max_len - tensor.size(0))) for tensor in data]

  # Stack the tensors
  stacked_tensor = torch.stack(padded_data, dim=0)
  return stacked_tensor

# mlp fc1
# attn attn1
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


def get_features(model,dataloader,device):
  features=[]
  for inputs,labels in dataloader:
    inputs=inputs.to(device)
    attn_fc1 = model(inputs)
    
    features.append(attn_fc1)
  return features


def save_features_as_tensors():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  mlp = MLP().to(device)
  ant = AttnVGG(num_classes=41, normalize_attn=False).to(device)

  # Load checkpoints
  ant.load_state_dict(torch.load("checkpoint_attention", map_location=device))
  mlp.load_state_dict(torch.load("checkpoint_MLP", map_location=device))

  # Define nodes for feature extraction
  mlp_nodes={
    "fc1":"fc1"
  }
  attn_nodes={
    "attn1":"attn1"
  }

  # Initialize extractor
  csv_file1 = "partitions/partition_1.csv"
  csv_file2 = "partitions/partition_2.csv"
  csv_file3 = "partitions/partition_3.csv"
  # Example input tensor
  root_dir = "E:\\FSDKaggle2018.audio_train\\FSDKaggle2018.audio_train"
  attn_features = 64   # Number of attention features
  up_factor = 2  
  extractor = Extractor(mlp, ant, mlp_nodes, attn_nodes, attn_features, up_factor).to(device)
  dataset = Dataset_prep(csv_file1, root_dir, "t")
  dataloader = DataLoader(dataset, batch_size=30, shuffle=True,collate_fn=collate_fn)
  d1=pad_data(get_features(extractor,dataloader,device))
  torch.save(d1, "partition_1_features.pt")
  del d1
  torch.cuda.empty_cache()
  dataset = Dataset_prep(csv_file2, root_dir, "t")
  dataloader = DataLoader(dataset, batch_size=30, shuffle=True,collate_fn=collate_fn)
  d2=pad_data(get_features(extractor,dataloader,device))
  torch.save(d2, "partition_2_features.pt")
  del d2
  torch.cuda.empty_cache()
  dataset = Dataset_prep(csv_file3, root_dir, "t")
  dataloader = DataLoader(dataset, batch_size=30, shuffle=True,collate_fn=collate_fn)
  d3=pad_data(get_features(extractor,dataloader,device))
  torch.save(d3, "partition_3_features.pt")
  del d3


def km_test(file_1,file_2,device):
  data1=torch.load(file_1,map_location=device,weights_only=True)
  data2=torch.load(file_2,map_location=device,weights_only=True)
  data1=reshaped_data = data1.reshape(-1, data1.shape[-1]).detach().cpu().numpy()
  data2=reshaped_data = data2.reshape(-1, data2.shape[-1]).detach().cpu().numpy()
  detector = KSTest()
  dd=[]
  for index,wf in enumerate(data1):
    _=detector.fit(X=wf)
    drift_score,_ = detector.compare(X=data2[index])[0]
    dd.append(drift_score)
  return sum(dd)/len(dd)

if __name__ =="__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print(km_test("partition_1_features.pt","partition_2_features.pt",device))