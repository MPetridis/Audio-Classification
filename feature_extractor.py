from torchvision.models.feature_extraction import create_feature_extractor
import torch
import torch.nn as nn
from model import Attn
from model import MLP
from dataset import Dataset_prep
from torch.utils.data import DataLoader
from frouros.detectors.data_drift import KSTest,EMD,EnergyDistance,BWSTest,KuiperTest,HellingerDistance,KL,PSI,HINormalizedComplement
from tqdm import tqdm

class Extractor(torch.nn.Module):
    def __init__(self, mlp, ant, mlp_nodes, attn_nodes):
        super(Extractor, self).__init__()
        # Create feature extractors for the specified nodes
        self.mlp_extractor = create_feature_extractor(mlp, mlp_nodes)
        self.linear_transform=nn.Linear(1221, 156288)
        self.attn_extractor = create_feature_extractor(ant, attn_nodes)
        self.layer=nn.Linear(1221,32)


    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input for MLP
        mlp_features = self.mlp_extractor(x)
        x = self.linear_transform(mlp_features['fc1'])
        x = x.view(x.size(0),-1)  # Flatten for attention
        x=self.attn_extractor(x)
        out = x['attention'].view(x['attention'].size(0), -1)  # Directly use the model
        return out

class Extractor_MLP(torch.nn.Module):
  def __init__(self, mlp, mlp_nodes):
    super(Extractor_MLP, self).__init__()
    self.mlp_extractor = create_feature_extractor(mlp, mlp_nodes)
    self.layer=nn.Linear(1221,32)
  def forward(self,x):
    x = x.view(x.size(0), -1)  # Flatten input for MLP
    x = self.mlp_extractor(x)
    x = x['fc1'].view( x['fc1'].size(0),-1) 
    x= self.layer(x)
  
    return x

class Extractor_Attn(torch.nn.Module):
  def __init__(self,ant, attn_nodes):
    super(Extractor_Attn, self).__init__()
    self.extractor = create_feature_extractor(ant, attn_nodes)
  def forward(self, x):
    x = x.view(x.size(0), -1)  
    x=self.extractor(x)
    out = x['attention'].view(x['attention'].size(0), -1)  
    return out

class Extractor_reverse(torch.nn.Module):
  def __init__(self, mlp, ant, mlp_nodes, attn_nodes):
    super(Extractor_reverse, self).__init__()
    # Create feature extractors for the specified nodes
    self.attn_extractor = create_feature_extractor(ant, attn_nodes)
    self.linear_transform=nn.Linear(32, 156288)
    self.mlp_extractor = create_feature_extractor(mlp, mlp_nodes)
    self.layer=nn.Linear(1221,32)

  def forward(self, x):
    x = x.view(x.size(0), -1)  # Flatten input for MLP
    x=self.attn_extractor(x)
    x = x['attention'].view(x['attention'].size(0), -1)  # Directly use the model
    x=self.linear_transform(x)
    mlp_features = self.mlp_extractor(x)
    x = mlp_features['fc1'].view(mlp_features['fc1'].size(0), -1) 
    x=self.layer(x)
    return x

def pad_data(data):
  max_len = max(tensor.size(0) for tensor in data)

  # Pad each tensor
  padded_data = [torch.nn.functional.pad(tensor, (0, 0, 0, max_len - tensor.size(0))) for tensor in data]

  # Stack the tensors
  stacked_tensor = torch.stack(padded_data, dim=0)
  return stacked_tensor

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
  # model.eval()
  for inputs,labels in tqdm(dataloader):
    inputs=inputs.to(device)
    attn_fc1 = model(inputs)
    
    features.append(attn_fc1)
  
  return features

def save_features_as_tensors():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  mlp = MLP().to(device)
  ant = Attn().to(device)

  # Load checkpoints
  ant.load_state_dict(torch.load("checkpoint_attention", map_location=device))
  mlp.load_state_dict(torch.load("checkpoint_MLP", map_location=device))

  # Define nodes for feature extraction
  mlp_nodes={
    "fc1":"fc1"
  }
  attn_nodes={
    "attention":"attention"
  }

  # Initialize extractor
  csv_file1 = "partitions/partition_1.csv"
  csv_file2 = "partitions/partition_2.csv"
  csv_file3 = "partitions/partition_3.csv"
  # Example input tensor
  root_dir = "E:\\FSDKaggle2018.audio_train\\FSDKaggle2018.audio_train" 
  # extractor = Extractor(mlp, ant, mlp_nodes, attn_nodes).to(device) #mlp + attention
  # extractor = Extractor_reverse(mlp, ant, mlp_nodes, attn_nodes).to(device) #atterntio + mlp
  # extractor=Extractor_MLP(mlp,mlp_nodes).to(device) #mlp
  extractor=Extractor_Attn(ant,attn_nodes).to(device) #attention
  dataset = Dataset_prep(csv_file1, root_dir, "t")
  dataloader = DataLoader(dataset, batch_size=128, shuffle=True,collate_fn=collate_fn)
  d1=pad_data(get_features(extractor,dataloader,device))
  torch.save(d1, "partition_1_features_attn2.pt")
  del d1
  torch.cuda.empty_cache()
  dataset = Dataset_prep(csv_file2, root_dir, "t")
  dataloader = DataLoader(dataset, batch_size=128, shuffle=True,collate_fn=collate_fn)
  d2=pad_data(get_features(extractor,dataloader,device))
  torch.save(d2, "partition_2_features_attn2.pt")
  del d2
  torch.cuda.empty_cache()
  dataset = Dataset_prep(csv_file3, root_dir, "t")
  dataloader = DataLoader(dataset, batch_size=128, shuffle=True,collate_fn=collate_fn)
  d3=pad_data(get_features(extractor,dataloader,device))
  torch.save(d3, "partition_3_features_attn2.pt")
  del d3

def km_test(file_1,file_2,device):
  data1=torch.load(file_1,map_location=device,weights_only=True)
  data2=torch.load(file_2,map_location=device,weights_only=True)
  data1=data1.reshape(-1, data1.shape[-1]).detach().cpu().numpy()
  data2=data2.reshape(-1, data2.shape[-1]).detach().cpu().numpy()
  detector = KSTest()
  dd=[]
  for index,wf in tqdm(enumerate(data1)):
    _=detector.fit(X=wf)
    drift_score,_ = detector.compare(X=data2[index])
    dd.append(drift_score[1])
  return sum(dd)/len(dd)

def emd_detector(file_1,file_2,device):
  data1=torch.load(file_1,map_location=device,weights_only=True)
  data2=torch.load(file_2,map_location=device,weights_only=True)
  data1=data1.reshape(-1, data1.shape[-1]).detach().cpu().numpy()
  data2=data2.reshape(-1, data2.shape[-1]).detach().cpu().numpy()
  detector = EMD()
  dd=[]
  for index,wf in tqdm(enumerate(data1)):
    _=detector.fit(X=wf)
    drift_score= detector.compare(X=data2[index])[0]
    dd.append(drift_score.distance)
  
  return sum(dd)/len(dd)

def PSI_detector(file_1,file_2,device):
  data1=torch.load(file_1,map_location=device,weights_only=True)
  data2=torch.load(file_2,map_location=device,weights_only=True)
  data1=data1.reshape(-1, data1.shape[-1]).detach().cpu().numpy()
  data2=data2.reshape(-1, data2.shape[-1]).detach().cpu().numpy()
  detector = PSI(num_bins=20)
  dd=[]
  for index,wf in tqdm(enumerate(data1)):
    _=detector.fit(X=wf)
    drift_score= detector.compare(X=data2[index])[0]
    dd.append(drift_score.distance)
  
  return sum(dd)/len(dd)

def HINormalizedComplement_detector(file_1,file_2,device):
  data1=torch.load(file_1,map_location=device,weights_only=True)
  data2=torch.load(file_2,map_location=device,weights_only=True)
  data1=data1.reshape(-1, data1.shape[-1]).detach().cpu().numpy()
  data2=data2.reshape(-1, data2.shape[-1]).detach().cpu().numpy()
  detector =   HINormalizedComplement(num_bins=20)
  dd=[]
  for index,wf in tqdm(enumerate(data1)):
    _=detector.fit(X=wf)
    drift_score= detector.compare(X=data2[index])[0]
    dd.append(drift_score.distance)
  
  return sum(dd)/len(dd)

def energy_Distance(file_1,file_2,device):
  data1=torch.load(file_1,map_location=device,weights_only=True)
  data2=torch.load(file_2,map_location=device,weights_only=True)
  data1=data1.reshape(-1, data1.shape[-1]).detach().cpu().numpy()
  data2=data2.reshape(-1, data2.shape[-1]).detach().cpu().numpy()
  detector = EnergyDistance()
  dd=[]
  # print(data1.shape) #(3200,32)
  
  for index,wf in tqdm(enumerate(data1)):
    # print(wf) #(32,0)
    _=detector.fit(X=wf)
    drift_score= detector.compare(X=data2[index])[0]
    dd.append(drift_score.distance)
  
  return sum(dd)/len(dd)

def data_drift_bwsTest(file_1,file_2,device):
  data1=torch.load(file_1,map_location=device,weights_only=True)
  data2=torch.load(file_2,map_location=device,weights_only=True)
  data1=data1.reshape(-1, data1.shape[-1]).detach().cpu().numpy()
  data2=data2.reshape(-1, data2.shape[-1]).detach().cpu().numpy()
  detector = BWSTest()
  dd=[]
  for index,wf in tqdm(enumerate(data1)):
    _=detector.fit(X=wf)
    drift_score,_ = detector.compare(X=data2[index])
    dd.append(drift_score[1])
  return sum(dd)/len(dd)

def data_drift_KuiperTest(file_1,file_2,device):
  data1=torch.load(file_1,map_location=device,weights_only=True)
  data2=torch.load(file_2,map_location=device,weights_only=True)
  data1=data1.reshape(-1, data1.shape[-1]).detach().cpu().numpy()
  data2=data2.reshape(-1, data2.shape[-1]).detach().cpu().numpy()
  detector = KuiperTest()
  dd=[]
  for index,wf in tqdm(enumerate(data1)):
    _=detector.fit(X=wf)
    drift_score,_ = detector.compare(X=data2[index])
    dd.append(drift_score[1])
  return sum(dd)/len(dd)

def data_drift_HellingerDistance(file_1,file_2,device):
  data1=torch.load(file_1,map_location=device,weights_only=True)
  data2=torch.load(file_2,map_location=device,weights_only=True)
  data1=data1.reshape(-1, data1.shape[-1]).detach().cpu().numpy()
  data2=data2.reshape(-1, data2.shape[-1]).detach().cpu().numpy()
  detector = HellingerDistance(num_bins=20)
  dd=[]
  for index,wf in tqdm(enumerate(data1)):
    _=detector.fit(X=wf)
    drift_score= detector.compare(X=data2[index])[0]
    dd.append(drift_score.distance)
  
  return sum(dd)/len(dd)

def data_drift_KL(file_1,file_2,device):
  data1=torch.load(file_1,map_location=device,weights_only=True)
  data2=torch.load(file_2,map_location=device,weights_only=True)
  data1=data1.reshape(-1, data1.shape[-1]).detach().cpu().numpy()
  data2=data2.reshape(-1, data2.shape[-1]).detach().cpu().numpy()
  detector = KL(num_bins=20)
  dd=[]
  for index,wf in tqdm(enumerate(data1)):

    _=detector.fit(X=wf)
    drift_score= detector.compare(X=data2[index])[0]
    dd.append(drift_score.distance)
  
  return sum(dd)/len(dd)



if __name__ =="__main__":
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  print("HINormalizedComplement: " ,HINormalizedComplement_detector("partition_1_features_re2.pt","partition_2_features_re2.pt",device))
  print("EMD: " ,emd_detector("partition_1_features_re2.pt","partition_2_features_re2.pt",device))
  print("Energy: " ,energy_Distance("partition_1_features_re2.pt","partition_2_features_re2.pt",device))
  print("PSI_detector: " ,PSI_detector("partition_1_features_re2.pt","partition_2_features_re2.pt",device))
  print("HellingerDistance: " ,data_drift_HellingerDistance("partition_1_features_re2.pt","partition_2_features_re2.pt",device))
  # save_features_as_tensors()
