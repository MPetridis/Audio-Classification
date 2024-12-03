import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models

class ResBlock(nn.Module):
  def __init__(self, in_channels, out_channels, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(0.2)
    self.downsample = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_channels)
    ) if stride != 1 or in_channels != out_channels else nn.Identity()

  def forward(self, x):
    identity = x
    identity = self.downsample(identity)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x += identity
    x = self.relu(x)
    x = self.dropout(x)
    return x
  

class SoundClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.con1 = nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
    self.dropout = nn.Dropout(0.2)
    self.bn1 = nn.BatchNorm2d(16)
    self.layers = nn.Sequential(
        ResBlock(16, 16, 1),
        ResBlock(16, 32, 2),
        ResBlock(32, 32, 1),
        ResBlock(32, 32, 1),
        ResBlock(32, 64, 2),
        ResBlock(64, 64, 1),
        ResBlock(64, 64, 1),
        ResBlock(64, 128, 2)
        )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(128, 41)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.relu(self.bn1(self.con1(x)))
    x = self.dropout(x)
    x = self.layers(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x
class MLP(nn.Module):
  def __init__(self):
    super(MLP, self).__init__()
    # First fully connected layer
    self.fc1 = nn.Linear(64 * 2442, 512)
    # Activation function
    # self.relu = nn.ReLU()
    # # Second fully connected layer (output layer)
    self.fc2 = nn.Linear(512, 41)

  def forward(self, x):
    x = x.view(x.size(0), -1)  # Flatten input
    x = self.fc1(x)            # Apply first fully connected layer          # Apply activation
    out = self.fc2(x)          # Apply second fully connected layer
    return out
  

# class SelfAttention(nn.Module):
#     def __init__(self, d, d_q, d_k, d_v):
#         super(SelfAttention, self).__init__()
#         self.d = d
#         self.d_q = d_q
#         self.d_k = d_k
#         self.d_v = d_v
#         self.W_query = nn.Parameter(torch.rand(d, d_q))
#         self.W_key = nn.Parameter(torch.rand(d, d_k))
#         self.W_value = nn.Parameter(torch.rand(d, d_v))
#     def forward(self, x):
#         Q = x @ self.W_query
#         K = x @ self.W_key
#         V = x @ self.W_value
#         attention_scores = Q @ K.T / math.sqrt(self.d_k)
#         attention_weights = F.softmax(attention_scores, dim=-1)
#         context_vector = attention_weights @ V
#         return context_vector

class AttentionBlock(nn.Module):
  def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True):
    super(AttentionBlock, self).__init__()
    self.up_factor = up_factor
    self.normalize_attn = normalize_attn
    self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
    self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
    self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
  def forward(self, l, g):
    N, C, W, H = l.size()
    l_ = self.W_l(l)
    g_ = self.W_g(g)
    # if l_.shape[2] != g_.shape[2] or l_.shape[3] != g_.shape[3]:
    #   g_ = F.interpolate(g_, size=(l_.size(2), l_.size(3)), mode='bilinear', align_corners=False)
    # c = self.phi(F.relu(l_ + g_))
    # if self.up_factor > 1:
    #     g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
    g_ = F.interpolate(g_, size=l_.shape[2:], mode='bilinear', align_corners=False)
    c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
    
    # compute attn map
    if self.normalize_attn:
        a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
    else:
        a = torch.sigmoid(c)
    # re-weight the local feature
    f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
    if self.normalize_attn:
        output = f.view(N,C,-1).sum(dim=2) # weighted sum
    else:
        output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C) # global average pooling
    return a, output
  

class AttnVGG(nn.Module):
  def __init__(self, num_classes, normalize_attn=False, dropout=None,input_channels=1):
      super(AttnVGG, self).__init__()
      net = models.vgg16_bn(pretrained=True)
      if input_channels == 1:
            net.features[0] = nn.Conv2d(
                in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
            )
            # Initialize with pretrained weights
            pretrained_weights = models.vgg16_bn(pretrained=True).features[0].weight
            net.features[0].weight.data = pretrained_weights.sum(dim=1, keepdim=True)
      self.conv_block1 = nn.Sequential(*list(net.features.children())[0:6])
      self.conv_block2 = nn.Sequential(*list(net.features.children())[7:13])
      self.conv_block3 = nn.Sequential(*list(net.features.children())[14:23])
      self.conv_block4 = nn.Sequential(*list(net.features.children())[24:33])
      # self.conv_block5 = nn.Sequential(*list(net.features.children())[34:43])
      self.pool = nn.AdaptiveAvgPool2d((1,1))
      self.dpt = None
      if dropout is not None:
          self.dpt = nn.Dropout(dropout)
      self.cls = nn.Linear(in_features=896, out_features=num_classes, bias=True)
      
      # initialize the attention blocks defined above
      self.attn1 = AttentionBlock(128, 512, 256, 4, normalize_attn=normalize_attn)
      self.attn2 = AttentionBlock(256, 512, 256, 2, normalize_attn=normalize_attn)
      
      
      self.reset_parameters(self.cls)
      self.reset_parameters(self.attn1)
      self.reset_parameters(self.attn2)
  def reset_parameters(self, module):
      for m in module.modules():
          if isinstance(m, nn.Conv2d):
              nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
              if m.bias is not None:
                  nn.init.constant_(m.bias, 0.)
          elif isinstance(m, nn.BatchNorm2d):
              nn.init.constant_(m.weight, 1.)
              nn.init.constant_(m.bias, 0.)
          elif isinstance(m, nn.Linear):
              nn.init.normal_(m.weight, 0., 0.01)
              nn.init.constant_(m.bias, 0.)
  def forward(self, x):
      block1 = self.conv_block1(x)       # /1
      pool1 = F.max_pool2d(block1, 2, 2) # /2
      block2 = self.conv_block2(pool1)   # /2
      pool2 = F.max_pool2d(block2, 2, 2) # /4
      block3 = self.conv_block3(pool2)   # /4
      pool3 = F.max_pool2d(block3, 2, 2) # /8
      block4 = self.conv_block4(pool3)   # /8
      pool4 = F.max_pool2d(block4, 2, 2) # /16
      # block5 = self.conv_block5(pool4)   # /16
      # pool5 = F.max_pool2d(block5, 2, 2) # /32
      # N, __, __, __ = pool4.size()
      
      g = self.pool(pool4).view(pool4.size(0), -1)
      a1, g1 = self.attn1(pool2, pool4)
      a2, g2 = self.attn2(pool3, pool4)
      g_hat = torch.cat((g,g1,g2), dim=1) # batch_size x C
      if self.dpt is not None:
          g_hat = self.dpt(g_hat)
      out = self.cls(g_hat)

      return [out, a1, a2]