import torch
from torch import nn
import torchaudio
from .config import SAMPLE_RATE

class MelFrontEnd(nn.Module):
 def __init__(self):super().__init__();self.mel=torchaudio.transforms.MelSpectrogram(SAMPLE_RATE,n_fft=512,win_length=400,hop_length=160,n_mels=64)
 def forward(self,x):
  z=torch.log(self.mel(x).clamp_min(1e-6));return (z-z.mean((-2,-1),keepdim=True))/(z.std((-2,-1),keepdim=True)+1e-5)
class MelCNN(nn.Module):
 name="Mel-CNN"
 def __init__(self,classes=4):
  super().__init__();self.front=MelFrontEnd();self.net=nn.Sequential(nn.Conv2d(1,24,3,padding=1),nn.BatchNorm2d(24),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(24,48,3,padding=1),nn.BatchNorm2d(48),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(48,96,3,padding=1),nn.BatchNorm2d(96),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)));self.drop=nn.Dropout(.25);self.fc=nn.Linear(96,classes)
 def encode(self,x):return self.net(self.front(x).unsqueeze(1)).flatten(1)
 def forward(self,x,prosody=None):return self.fc(self.drop(self.encode(x)))
class RawWaveCNN(nn.Module):
 name="RawWave-CNN"
 def __init__(self,classes=4):
  super().__init__();ch=[1,16,32,64,96,128];b=[]
  for a,c in zip(ch[:-1],ch[1:]):b += [nn.Conv1d(a,c,15 if a==1 else 7,stride=4,padding=7 if a==1 else 3),nn.BatchNorm1d(c),nn.ReLU()]
  self.net=nn.Sequential(*b,nn.AdaptiveAvgPool1d(1));self.head=nn.Sequential(nn.Flatten(),nn.Dropout(.25),nn.Linear(128,classes))
 def forward(self,x,prosody=None):return self.head(self.net(x.unsqueeze(1)))
class ProsodyFusion(nn.Module):
 name="Prosody-Fusion"
 def __init__(self,classes=4):
  super().__init__();self.backbone=MelCNN();self.prosody=nn.Sequential(nn.Linear(4,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,32),nn.ReLU());self.head=nn.Sequential(nn.Dropout(.25),nn.Linear(128,classes))
 def forward(self,x,prosody):return self.head(torch.cat([self.backbone.encode(x),self.prosody(prosody)],1))
MODEL_CLASSES={"mel_cnn":MelCNN,"raw_cnn":RawWaveCNN,"prosody_fusion":ProsodyFusion}
