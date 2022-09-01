import torch, pdb
import torch.nn as nn

from model.sincdsnet import  SincDSNet
from model.visualEncoder     import visualFrontend, visualTCN, visualConv1D
from .vit import Transformer

class FSD(nn.Module):
    def __init__(self):
        super(FSD, self).__init__()
        # Visual Temporal Encoder from TalkNet
        self.visualFrontend  = visualFrontend() # Visual Frontend 
        self.visualTCN       = visualTCN()      # Visual Temporal Network TCN
        self.visualConv1D    = visualConv1D()   # Visual Temporal Network Conv1d

        # Audio Temporal Encoder from ASDNet
        self.audioEncoder  = SincDSNet()
        
        self.selfAV = Transformer(dim = 256, depth=4, heads=8, dim_head=32, mlp_dim=512, dropout=0.1)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape  
        x = x.view(B*T, 1, 1, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        x = self.visualFrontend(x)
        x = x.view(B, T, 512)        
        x = x.transpose(1,2)     
        x = self.visualTCN(x)
        x = self.visualConv1D(x)
        x = x.transpose(1,2)
        return x

    def forward_audio_frontend(self, x):    
        x = x.unsqueeze(2)
        x = self.audioEncoder(x)
        return x

    def forward_audio_visual_backend(self, x1, x2): 
        x = torch.cat((x1,x2), 2)    
        x = self.selfAV(x)
        x = torch.reshape(x, (-1, 256))
        return x    

    def forward_audio_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

    def forward_visual_backend(self,x):
        x = torch.reshape(x, (-1, 128))
        return x

