import torch.nn as nn
import math


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)

def layerInitializer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


## convolution block [C: conv; B: batchNormalization, R: relu, P: maxpooling]
class ConvBlock_CBRP(nn.Module):
    def __init__(self, in_dim, hid_dim, padding = 1):
        super(ConvBlock_CBRP, self).__init__()
        self.blocks = [nn.Conv2d(in_dim, hid_dim, kernel_size = 3, padding = padding),
                          nn.BatchNorm2d(hid_dim),  
                          nn.ReLU(), ## inplace = True
                          nn.MaxPool2d(2)]
        
        for layer in self.blocks:
            layerInitializer(layer)
            
        self.convBlock = nn.Sequential(*self.blocks)
        
    def forward(self, x):
        return self.convBlock(x)
    
## convolution block [C: conv; B: batchNormalization, R: relu, P: maxpooling]
class ConvBlock_CBR(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(ConvBlock_CBR, self).__init__()
        self.blocks = [nn.Conv2d(in_dim, hid_dim, kernel_size = 3, padding = 1),
                          nn.BatchNorm2d(hid_dim),  
                          nn.ReLU()]
        
        for layer in self.blocks:
            layerInitializer(layer)
            
        self.convBlock = nn.Sequential(*self.blocks)
        
    def forward(self, x):
        return self.convBlock(x)



## Building the shallow network
class Conv4Net(nn.Module):
    def __init__(self, hid_dim=64):
        super (Conv4Net, self).__init__()
        nConvBlock = []
        nConvBlock.append(ConvBlock_CBRP(3, hid_dim))
        nConvBlock.append(ConvBlock_CBRP(hid_dim, hid_dim))
        nConvBlock.append(ConvBlock_CBRP(hid_dim, hid_dim))
        nConvBlock.append(ConvBlock_CBRP(hid_dim, hid_dim))
        
        self.nConvBlock = nn.Sequential(*nConvBlock)

    def forward(self, x):
        out = self.nConvBlock(x)
        return out.view(out.size(0), -1)
        
        
## Building the shallow network
class Conv4Net_RN(nn.Module):
    def __init__(self, hid_dim=64):
        super (Conv4Net_RN, self).__init__()
        nConvBlock = []
        nConvBlock.append(ConvBlock_CBRP(3, hid_dim, padding = 0))
        nConvBlock.append(ConvBlock_CBRP(hid_dim, hid_dim, padding = 0))
        nConvBlock.append(ConvBlock_CBR(hid_dim, hid_dim))
        nConvBlock.append(ConvBlock_CBR(hid_dim, hid_dim))
        
        self.nConvBlock = nn.Sequential(*nConvBlock)

    def forward(self, x):
        return self.nConvBlock(x)
         
            
class Discriminator(nn.Module):
    def __init__(self, in_dim, hid_dim, cond_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(in_dim+cond_size, hid_dim)
        self.relu = nn.ReLU() 
        self.out = nn.Linear(hid_dim, 1)

    def forward(self, x):
        z = self.relu(self.fc(x))
        return self.out(z)            
            
            
            














          
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
