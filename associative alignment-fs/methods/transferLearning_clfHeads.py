import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):        
        return x.view(x.size(0), -1)
    
    
 
class arcMax(nn.Module):
    def __init__(self, in_dim, n_class, s, m):
        super(arcMax, self).__init__()
        self.easy_margin = True
        self.criterion = torch.nn.CrossEntropyLoss() 
        self.weight = Parameter(torch.Tensor(n_class, in_dim)) 
        nn.init.xavier_uniform_(self.weight)   
        self.s = s
        self.margin = m
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        
    def forward(self, z):                     ### x.shape = [55, 64], y.shape = [55]
        norm_w = F.normalize(self.weight) #+ (self.weight*(epoch+1)*0.001)
        norm_z = F.normalize(z)
        ## cos(theta):  ar: algebraicly: a.b = norm(a)norm(b) cos(theta)  
        cosine_theta = F.linear(norm_z, norm_w)      ### [55, 10]
        return cosine_theta
    
    def loss(self, cosine_theta, y):
        ## cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        sine_theta = torch.sqrt(1.0 - torch.pow(cosine_theta, 2))              ### [55, 10] 
        cons_theta_m = cosine_theta * self.cos_m - sine_theta * self.sin_m     ### [55, 10] 
        if self.easy_margin:
            cons_theta_m = torch.where(cosine_theta > 0, cons_theta_m, cosine_theta)
        else:
            cons_theta_m = torch.where((cosine_theta - self.th) > 0, cons_theta_m, cosine_theta - self.mm) 
        y_1Hot = torch.zeros_like(cons_theta_m)         ### [25, 5]
        y_1Hot.scatter_(1, y.view(-1, 1), 1)            ### [25, 5]
        logits = (y_1Hot * cons_theta_m) + ((1.0 - y_1Hot) * cosine_theta)
        logits = self.s * logits
        loss = self.criterion(logits, y)
        return loss



    
class cosMax(nn.Module):
    def __init__(self, in_dim, n_class, s):
        super(cosMax, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss() 
        self.weight = Parameter(torch.Tensor(n_class, in_dim)) 
        nn.init.xavier_uniform_(self.weight)   
        self.s = s
        
    def forward(self, z):
        norm_w = F.normalize(self.weight) #+ (self.weight*(epoch+1)*0.001)
        norm_z = F.normalize(z)
        ## cos(theta):  ar: algebraicly: a.b = norm(a)norm(b) cos(theta)  
        cosine_theta = F.linear(norm_z, norm_w)      ### [55, 10]
        return self.s * (cosine_theta) 
    
    def loss(self, cosine_theta, y):
        
        return self.criterion(cosine_theta, y)


class softMax(nn.Module):
    def __init__(self, in_dim, n_class):
        super(softMax, self).__init__()        
        self.out = torch.nn.Linear(in_dim, n_class)
        self.criterion = torch.nn.CrossEntropyLoss()
        
    def forward(self, z): 
        logits = self.out(z)
        return logits
    
    def loss(self, logits, y):
        return self.criterion(logits, y)
