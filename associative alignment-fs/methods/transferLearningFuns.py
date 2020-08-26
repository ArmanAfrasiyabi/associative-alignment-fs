#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 02:31:23 2020

@author: ari
"""

from __future__ import print_function 
from methods.transferLearning_clfHeads import softMax, cosMax, arcMax
from torch.autograd import Variable
from torch.optim import  Adam
import numpy as np
import torch 
import torch.nn as nn
import torchvision.models as models
from backbones.shallow_backbone import Conv4Net, Flatten 
from backbones.utils import device_kwargs
 
def clf_fun(self, n_class, device, s = 20, m = 0.01):
    if self.method == 'softMax':
        clf = softMax(self.out_dim, n_class).to(device)
    elif self.method == 'cosMax':
        clf = cosMax(self.out_dim, n_class, s).to(device)
    elif self.method == 'arcMax':
        clf = arcMax(self.out_dim, n_class, s, m).to(device) 
        
    return clf

class transferLearningFuns(nn.Module):
    def __init__(self, args, net, n_class):
        super(transferLearningFuns, self).__init__()
        
        self.device = device_kwargs(args) 
        self.method = args.method
        self.lr = args.lr
        self.backbone = args.backbone
        
        self.n_epoch = args.n_epoch
        self.n_class = n_class
        self.n_support = args.n_shot
        self.n_query = args.n_query
        
        self.out_dim = args.out_dim
        self.lr = args.lr
        self.net = net.to(self.device)
        self.over_fineTune = args.over_fineTune
        
        self.ft_n_epoch = args.ft_n_epoch 
        self.n_way = args.n_way
        
        self.base_clf = clf_fun(self, self.n_class, self.device)
        self.optimizer = Adam([{'params': self.net.parameters()}, 
                                {'params': self.base_clf.parameters()}], 
                                  lr = self.lr) 
        
    def accuracy_fun_tl(self, data_loader):        # this is typical batch based testing (should be only used for base categories) 
        Acc = 0
        self.net.eval()  
        with torch.no_grad(): 
            for x, y in data_loader:
                x, y = Variable(x).to(self.device), Variable(y).to(self.device)
                logits = self.clf(self.net(x))
                y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
                Acc += np.mean((y_hat == y.data.cpu().numpy()).astype(int))
        return Acc.item()/len(data_loader) 
    
    def accuracy_fun(self, x, n_way):            
        novel_clf = clf_fun(self, self.n_way, self.device, s=5) 
        novel_optimizer = torch.optim.Adam(novel_clf.parameters(), lr = self.lr)   
        x_support   = x[:, :self.n_support, :,:,:].contiguous()
        x_support = x_support.view(n_way * self.n_support, *x.size()[2:]) 
        y_support = torch.from_numpy(np.repeat(range(n_way), self.n_support))
        y_support = Variable(y_support.to(self.device))
        
        
        with torch.no_grad():
            z_support  = self.net(x_support)
        for epoch in range(self.ft_n_epoch):
            loss = novel_clf.loss(novel_clf(z_support), y_support) 
            novel_optimizer.zero_grad()
            loss.backward()
            novel_optimizer.step()    
            
        x_query = x[:, self.n_support:, :,:,:].contiguous()
        x_query = x_query.view(n_way * self.n_query, *x.size()[2:]) 
        y_query = torch.from_numpy(np.repeat(range(n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        logits = novel_clf(self.net(x_query))
        y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)  
        return np.mean((y_hat == y_query.data.cpu().numpy()).astype(int))*100
        
    # note: this is typical/batch based training
    def train_loop(self, trainLoader): 
        self.net.train() 
        loss_sum = 0
        for i, (x, y) in enumerate(trainLoader):
            x, y = Variable(x).to(self.device), Variable(y).to(self.device)
            loss = self.base_clf.loss(self.base_clf(self.net(x)), y) 
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item()     
        return loss_sum/len(trainLoader)
    
    
    # note this is episodic testing 
    def test_loop(self, test_loader, n_way):
        acc_all = []
        self.net.eval()  
        for i, (x,_) in enumerate(test_loader):
            x = Variable(x).to(self.device)
            self.n_query = x.size(1) - self.n_support
            acc_all.append(self.accuracy_fun(x, n_way)) 
            #print(np.mean(acc_all))
             
        acc_all  = np.asarray(acc_all) 
        teAcc = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        conf_interval = 1.96* acc_std/np.sqrt(len(test_loader))
        return teAcc, conf_interval