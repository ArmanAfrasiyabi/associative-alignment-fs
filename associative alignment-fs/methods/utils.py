#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 02:27:29 2020

@author: ari
"""
from methods.transferLearning_clfHeads import softMax, cosMax, arcMax
from torch.optim import  Adam
import torch


def clf_optimizer(args, net, device, frozen_net, s = 15, m = 0.01):
    if frozen_net: s=5
    if args.method == 'softMax':
        clf = softMax(args.out_dim, args.test_n_way).to(device)
    elif args.method == 'cosMax':
        clf = cosMax(args.out_dim, args.test_n_way, s).to(device)
    elif args.method == 'arcMax':
        clf = arcMax(args.out_dim, args.test_n_way, s, m).to(device)
    if frozen_net:  
        optimizer = torch.optim.Adam(clf.parameters(), lr = args.lr)  
    else:
        optimizer = Adam([{'params': net.parameters()}, 
                          {'params': clf.parameters()}], 
                          lr = args.lr) 
    return clf, optimizer



