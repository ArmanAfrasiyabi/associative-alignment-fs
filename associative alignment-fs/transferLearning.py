from __future__ import print_function 
import os
import torch
import numpy as np
from args_parser import args_parser 
from backbones.utils import clear_temp
from backbones.utils import backboneSet    
from data.tl_dataFunctions import ar_base_DataLaoder  
from data.ml_dataFunctions import SetDataManager
from methods.transferLearningFuns import transferLearningFuns

def meta_training(args, model, file_name, resume):    
    clear_temp(args.benchmarks_dir + args.dataset+'/base/')    
    print('transfer training...')
    checkpoint_dir = os.path.join(args.checkpoint_dir, file_name)  
    max_acc = 0 
    first_epoch = 0
    if resume and os.path.isfile(checkpoint_dir): 
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name))
        first_epoch = checkpoint['epoch']+1
        model.net.load_state_dict(checkpoint['state'])
        max_acc = checkpoint['max_acc'] 
        model.optimizer = checkpoint['optimizer'] 
        model.base_clf = checkpoint['base_clf'] 
        model.base_clf.train()
        model.net.train() 
        print('resume: up to now the best model has:', str(max_acc), 'accuracy at ep.',first_epoch,'!')
        
        
    trLoader = ar_base_DataLaoder(args, aug=True, shuffle=True)
    val_file = args.benchmarks_dir + args.dataset + '/val.json'
    val_datamgr = SetDataManager(args.img_size, args.test_n_way, args.n_shot, args.n_query, n_episodes = 400)
    vaLoader  = val_datamgr.get_data_loader(val_file, aug = False)    
    for epoch in range(first_epoch, args.n_epoch):
        model.net.train()
        trLoss = model.train_loop(trLoader)
        model.net.eval() 
        vaAcc, vaConf_interval = model.test_loop(vaLoader, args.test_n_way)  
        print_txt = 'epoch %d trLoss: %4.2f%%  ||| vaAcc: %4.2f%% +- %4.2f%% |||'
        if vaAcc > max_acc :  
            max_acc = vaAcc 
            torch.save({'epoch':epoch, 'state':model.net.state_dict(), 
                        'max_acc':max_acc, 
                        'base_clf':model.base_clf,
                        'optimizer':model.optimizer}, 
                checkpoint_dir) 
            print_txt = print_txt + ' update...' 
#        if epoch==20: 
#            val_datamgr = SetDataManager(args.img_size, args.test_n_way, args.n_shot, args.n_query, n_episodes = 400) 
#            vaLoader  = val_datamgr.get_data_loader(val_file, aug = False)    # to fasten the training; if turn off then increase n_episodes to 400 in line 27
        print(print_txt %(epoch,  trLoss, vaAcc, vaConf_interval))  
    
def meta_testing(args, model, file_name, partition='novel'):       
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name))
    model.net.load_state_dict(checkpoint['state'])
    best_epoch = checkpoint['epoch']+1
    print('transfer testing... best model at epoch ', best_epoch)
    novel_file = args.benchmarks_dir + args.dataset + '/'+partition+'.json' 
    novel_datamgr = SetDataManager(args.img_size, args.test_n_way, args.n_shot, args.n_query, n_episodes = 600)
    noLoader = novel_datamgr.get_data_loader(novel_file, aug = False)  
    model.net.eval()
    acc, confInt = [], []
    for epoch in range(args.testing_epochs): 
        noAcc, noConf_interval = model.test_loop(noLoader, args.test_n_way)   
        acc.append(noAcc)
        confInt.append(noConf_interval)
        print('meta-testing acc: %4.2f%% +- %4.2f%%' %(np.average(acc), np.average(confInt)))    
  
 
if __name__ == '__main__':
    fs_approach = 'transfer-learning'
    args = args_parser(fs_approach)
    args, net, file_name = backboneSet(args, fs_approach) 
    model = transferLearningFuns(args, net, args.n_base_class) 
    meta_training(args, model, file_name, resume=False) 
    meta_testing(args, model, file_name)
    
    
    
    
    
# 1-shot
# transfer testing... best model at epoch  267
# 1-shot: meta-testing acc: 51.37% +- 0.69% 
    
     
# 5-shot
# transfer testing... best model at epoch  336
# meta-testing acc: 69.13% +- 0.59% 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


        
        
        
        
        
        
    
    
    
    
  