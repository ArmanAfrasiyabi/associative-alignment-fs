import os 
import torch 
import numpy as np
import torch.optim
from backbones.utils import backboneSet
from data.ml_dataFunctions import SetDataManager as EpisodicDataManager
from methods.metaLearningFuns import metaLearningFuns 
from args_parser import args_parser
from backbones.utils import clear_temp

def meta_training(args, net, file_name, resume):   
    print('meta training...')
    clear_temp(args.benchmarks_dir + args.dataset+'/base/')    
    checkpoint_dir = os.path.join(args.checkpoint_dir, file_name)  
    optimizer = torch.optim.Adam(net.parameters(),  lr = args.lr) 
    first_epoch = 0
    max_acc = 0  
    if resume and os.path.isfile(checkpoint_dir):
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name))
        first_epoch = checkpoint['epoch']+1
        net.load_state_dict(checkpoint['state'])
        max_acc = checkpoint['max_acc'] 
        optimizer = checkpoint['optimizer']     
    base_file = args.benchmarks_dir + args.dataset + '/base.json'  
    base_datamgr= EpisodicDataManager(args, args.train_n_way) 
    base_loader = base_datamgr.get_data_loader(base_file, aug = args.data_aug)  
    val_file = args.benchmarks_dir + args.dataset + '/val.json'
    val_datamgr = EpisodicDataManager(args, args.test_n_way) 
    val_loader  = val_datamgr.get_data_loader(val_file, aug = False)     
    net = net.cuda()
    for epoch in range(first_epoch, args.n_epoch):
        net.train() 
        trLoss = net.train_loop(epoch, base_loader, optimizer)   
        net.eval()
        vaAcc, vaConf_interval = net.test_loop(val_loader, args.test_n_way)  
        print_txt = 'epoch %d trLoss: %4.2f%%  ||| vaAcc: %4.2f%% +- %4.2f%% |||'
        if vaAcc > max_acc :  
            max_acc = vaAcc 
            torch.save({'epoch':epoch, 'state':net.state_dict(), 
                        'max_acc':max_acc, 'optimizer':optimizer}, checkpoint_dir) 
            print_txt = print_txt + ' update...'
        print(print_txt %(epoch,  trLoss, vaAcc, vaConf_interval))    
        
def meta_testing(args, net, file_name, partition='novel'):       
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name))
    best_epoch = checkpoint['epoch']+1
    print('meta testing... best model at epoch ', best_epoch)
    net.load_state_dict(checkpoint['state'])
    novel_file = args.benchmarks_dir + args.dataset + '/'+partition+'.json' 
    novel_datamgr = EpisodicDataManager(args, args.test_n_way, args.n_episodes)
    novel_loader = novel_datamgr.get_data_loader(novel_file, aug = False)  
    net = net.cuda()
    net.eval()
    acc, confInt = [], []
    for epoch in range(args.testing_epochs): 
        noAcc, noConf_interval = net.test_loop(novel_loader, args.test_n_way)  
        acc.append(noAcc)
        confInt.append(noConf_interval)
        print('acc: %4.2f%% +- %4.2f%%' %(np.average(acc), np.average(confInt)))
if __name__=='__main__':
    np.random.seed(10) 
    fs_approach = 'meta-learning'
    args = args_parser(fs_approach)
    args, net, file_name = backboneSet(args, fs_approach)
    net_method = metaLearningFuns(args, net)
    meta_training(args, net_method, file_name, resume=True)
    meta_testing(args, net_method, file_name) 
    
    
    
    
    
    
               
            
 
            
            
            
            
            
            
            
            
            
            
            
            
            
            
 
   
