import torch
import os
import numpy as np
import torch.optim
from torch.optim import  Adam
from args_parser import args_parser
from torch.autograd import Variable
from methods.utils import clf_optimizer
from backbones.utils import backboneSet, clear_temp 
from data.ml_dataFunctions import SetDataManager as EpisodicDataManager
from methods.transferLearningFuns import device_kwargs 
from data.related_base_detector_tl_fast import related_base_data, ar_rs_episode, saveLoad_base_embedding
#from data.related_base_detector_tl import related_base_data, ar_rs_episode, saveLoad_base_embedding
from methods.alignments import centroid_aligner 
#from data.related_source_net import ar_rs_DataLaoder

def data_sep(x):
    x = Variable(x).to(device)
    x_support   = x[:, :args.n_shot, :,:,:].contiguous()
    x_support = x_support.view(args.test_n_way * args.n_shot, *x.size()[2:]) 
    y_support = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_shot))
    y_support = Variable(y_support.to(device))
    x_query = x[:, args.n_shot:, :,:,:].contiguous()
    x_query = x_query.view(args.test_n_way * args.n_query, *x.size()[2:]) 
    y_query = torch.from_numpy(np.repeat(range(args.test_n_way), args.n_query))
    y_query = Variable(y_query.cuda())  
    return [x_support, y_support, x_query, y_query] 

 
def loss_bp(Net, dnn_params, x, y):
    [clf, optimizer] = dnn_params
    loss = clf.loss(clf(Net(x)), y)  
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  
    
def target_test(net, clf, xt_q, yt_q):
    net.eval()
    with torch.no_grad():
        logits = clf(net(xt_q))
        y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
    return np.mean((y_hat == yt_q.data.cpu().numpy()).astype(int))  
 
### ################################
def euclidean_alignment(args, net, x, device, aug_size= 50, over_ft_iter = 150):
    [x_support, y_support, x_query, y_query]  = data_sep(x)
    net.eval()
    clf, clf_optimizer_a = clf_optimizer(args, net, device, s=5, frozen_net=True, m=0.001)
    with torch.no_grad():
        z_support  = net(x_support)
    for epoch in range(args.ft_n_epoch):
        loss = clf.loss(clf(z_support), y_support) 
        clf_optimizer_a.zero_grad()
        loss.backward()
        clf_optimizer_a.step() 
        
    logits = clf(net(x_query))
    y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)  
    ft_res = np.mean((y_hat == y_query.data.cpu().numpy()).astype(int))*100
     
    dm_optimizer = Adam(net.parameters(), lr = args.lr) 
    clf_optimizer_t = Adam([{'params': net.parameters()}, 
                            {'params': clf.parameters()}], lr = args.lr)   
    clf.s = 15
    aug_data = related_base_data(args, net, clf, z_embed, folders, device, args.n_B)   
#    aug_data = related_base_data(args, net, clf, z_embed, folders, device, args.n_B) 
    for _ in range(over_ft_iter): 
        net.train()
        xa, ya, support_size = ar_rs_episode(aug_data, aug_size, device=device)
        dm_loss = centroid_aligner(args, net, data = [x_support, xa],  
                                   support_size = args.n_shot, 
                                   query_size = support_size, 
                                   device = device)
        dm_optimizer.zero_grad()
        dm_loss.backward()
        dm_optimizer.step() 
        
        net.eval()
        loss_bp(net, [clf, clf_optimizer_a], xa, ya)
        net.train()
        loss_bp(net, [clf, clf_optimizer_t], x_support, y_support) 
    aa_res = target_test(net.eval(), clf, x_query, y_query)*100   
    return ft_res, aa_res

if __name__=='__main__':
    np.random.seed(10) 
    fs_approach = 'transfer-learning'
    args = args_parser(fs_approach)
    device = device_kwargs(args)
    clear_temp(args.benchmarks_dir + args.dataset+'/base/')     
    
    novel_file = args.benchmarks_dir + args.dataset + '/novel.json'   
    novel_datamgr = EpisodicDataManager(args.img_size, args.n_way, args.n_shot, args.n_query, n_episodes = 600)
    noLoader = novel_datamgr.get_data_loader(novel_file, aug = False)    
     
    args, net, file_name = backboneSet(args, fs_approach)  
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name))
    
    net.load_state_dict(checkpoint['state'])
    net.to(device)
    net.eval()
    folders, z_embed = saveLoad_base_embedding(args, net, fs_approach, device)
#    folders, z_embed = saveLoad_base_embedding(args, fs_approach)
    ft_acc, aa_acc = [], [] 
    print('-------------------------------------------------------------------------- ')
    print('-------- fine tuning (f_tune) vs. associative alignment (a_align) -------- ')
    print('-------------------------------------------------------------------------- ')
    for i, (x, _) in enumerate(noLoader):
        args, net, file_name = backboneSet(args, fs_approach)  
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name))
        net.load_state_dict(checkpoint['state'])
        net.to(device)
        
        ft_acc_i, aa_acc_i = euclidean_alignment(args, net, x, device)
        
        ft_acc.append(ft_acc_i) 
        aa_acc.append(aa_acc_i)
        print('ep.%d  f_tune. %2.2f%%  a_align. %2.2f%% ||| diff: %2.2f%%' %(i+1, np.average(ft_acc), np.average(aa_acc),
                                                                                   np.average(aa_acc)-np.average(ft_acc)))  
    print('-------------------------------------------------------------------------- ')
    acc_std  = np.std(aa_acc)
    conf_interval_aa = 1.96* acc_std/np.sqrt(i+1)
    print('----------- associative alignment: finished!')
    print('ep.%3.f  acc: %4.2f%% +- %2.2f%% ' %(i+1, np.average(aa_acc), conf_interval_aa))  
    print('-------------------------------------------------------------------------- ')    

