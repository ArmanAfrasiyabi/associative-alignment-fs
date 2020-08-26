from torch.utils.data import DataLoader
import torch
import os
import numpy as np
from torch.autograd import Variable
import torch.optim
from backbones.utils import backboneSet
from torch.optim import  Adam
from methods.transferLearning_clfHeads import softMax, cosMax, arcMax
from torchvision.datasets import ImageFolder
from data.tl_dataFunctions import ar_transform
from backbones.utils import clear_temp
import shutil

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
def saveLoad_base_embedding(args, fs_approach):
    z_filename =  str(args.test_n_way)+'way_'+str(args.n_shot)+'shot_'+args.dataset+'_'+args.method+'_'+args.backbone  
    
    if os.path.isfile(args.benchmarks_dir+args.dataset+'/associative_alignment/'+z_filename):
        print('ar: Note, the base categories are PRE-saved in ', z_filename)
        embed_info = torch.load(args.benchmarks_dir+args.dataset+'/associative_alignment/'+z_filename)
        class_names = embed_info['class_names']
        z_embed = embed_info['z_embed']
        return class_names, z_embed
    else:
        args, net, file_name = backboneSet(args, fs_approach)  
        print('saving embedings of base categories, please wait...')
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, file_name))
        net.load_state_dict(checkpoint['state'])
        net.eval()
        
        data_path = args.benchmarks_dir+args.dataset+'/base/' 
#        if os.path.isdir(data_path+'temp'):
#            os.rmdir(data_path+'/temp')
        clear_temp(data_path)
        for _, names, _ in os.walk(data_path): break 
        os.mkdir(data_path+'/temp')
        z_embed = []
        for b in range(len(names)):
            shutil.move(data_path+names[b], data_path+'temp')
            transforms = ar_transform(args, aug=False)           
            dataset = ImageFolder(root=data_path+'temp/', transform = transforms)  
            dataLoader_i = DataLoader(dataset = dataset,
                                      batch_size = 500,
                                      num_workers = args.num_workers,
                                      shuffle = False, 
                                      drop_last = False)  
            for _, (x, y) in enumerate(dataLoader_i): break
            with torch.no_grad():
                z_embed.append(net(x))
            shutil.move(data_path+'temp/'+names[b], data_path)
        
        torch.save({'class_names':names, 'z_embed':z_embed}, 
                   args.benchmarks_dir+args.dataset+'/associative_alignment/'+z_filename) 
        print('finished!')
        return names, z_embed
    
def related_base_data(args, net, clf, z_embed, folders, device, n_B):
    clf_simMat = torch.zeros(len(z_embed), args.test_n_way)
    for b in range(len(z_embed)):
        zb = z_embed[b].to(device)
        logits = clf(zb)
        y_hat = np.argmax(logits.data.cpu().numpy(), axis=1)
        for w in range(args.test_n_way):
            clf_simMat[b, w] = y_hat.tolist().count(w)/zb.size(0)
            
    ### sorting clf_simMat 
    sort_simMat = []        
    for w in range(args.test_n_way): 
        m = clf_simMat[:, w]
        sort_simMat.append(sorted(range(len(m)), key=lambda k: m[k], reverse = True))  ## reverse in the case of cos
                
    related_classes_folders = []
    related_classes = []
    used_classes = []
    for tc in range(clf_simMat.size(1)):  
        class_counter = 0
        related_classes_i = []
        related_classes_folders_i = []
        for sc in range(clf_simMat.size(0)):
            if folders[sort_simMat[tc][sc]] not in used_classes:
                class_counter += 1
                related_classes_i.append(sort_simMat[tc][sc])
                related_classes_folders_i.append(folders[sort_simMat[tc][sc]])
                used_classes.append(folders[sort_simMat[tc][sc]]) 
                
            if class_counter==n_B: 
                related_classes.append(related_classes_i)
                related_classes_folders.append(related_classes_folders_i)
                break 
            
    print('finished 3!')
    ######################################   
    aug_x = []
    aug_y = []
    data_path = args.benchmarks_dir+args.dataset+'/base/' 
    for w in range(args.test_n_way):    #5 
        print('finished 4!', w)
        f_i = related_classes_folders[w] 
        for s in range(len(related_classes_folders_i)):  
            print('finished 5!', s)
            clear_temp(data_path) 
            os.mkdir(data_path+'/temp') 
            for b in range(n_B):
                shutil.move(data_path+f_i[b], data_path+'temp')
                transforms = ar_transform(args, aug=False)
                dataset = ImageFolder(root=data_path+'temp/', transform = transforms)  
                dataLoader_i = DataLoader(dataset = dataset,
                                          batch_size = 500,
                                          num_workers = args.num_workers,
                                          shuffle = False, 
                                          drop_last = False)  
                for _, (xr, _) in enumerate(dataLoader_i): break
                shutil.move(data_path+'temp/'+f_i[b], data_path)
                logits = clf(net(Variable(xr).to(device)))
                y_hat = torch.tensor(np.argmax(logits.data.cpu().numpy(), axis=1))
                x_aug_w_i = xr[y_hat==w] 
                y_aug_w_i = torch.zeros(x_aug_w_i.size(0)).long()+w 
                
                if s==0:
                    x_aug_w = x_aug_w_i
                    y_aug_w = y_aug_w_i
                else:
                    x_aug_w = torch.cat((x_aug_w, x_aug_w_i), dim=0)  
                    y_aug_w = torch.cat((y_aug_w, y_aug_w_i), dim=0)  
                    
        aug_x.append(x_aug_w) 
        aug_y.append(y_aug_w)   
        
    return [aug_x, aug_y]


def ar_rs_episode(aug_data, aug_size, device):
    [xa0, xa1, xa2, xa3, xa4] = aug_data[0]
    [ya0, ya1, ya2, ya3, ya4] = aug_data[1]
    
    min_aug_size = min(xa0.size(0), xa1.size(0), xa2.size(0), xa3.size(0), xa4.size(0))
    if aug_size>min_aug_size:
        aug_size = min_aug_size
    
    select_mask0 = torch.tensor(np.random.choice(xa0.size(0), 
                                                 size=aug_size, 
                                                 replace=False))
    select_mask1 = torch.tensor(np.random.choice(xa1.size(0), 
                                                 size=aug_size, 
                                                 replace=False))
    select_mask2 = torch.tensor(np.random.choice(xa2.size(0), 
                                                 size=aug_size, 
                                                 replace=False))
    select_mask3 = torch.tensor(np.random.choice(xa3.size(0), 
                                                 size=aug_size, 
                                                 replace=False))
    select_mask4 = torch.tensor(np.random.choice(xa4.size(0), 
                                                 size=aug_size, 
                                                 replace=False))
    xa_i = torch.cat((xa0[select_mask0], 
                      xa1[select_mask1],
                      xa2[select_mask2], 
                      xa3[select_mask3],
                      xa4[select_mask4]), dim=0)
                
    ya_i = torch.cat((ya0[select_mask0], 
                      ya1[select_mask1],
                      ya2[select_mask2], 
                      ya3[select_mask3],
                      ya4[select_mask4]), dim=0)
                    
    xa_i = Variable(xa_i).to(device)             
    ya_i = Variable(ya_i).to(device)  
    
    return xa_i, ya_i, aug_size











