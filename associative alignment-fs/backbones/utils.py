import torchvision.models as models
from backbones.shallow_backbone import Conv4Net, Conv4Net_RN, Flatten 
from torch import nn
import shutil
import torch 
import os

 
def clear_temp(data_path):
    if os.path.isdir(data_path+'temp'):
        for _, name_temp, _ in os.walk(data_path+'temp'): break
        if name_temp != []:
            shutil.move(data_path+'temp/'+name_temp[0], data_path)
        os.rmdir(data_path+'/temp')

def device_kwargs(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    return device

def backboneSet(args, fs_approach):
    if args.method in ['RelationNet', 'RelationNet_softmax']:
        args.out_dim =  [64, 19, 19]
        net = Conv4Net_RN()
        args.img_size = 84 
    else: 
        net = Conv4Net() 
        args.out_dim = 1600
        args.img_size = 84  
        
    if args.dataset=='miniImagenet':
        args.n_base_class = 64
    elif args.dataset=='CUB':
        args.n_base_class = 100    
    else:
        raise "ar: sepcify the number of base categories!"    
        
    file_name = str(args.test_n_way)+'way_'+str(args.n_shot)+'shot_'+args.dataset+'_'+args.method+'_'+args.backbone+'_bestModel.tar'     
     
    return args, net, file_name


