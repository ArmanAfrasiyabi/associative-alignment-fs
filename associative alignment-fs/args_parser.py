import argparse 
def args_parser(fs_approach):
    args = argparse.ArgumentParser(description= 'experiment mode %s' %(fs_approach))
    args.add_argument('--dataset', default='miniImagenet', help='miniImagenet|CUB|tieredImageNet|FC100')
    args.add_argument('--backbone', default='Conv4', help='Conv4|ResNet12|ResNet18|WideResNet')   
    
    args.add_argument('--seed', default = 10, help = 'the seed used for training')
    args.add_argument('--num_workers', default = 12, help = 'the number of workers')
    
    args.add_argument('--data_aug', action='store_true', help='perform data augmentation or not during training')   
    args.add_argument('--img_size', default=84, help = 'input size of the backbone: 84, 224, 80  for miniImagenet, ResNet18, WideResNet')
    
    args.add_argument('--n_epoch', default=600, type=int, help ='the last epoch for stop')
    args.add_argument('--lr', default = 0.001, help = '0.001 the learning rate')
    
    args.add_argument('--benchmarks_dir', default = '/home/ari/Documents/few_shot_lablatory/fs_benchmarks/', help = 'the directory which the benchmarks are strored')
    args.add_argument('--checkpoint_dir', default = './results/models/', help = 'the directory for saving the best_model')
    
    args.add_argument('--test_n_way'  , default=5, type=int,  help='number of classes in each meta-validation')  
    args.add_argument('--n_shot'      , default=1, type=int,  help='support size in the episodic training literature')  
    args.add_argument('--n_query'     , default=50,type=int,  help='pretented unlabled data for loss calculation in meta-learning')
    
    args.add_argument('--testing_epochs', default=10, help = 'the number of epoch for measuring the accuracy') 
    args.add_argument('--n_episodes', default=600, help = 'the number of episodes for measuring the accuracy') 
    
    args.add_argument('--n_way' , default=5, type=int,  help='number of classes in each meta-training') 
    args.add_argument('--n_B' , default=10, type=int,  help='number of related base classes') 
    
    if fs_approach == 'meta-learning':
        args.add_argument('--method'      , default='ProtoNet', help='MatchingNet|ProtoNet|RelationNet{_softmax}')
        args.add_argument('--train_n_way' , default=5, type=int,  help='number of classes in each meta-training')   
        args.add_argument('--n_support' , default=5, type=int,  help='number of classes in each meta-training')  
    
    elif fs_approach == 'transfer-learning': 
        args.add_argument('--over_fineTune', type = bool, default = False, help = 'perform over fine-tunining')
        args.add_argument('--n_base_class', default = 64, help =  'number of base categories')
        args.add_argument('--method'      , default='arcMax', help='softMax|cosMax|arcMax')
        args.add_argument('--batch_size', default = 64, type=int,  help='64 the batch size during base-training')  
        args.add_argument('--ft_n_epoch', default = 200, type=int,  help='the number of testing epochs during the novel and validation fine-tunining')  
    else:
       raise ValueError('unknown few-shot approach!')
       
    return args.parse_args()


    
    
