import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from methods.protonet import pn_loss
from methods.matchingnet import mn_loss, FullyContextualEmbedding
from methods.relationnet import rn_loss, RelationModule
from backbones.utils import device_kwargs

class metaLearningFuns(nn.Module):
    def __init__(self, args, net):   
        super(metaLearningFuns, self).__init__()
        self.n_way = args.train_n_way
        self.n_support = args.n_shot
        self.n_query = args.n_query              
        self.net = net              
        self.out_dim = args.out_dim 
        self.device = device_kwargs(args) 
         
        if args.method == 'ProtoNet':
            self.score_loss = pn_loss
            self.loss_fn = nn.CrossEntropyLoss() 

        if args.method == 'MatchingNet':
            self.score_loss = mn_loss
            self.loss_fn = nn.NLLLoss()
            self.FCE = FullyContextualEmbedding(self.out_dim)
            self.G_encoder = nn.LSTM(self.out_dim, self.out_dim, 1, batch_first=True, bidirectional=True)
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax(dim=1)
            
        self.loss_type = 'mse'  #'softmax'# 'mse'    
        if args.method == 'RelationNet':
            self.score_loss = rn_loss
            self.relation_module = RelationModule(self.out_dim , 8, self.loss_type) #relation net features are not pooled, so self.feat_dim is [dim, w, h] 
            if self.loss_type == 'mse':
                self.loss_fn = nn.MSELoss()  
            else:
                self.loss_fn = nn.CrossEntropyLoss()
                 
    def embedding_fun(self, x, n_way):
        x           = Variable(x.cuda()).contiguous()
        x           = x.view(n_way * (self.n_support + self.n_query), *x.size()[2:]) 
        
        z_all       = self.net.forward(x)
        z_all       = z_all.view(n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]
        return z_support, z_query
    
    def accuracy_fun(self, x, n_way):    
        z_support, z_query = self.embedding_fun(x, n_way)
        scores = self.score_loss(self, z_support, z_query, loss_fn = None, loss = False, score = True, rb=None)
                #self, z_support, z_query, score = True)
        y_query = np.repeat(range(n_way), self.n_query )

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        
        return float(top1_correct)/ len(y_query)*100 

    def train_loop(self, epoch, train_loader, optimizer): 
        loss_sum=0
        for i, (x,_ ) in enumerate(train_loader):           ## x.shape = [30, 7, 3, 224, 224]
            self.n_query = x.size(1) - self.n_support           
            optimizer.zero_grad()
            z_support, z_query = self.embedding_fun(x, self.n_way)
            loss = self.score_loss(self, z_support, z_query, 
                                   loss_fn = self.loss_fn, loss = True)           
            loss.backward()
            optimizer.step()
            loss_sum = loss_sum+loss.item()  
        return loss_sum/float(i+1)
                         
    def test_loop(self, test_loader, n_way):
        acc_all = []
        iter_num = len(test_loader) 
        for i, (x,_) in enumerate(test_loader):
            self.n_query = x.size(1) - self.n_support
            acc_all.append(self.accuracy_fun(x, n_way))
        acc_all  = np.asarray(acc_all)
        teAcc = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        conf_interval = 1.96* acc_std/np.sqrt(iter_num)
        return teAcc, conf_interval

