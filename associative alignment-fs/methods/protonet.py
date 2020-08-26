# This code is modified from https://github.com/jakesnell/prototypical-networks 
import torch
from torch.autograd import Variable
import numpy as np

def pn_loss(self, z_support, z_query, loss_fn = None, loss = False, score = False, rb=None): 
    
    if rb:    
        n_query = z_query.shape[0]
        n_way = z_support.shape[0]
        n_support = z_support.shape[1]
    else:
        n_way = self.n_way
        n_query = self.n_query
        n_support = self.n_support
    
    
    y_query = torch.from_numpy(np.repeat(range(n_way), n_query))
    y_query = Variable(y_query.cuda())
    
    z_support   = z_support.contiguous()
    z_proto     = z_support.view(n_way, n_support, -1 ).mean(1)  
    
    if rb:
        z_query     = z_query.contiguous().view(n_query, -1 )
    else:
        z_query     = z_query.contiguous().view(n_way* n_query, -1 )
    
    dists = euclidean_dist(z_query, z_proto)               ## torch.Size([80, 5])
    scores = -dists
    
    if score: 
        return scores
    elif loss:
        return loss_fn(scores, y_query) 


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)












#def pn_loss_acc(zs, zq, y, n_way, tr_Nshot, te_Nshot): 
#    
#    zs = zs.view(n_way, tr_Nshot, -1)                            ## z.size(-1) = 64
#    zq = zq.view(zq.size(0),-1)
#    zs_proto = zs.mean(1)                                                  ## 60, 64  
#    ### Ar: Computing distances between quary and prototypes
#    dists_zqzs_proto = euclidean_dist(zq, zs_proto)                        ## 60, 60
#    log_exp_ndist  = nn.functional.log_softmax(-dists_zqzs_proto, dim=1)          ## 5x5
#    
#    #backing to original query structure 
#    log_exp_ndist = log_exp_ndist.view(n_way, te_Nshot, -1)                 ## 5x1x5
#    _, y_hat = log_exp_ndist.max(2)                                        ## 5x1
#    
#    gather_log_exp_dist = -log_exp_ndist.gather(2, y)                      ## 60x5x1 selecting the corresponding distances
#    loss_ = gather_log_exp_dist.squeeze().view(-1).mean()                  ## mean of all   
#    acc_ = torch.eq(y_hat.squeeze(), y.squeeze()).float().mean()           ## mean within all class in n_way
#    
#    return loss_, acc_