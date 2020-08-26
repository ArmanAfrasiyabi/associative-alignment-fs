import torch
from torch.autograd import Variable 
def euclidean_dist(x, y):
    n = x.size(0)  
    m = y.size(0)  
    d = x.size(1)  
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)       
    y = y.unsqueeze(0).expand(n, m, d)      

    return torch.pow(x - y, 2).sum(2)

def centroid_aligner(args,net, data, support_size, query_size, device):
    [xt, xa] = data
    
    za = net(xa)
    anchor = net(xt)
    
    yt = torch.arange(0, args.test_n_way).view(args.test_n_way, 1, 1).expand(args.test_n_way, query_size, 1).long()
    yt = Variable(yt, requires_grad=False).to(device)
    
    anchor = anchor.view(args.test_n_way, support_size, -1)                                           
    za = za.view(za.size(0), -1)                                                
    anchor_mu = anchor.mean(1)  
                                                          
    dists = -euclidean_dist(za, anchor_mu)       
    log_exp_ndist  = torch.nn.functional.log_softmax(dists, dim=1)          
    log_exp_ndist = log_exp_ndist.view(args.test_n_way, query_size, -1)                 
                                          
    gather_log_exp_dist = -log_exp_ndist.gather(2, yt)                       
    loss = gather_log_exp_dist.squeeze().view(-1).mean()  
    
    return loss

    
    
 