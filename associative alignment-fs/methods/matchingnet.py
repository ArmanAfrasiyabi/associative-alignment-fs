# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)
    
def mn_loss(self, z_support, z_query, n_way, loss_fn=None, score=False, loss=False, FCE = None):
    z_support   = z_support.contiguous().view( n_way* self.n_support, -1 )
    z_query     = z_query.contiguous().view( n_way* self.n_query, -1 )
    G, G_normalized = encode_training_set(self, z_support)
    y_s         = torch.from_numpy(np.repeat(range( n_way ), self.n_support ))
    Y_S         = Variable(one_hot(y_s, n_way ) ).cuda()
    f           = z_query
    if FCE is None:
        FCE = self.FCE.cuda()
    F = FCE(f, G)
    F_norm = torch.norm(F,p=2, dim =1).unsqueeze(1).expand_as(F)
    F_normalized = F.div(F_norm+ 0.00001) 
    #scores = F.mm(G_normalized.transpose(0,1)) #The implementation of Ross et al., but not consistent with origin paper and would cause large norm feature dominate 
    scores = self.relu( F_normalized.mm(G_normalized.transpose(0,1))  ) *100 # The original paper use cosine simlarity, but here we scale it by 100 to strengthen highest probability after softmax
    softmax = self.softmax(scores)
    logprobs =(softmax.mm(Y_S)+1e-6).log()
    if score: 
        return logprobs 
    elif loss:
        y_query = torch.from_numpy(np.repeat(range( n_way ), self.n_query ))
        y_query = Variable(y_query.cuda()) 
        return loss_fn(logprobs, y_query)

def encode_training_set(self, S, G_encoder = None):
    if G_encoder is None:
        G_encoder = self.G_encoder
    out_G = G_encoder(S.unsqueeze(0))[0]
    out_G = out_G.squeeze(0)
    G = S + out_G[:,:S.size(1)] + out_G[:,S.size(1):]
    G_norm = torch.norm(G,p=2, dim =1).unsqueeze(1).expand_as(G)
    G_normalized = G.div(G_norm+ 0.00001) 
    return G, G_normalized

class FullyContextualEmbedding(nn.Module):
    def __init__(self, out_dim):
        super(FullyContextualEmbedding, self).__init__()
        self.lstmcell = nn.LSTMCell(out_dim*2, out_dim)
        self.softmax = nn.Softmax(dim=1)
        self.c_0 = Variable(torch.zeros(1, out_dim))
#        self.out_dim = out_dim
    def forward(self, f, G):
        h = f
        c = self.c_0.expand_as(f)
        G_T = G.transpose(0,1)
        K = G.size(0) #Tuna to be comfirmed
        for k in range(K):
            logit_a = h.mm(G_T)
            a = self.softmax(logit_a)  #dim=1
            r = a.mm(G)
            x = torch.cat((f, r),1)

            h, c = self.lstmcell(x, (h, c))
            h = h + f

        return h
    def cuda(self):
        super(FullyContextualEmbedding, self).cuda()
        self.c_0 = self.c_0.cuda()
        return self

