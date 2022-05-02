import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
import scipy.io as sio
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.autograd import Variable

import fairseq

___author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

#from losses_anti_spoofing import AMSoftmax

############################
## FOR fine-tuning SSL MODEL
############################


class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = '/change_to_path_to_pre_trained_model_XLR_300M/xlsr2_300m.pt'  
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        
        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb

#---------Graph attention simple back-end------------------------#
''' 
    Hemlata Tak, Jee-weon Jung, Jose Patino, Madhu Kamble, Massimiliano Todisco, Nicholas Evans.
    End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection.
    In Proc. Automatic Speaker Verification and Spoofing Countermeasures Challenge 2021 Interspeech 2021 satellite workshop.
'''


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(GraphAttentionLayer, self).__init__()

        #attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        #project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        #batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        #dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)
        
        self.act = nn.SELU(inplace=True)


    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        #apply input dropout
        x = self.input_drop(x)

        #derive attention map
        att_map = self._derive_att_map(x)

        #projection 
        x = self._project(x, att_map)

        #apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)        
        x = x.unsqueeze(2).expand(-1,-1,nb_nodes,-1)
        x_mirror = x.transpose(1,2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))   #size: (#bs, #node, #node, #dim_out)
        att_map = torch.matmul(att_map, self.att_weight)      #size: (#bs, #node, #node, 1)
        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1]) 
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out




class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h

    def top_k_graph(self, scores, h, k):
        """
        args
        =====
        scores: attention-based weights (#bs, #node, 1)
        h: graph data (#bs, #node, #dim)
        k: ratio of remaining nodes, (float)
        returns
        =====
        h: graph pool applied data (#bs, #node', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)

        return h





class Model(nn.Module):
    def __init__(self, d_args, device):
        super(Model, self).__init__()

        #SSL model
        self.device=device
        self.ssl_model = SSLModel(self.device)
        self.LL = nn.Linear(self.ssl_model.out_dim, 128)
        self.first_bn = nn.BatchNorm1d(num_features=128)
        self.selu = nn.SELU(inplace=True)
        
        # graph module layer
        self.GAT_layer=GraphAttentionLayer(128,64)
        self.proj = nn.Linear(64,1)
        self.pool=GraphPool(0.8, 64, 0.3)
       
        #classifier head
        self.proj_node = nn.Linear(53,2)
        

        
    def forward(self, x_inp, Freq_aug=False):
        # SSL wav2vec 2.0 model
        x_ssl_feat = self.ssl_model.extract_feat(x_inp.squeeze(-1))
        x_SSL=self.LL(x_ssl_feat)      #(bs,frame_number,feat_out_dim)
        x_SSL= x_SSL.transpose(1, 2)   #(bs,feat_out_dim,frame_number)
        
        x = F.max_pool1d(x_SSL,(3))
        x = self.first_bn(x)
        x = self.selu(x)
        
        
        x=self.GAT_layer(x.transpose(1,2))
        x=self.pool(x)
        x=self.proj(x).flatten(1) 
        output=self.proj_node(x)
        return output

        

        








       

    
