import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class SelfAttn(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttn, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax2d(dim=-1)
    def forward(self, x):
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, C)#BxNxC
        proj_key = self.key_conv(x).view(m_batchsize, C, -1)#BxCxN
        attn_mm = torch.bmm(proj_query, proj_key) #BxNxN batch_matrix_multiplication

        proj_value = self.value_conv(x).view(m_batchsize, C, -1) #BxCxN, this is V
        attn_map = self.softmax(attn_mm/torch.sqrt(width*height)) #softmax(QK/sqrt(N))
        attn_mm2 = torch.bmm(proj_value, attn_map.view(0,2,1))
        out = attn_mm2.view(m_batchsize, C, width, height)

        out = self.final_conv(out)

        out = self.gamma*out+x
        return out
    


