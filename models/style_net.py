# -*- coding : utf-8 -*-
# @FileName  : style_net.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Dec 03, 2023
# @Github    : https://github.com/songrise
# @Description: stylization net

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class StyleNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed, self.out_dim = get_embedder(4, i=0)
        self.rgb_embd, self.rgb_embd_dim = get_embedder(2, i=0)
        hidden_dim = 96
        fcs = []
        fcs.append(nn.Linear(self.out_dim, hidden_dim))
        self.i_skip = 3
        for _ in range(6):
            if _ == self.i_skip:
                fcs.append(nn.Linear(hidden_dim+self.rgb_embd_dim,hidden_dim))
            else:
                fcs.append(nn.Linear(hidden_dim,hidden_dim))
        self.delta_sh_head = nn.Linear(hidden_dim, 3)
        self.fcs = nn.ModuleList(fcs)
        self.hidden_act = nn.ReLU()
        self.out_act = nn.Tanh()
        self.delta_sh_head.weight.data.uniform_(-1e-2, 1e-2)
        self.delta_sh_head.bias.data.uniform_(-1e-3, 1e-3)

        self.rgb_proj = nn.Linear(3, 3)
        self.rgb_proj.weight.data.uniform_(-1e-2, 1e-2)
        self.rgb_proj.bias.data.uniform_(-1e-3, 1e-3)

    def forward(self, xyz, rgb):
        xyz = self.embed(xyz)
        for i, fc in enumerate(self.fcs):
            if i == self.i_skip+1:
                rgb_embd = self.rgb_embd(rgb)
                xyz = self.hidden_act(fc(torch.cat([xyz, rgb_embd], dim=-1)))
            else:
                xyz = self.hidden_act(fc(xyz))
        delta_rgb =  self.out_act(self.delta_sh_head(xyz))

        rgb_proj = self.out_act(self.rgb_proj(rgb))
        rgb_out = torch.clamp(rgb + delta_rgb + rgb_proj, 0, 1)
        return rgb_out
if __name__ == '__main__':

    xyz = torch.randn(1024,3)
    rgb = torch.randn(1024,3)
    style_net = StyleNet()
    out = style_net(xyz, rgb)
    print(out.shape)
    print(rgb.min())
    print(out.min())