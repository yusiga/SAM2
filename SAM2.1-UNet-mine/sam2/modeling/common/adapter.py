import torch
import torch.nn as nn


class Adapter(nn.Module):
    _printed = set()

    def __init__(self, D_features, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        # self.D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, 32)
        self.D_fc2 = nn.Linear(32, D_features)
        
    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        xs = self.act(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x