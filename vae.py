import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self, input_dim, batch_size):
        super().__init__()
        self.input_dim = input_dim
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size * 1000

    def __getitem__(self, i):
        one_hot = torch.zeros(self.input_dim)
        label = torch.randint(0, self.input_dim, (1, )).long()
        one_hot[label] = 1.0
        return one_hot, label

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.l = nn.Linear(self.hidden_dim, self.z_dim)
        self.encode = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU())
        self.decode = nn.Sequential(
                nn.Linear(self.z_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.input_dim))

    def encoder(self, x):
        # q(z|x)
        x = self.encode(x)
        mu = self.l(x)
        scale = self.l(x)
        return Normal(mu, scale)

    def decoder(self, z):
        # p(x|z)
        logits = self.decode(z)
        return Bernoulli(logits = logits)

    def forward(self, x):
        q = self.encoder(x.view(x.shape[0], -1))
        p = self.decoder(q.rsample())
        return p, q
