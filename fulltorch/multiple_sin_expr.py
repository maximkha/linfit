from matplotlib import pyplot as plt
import numpy as np
import torch
import linstorch
import torch.nn as nn

#create sample dataset
Xs = np.linspace(0, 10, 101) * 2 * np.pi * (1/10)
Xs = torch.tensor(Xs.reshape(-1,1)).float()

Ys_wish = np.sin(Xs) + 1
# Ys_wish = np.cos(Xs) + 1
Ys_wish = torch.tensor(Ys_wish.reshape(-1,1)).float()

def genmodel() -> torch.nn.Sequential:
    mod = nn.Sequential(
        nn.Linear(1, 11, True),
        nn.ReLU(),
        nn.Linear(11, 11, True),
        nn.ReLU(),
        nn.Linear(11, 1, True),
        nn.ReLU()
    )

    for module in mod.children():
        if type(module) == nn.Linear:
            torch.nn.init.uniform_(module.bias, 0, 1)
            torch.nn.init.uniform_(module.weight, 0, 1)

    return mod

linear_model = nn.Sequential(nn.Linear(1, 1, True))
linear_model = linstorch.solvemodel(linear_model, Xs, Ys_wish)
lin_mse = torch.mean(((linear_model(Xs) - Ys_wish)**2)).detach().item()

mse_s = []
for i in range(1000):
    mod = genmodel()
    mod = linstorch.solvemodel(mod, Xs, Ys_wish)
    mse = torch.mean(((mod(Xs) - Ys_wish)**2))

    mse_s.append(mse.detach().item())

print(np.mean(mse_s))