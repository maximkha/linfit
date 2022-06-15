import linstorch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import torch

Xs = np.linspace(0, 10, 6) * 2 * np.pi * (1/10)
Xs = torch.tensor(Xs.reshape(-1,1)).float()

Ys_wish = np.sin(Xs) + 1
Ys_wish = torch.tensor(Ys_wish.reshape(-1,1)).float()

def genmodel() -> torch.nn.Sequential:
    mod = nn.Sequential(
        nn.Linear(1, 2, True),
        nn.ReLU(),
        nn.Linear(2, 2, True),
        nn.ReLU(),
        nn.Linear(2, 1, True),
        nn.ReLU(),
    )

    return mod

mod = genmodel()

# for module in mod.children():
#     if type(module) == nn.Linear:
#         module.bias = nn.Parameter([1.,2.])
#         module.weight = nn.Parameter([[1.,2.]])

modules = list(mod.children())
print(f"{modules=}")

modules[0].bias = nn.Parameter(torch.tensor([[1,2]]).float())
modules[2].bias = nn.Parameter(torch.tensor([[1,2]]).float())
modules[4].bias = nn.Parameter(torch.tensor([[1]]).float())

modules[0].weight = nn.Parameter(torch.tensor([[1,2]]).T.float())
modules[2].weight = nn.Parameter(torch.tensor([[1,2],[1,2]]).float())
modules[4].weight = nn.Parameter(torch.tensor([[1],[2]]).T.float())

lay = linstorch.biasweight(modules[0])
print(f"{lay=}")

print(f"{lay@linstorch.APPENDONE(Xs).T=}")
print(f"{(lay@linstorch.APPENDONE(Xs).T).T[:,:-1]=}")

print(f"{modules[0](Xs)=}")
mods = linstorch.flattenseq(mod)
solvables = linstorch.getsolveable(mods)
s_mods = [mods[i] for i in solvables]
backw = linstorch.backwards(s_mods, 1, Ys_wish)
print(f"{backw=}")

linstorch.solvemodule(modules[0], Xs, backw)

exit()

mods = linstorch.flattenseq(mod)
solvables = linstorch.getsolveable(mods)
s_mods = [mods[i] for i in solvables]
backw = linstorch.backwards(s_mods, 1, Ys_wish)
print(f"{backw=}")
# linstorch.solvemodule(s_mods[0], Xs, backw)

# mod = linstorch.solvemodel(mod, Xs, Ys_wish)

print(f"{mod(Xs)=}")

# mse_mod = torch.mean((mod(Xs) - Ys_wish)**2)

modu = linstorch.flattenseq(mod)
# print(f"{linstorch.forwardsto(modu, len(modu)-1, Xs) - mod(Xs)=}")

# print(f"{mse_mod=}")

# import matplotlib.pyplot as plt
# plt.plot(Xs.detach().numpy(), mod(Xs).detach().numpy(), label='relu with training (custom)')
# plt.show()