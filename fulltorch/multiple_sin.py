from matplotlib import pyplot as plt
import numpy as np
import torch
import linstorch
import torch.nn as nn

#create sample dataset
Xs = np.linspace(0, 10, 101) * 2 * np.pi * (1/10)
Xs = torch.tensor(Xs.reshape(-1,1)).float()

Ys_wish = np.sin(Xs) + 1
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

    return mod

mod = genmodel()

for module in mod.children():
    if type(module) == nn.Linear:
        torch.nn.init.uniform_(module.bias, 0, 1)
        torch.nn.init.uniform_(module.weight, 0, 1)

mod = linstorch.solvemodel(mod, Xs, Ys_wish)

# for j in range(1):
#     for i in range(len(model.layers)):
#         model.solve_update(Xs, Ys_wish, n=len(model.layers)-i) #n=random.randint(1, len(model.layers)))
#         forw = model.forward(Xs)
#         #print(forw.shape)
#         # plt.plot(Xs[0,:], forw[0,:], label=f"Back->Front ({len(model.layers)-i})")
#         mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
#         print(mse)

#     for i in range(len(model.layers)):
#         model.solve_update(Xs, Ys_wish, n=i + 1) #n=random.randint(1, len(model.layers)))
#         forw = model.forward(Xs)
#         # plt.plot(Xs[0,:], forw[0,:], label=f"Front->Back ({i + 1})")
#         #print(forw.shape)
#         mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
#         print(mse)

print(f"{torch.mean(((mod(Xs) - Ys_wish)**2))=}")

# forw = mod(Xs)
# mse_mod = np.mean((forw - Ys_wish)**2)
# print(mse_mod)

# print("======")

#print((np.polyfit(Xs[0,:], Ys_wish[0,:], 1) @ Xs) - Ys_wish[0,:])
# mse_lin = np.mean(((np.polyfit(Xs, Ys_wish, 1) @ Xs) - Ys_wish)**2)
# print(mse_lin)

# print("+++++++++++++")
# print(f"mod/lin err: {mse_mod/mse_lin}")

plt.plot(Xs, mod(Xs).detach().numpy(), label='relu with training (custom)')
plt.plot(Xs, Ys_wish.detach().numpy(), label='real')

plt.legend()
plt.show()