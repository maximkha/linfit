from linfit import lay_holder, lay
import numpy as np

Xs = np.linspace(0, 10, 6) * 2 * np.pi * (1/10)
Xs = np.concatenate((Xs.reshape(1,-1), np.ones(Xs.shape[0]).reshape(1, -1)), axis=0)

Ys_wish = (np.sin(Xs[0,:])) + 1
Ys_wish = np.concatenate((Ys_wish.reshape(1,-1), np.ones(Ys_wish.shape[0]).reshape(1, -1)), axis=0)

model = lay_holder([lay(1, 2, True), lay(2, 2, True), lay(2, 1, True)])

print(f"{model.layers[0].f=}")
print(f"{model.layers[1].f=}")
print(f"{model.layers[2].f=}")

model.layers[0].f[:-1,-1] = np.array([1,2])
model.layers[0].f[:-1,:-1] = np.array([[1,2]]).T

model.layers[1].f[:-1,-1] = np.array([1,2])
model.layers[1].f[:-1,:-1] = np.array([[1,2],[1,2]]).T

model.layers[2].f[0,:-1] = np.array([1,2])
model.layers[2].f[0,-1] = np.array([1])

for j in range(1):
    for i in range(len(model.layers)):
        model.solve_update(Xs, Ys_wish, n=len(model.layers)-i)
        forw = model.forward(Xs)
        mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
        print(mse)

    for i in range(len(model.layers)):
        model.solve_update(Xs, Ys_wish, n=i + 1)
        forw = model.forward(Xs)
        mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
        print(mse)

model.solve_update(Xs, Ys_wish, len(model.layers))
forw = model.forward(Xs)
mse_mod = np.mean((forw[0,:] - Ys_wish[0,:])**2)

print(f"{mse_mod=}")

import matplotlib.pyplot as plt
plt.plot(Xs[0,:], model.forward(Xs)[0,:], label='relu with training (custom)')
plt.show()