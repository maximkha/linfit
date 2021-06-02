import sys

import numpy as np
sys.path.append(r"C:\Users\maxim\Desktop\js\linfit")
from linfit import lay, lay_holder

model = lay_holder([lay(2, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 1, True)])

Xs = np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
Xs = np.insert(Xs, Xs.shape[-1], np.ones(Xs.shape[0]), axis=1).T

Ys_wish = np.array([[0.],[1.],[1.],[0.]])
Ys_wish = np.insert(Ys_wish, Ys_wish.shape[-1], np.ones(Ys_wish.shape[0]), axis=1).T
print(Xs.shape)
print(Ys_wish.shape)

for j in range(1):
    for i in range(len(model.layers)):
        model.solve_update(Xs, Ys_wish, n=len(model.layers)-i)
        forw = model.forward(Xs)
        mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
        print(mse)

    for i in range(len(model.layers)):
        model.solve_update(Xs, Ys_wish, n=i + 1)
        forw = model.forward(Xs)
        #print(forw.shape)
        mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
        print(mse)

print(model.forward(Xs))

#WORKS!