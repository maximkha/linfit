from linfit import lay_holder, lay
import numpy as np

#create sample dataset
Xs = np.linspace(0, 10, 101) * 2 * np.pi * (1/10)
Xs = np.concatenate((Xs.reshape(1,-1), np.ones(Xs.shape[0]).reshape(1, -1)), axis=0)

Ys_wish = (np.sin(Xs[0,:])) + 1# (Xs[0,:] ** 8) 
Ys_wish = np.concatenate((Ys_wish.reshape(1,-1), np.ones(Ys_wish.shape[0]).reshape(1, -1)), axis=0)
print(Xs)

# #create simple lin net
# model = lay_holder([lay(1, 2, False), lay(2, 2, False), lay(2, 1, False)])
# print(Xs.T)

# Ys_wish = ((Xs[0,:] * 2) + 1)
# Ys_wish = np.concatenate((Ys_wish.reshape(1,-1), np.ones(Ys_wish.shape[0]).reshape(1, -1)), axis=0)

# model.solve_update(Xs, Ys_wish, 3)
# forw = model.forward(Xs)
# print(forw.T)

# print(np.mean((forw[0,:] - Ys_wish[0,:])**2))

import random

# model = lay_holder([lay(1, 20, True), lay(20, 10, True), lay(10, 10, True), lay(10, 10, True), lay(10, 10, True), lay(10, 10, True), lay(10, 20, True), lay(20, 1, True)])
# model = lay_holder([lay(1, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 1, True)])

#doublepass - x**8
#model = lay_holder([lay(1, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 1, True)])

#doublepass - sinx+1
model = lay_holder([lay(1, 11, True), lay(11, 11, True), lay(11, 1, True)])
#model = lay_holder([lay(1, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 1, True)])
#model = lay_holder([lay(1, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 1, True)])
#model = lay_holder([lay(1, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 10, True), lay(10, 20, True), lay(20, 1, True)])
#model = lay_holder([lay(1, 11, True), lay(11, 11, True), lay(11, 11, True), lay(11, 1, True)])

# for i in range(100):
#     model.solve_update(Xs, Ys_wish, n=random.randint(1, len(model.layers)))
#     forw = model.forward(Xs)
#     #print(forw.shape)
#     mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
#     print(mse)

# for j in range(100):
#     for i in range(len(model.layers)):
#         model.solve_update(Xs, Ys_wish, n=len(model.layers)-i)#n=random.randint(1, len(model.layers)))
#         forw = model.forward(Xs)
#         #print(forw.shape)
#         mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
#         print(mse)

# solutions = []
# for i in range(len(model.layers)):
#     layn = len(model.layers) - i
#     sol = model.solve_lay(Xs, Ys_wish, n=layn)
#     solutions.append(sol)
#     forw = model.forward(Xs)
#     mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
#     print(mse)

# #implement solutions
# for i in range(len(model.layers)):
#     layn = len(model.layers) - i
#     # update
#     model.layers[layn - 1].update(solutions[i])

# print(solutions)

# for i in range(len(model.layers)):
#     model.solve_update(Xs, Ys_wish, n=len(model.layers)-i) #n=random.randint(1, len(model.layers)))
#     forw = model.forward(Xs)
#     #print(forw.shape)
#     mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
#     print(mse)

import matplotlib.pyplot as plt

print(Xs.shape)
print(Ys_wish.shape)

for j in range(1):
    for i in range(len(model.layers)):
        model.solve_update(Xs, Ys_wish, n=len(model.layers)-i) #n=random.randint(1, len(model.layers)))
        forw = model.forward(Xs)
        #print(forw.shape)
        # plt.plot(Xs[0,:], forw[0,:], label=f"Back->Front ({len(model.layers)-i})")
        mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
        print(mse)
    
    # plt.legend()
    # plt.show()

    for i in range(len(model.layers)):
        model.solve_update(Xs, Ys_wish, n=i + 1) #n=random.randint(1, len(model.layers)))
        forw = model.forward(Xs)
        # plt.plot(Xs[0,:], forw[0,:], label=f"Front->Back ({i + 1})")
        #print(forw.shape)
        mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
        print(mse)
    
    # plt.legend()
    # plt.show()

model.solve_update(Xs, Ys_wish, len(model.layers))
forw = model.forward(Xs)
mse_mod = np.mean((forw[0,:] - Ys_wish[0,:])**2)
print(mse_mod)

print("======")
nlin = lay_holder([lay(1, 1, False)])
nlin.solve_update(Xs, Ys_wish, 1)
print(np.mean((nlin.forward(Xs)[0,:] - Ys_wish[0,:])**2))

#print((np.polyfit(Xs[0,:], Ys_wish[0,:], 1) @ Xs) - Ys_wish[0,:])
mse_lin = np.mean(((np.polyfit(Xs[0,:], Ys_wish[0,:], 1) @ Xs) - Ys_wish[0,:])**2)
print(mse_lin)

print("+++++++++++++")
print(f"mod/lin err: {mse_mod/mse_lin}")

plt.plot(Xs[0,:], np.polyfit(Xs[0,:], Ys_wish[0,:], 1) @ Xs, label='simple line')

plt.plot(Xs[0,:], model.forward(Xs)[0,:], label='relu with training (custom)')

plt.plot(Xs[0,:], Ys_wish[0,:], label='real')

plt.legend()
plt.show()