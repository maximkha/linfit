from typing import List
import numpy as np

# approximates the solution for A x = y, e.g. linear regression
def appx_solve(x, y):
    return (np.linalg.pinv(x.T) @ y.T).T

class lay:
    def __init__(self, x_dim, y_dim, relu=True) -> None:
        #self.f = np.random.randn(y_dim + 1, x_dim + 1)
        self.f = np.random.rand(y_dim + 1, x_dim + 1)# * (np.sqrt(y_dim))
        # self.f = np.abs(self.f)

        self.f[-1] = ([0]*x_dim) + [1]
        self.do_relu = relu
    
    def forward(self, x): #include bias column on first layer
        outp = self.f @ x
        if self.do_relu: outp = np.maximum(outp, 0)
        return outp

    def update(self, new_f, correct = True):
        self.f = new_f
        if correct: self.f[-1] = ([0]*(self.f.shape[-1] - 1)) + [1]

class lay_holder:
    def __init__(self, lays:List[lay]) -> None:
        if lays == None: raise ValueError("Invalid set of layers")
        self.layers = lays

    def forward(self, x, n = None):
        if n == None: n = len(self.layers)
        elif n == -1: return x
        result = x
        for layer in self.layers[:n]:
            result = layer.forward(result)
        return result

    def backward_mult(self, n = -1):
        if n == -1: n = len(self.layers)
        # it is more useful if n == 0 we return the identity matrix!
        elif n == 0: return np.eye(self.layers[-1].f.shape[0]) #create an identity matrix with the shape of the output.

        if len(self.layers) - n < 0:
            raise ValueError("Can't backward that many layers!")
        
        prod = self.layers[-1].f
        if n == 1: return prod
        for i in range(n - 1):
            layer = self.layers[len(self.layers) - i - 2]
            prod = prod @ layer.f
        
        return prod
    
    def solve_lay(self, Xs, Ys, n=-1):
        if n == -1: n = len(self.layers)
        backw = self.backward_mult(len(self.layers) - n)
        backw_val = np.linalg.pinv(backw) @ Ys
        forw_val = self.forward(Xs, n - 1)
        return appx_solve(forw_val, backw_val)

    def solve_update(self, Xs, Ys, n):
        if n == 0: raise ValueError("Can't solve for 0th layer!")
        solution = self.solve_lay(Xs, Ys, n)
        self.layers[n - 1].update(solution)

#create sample dataset
Xs = np.linspace(0, 10, 101) * 2 * np.pi * (1/10)
Xs = np.concatenate((Xs.reshape(1,-1), np.ones(Xs.shape[0]).reshape(1, -1)), axis=0)

Ys_wish = (np.sin(Xs[0,:])) + 1# (Xs[0,:] ** 8) 
Ys_wish = np.concatenate((Ys_wish.reshape(1,-1), np.ones(Ys_wish.shape[0]).reshape(1, -1)), axis=0)
#print(Xs)

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

#model = lay_holder([lay(1, 20, True), lay(20, 10, True), lay(10, 10, True), lay(10, 10, True), lay(10, 10, True), lay(10, 10, True), lay(10, 20, True), lay(20, 1, True)])
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

# for i in range(len(model.layers)):
#     model.solve_update(Xs, Ys_wish, n=len(model.layers)-i) #n=random.randint(1, len(model.layers)))
#     forw = model.forward(Xs)
#     #print(forw.shape)
#     mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
#     print(mse)

for j in range(1):
    for i in range(len(model.layers)):
        model.solve_update(Xs, Ys_wish, n=len(model.layers)-i) #n=random.randint(1, len(model.layers)))
        forw = model.forward(Xs)
        #print(forw.shape)
        mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
        print(mse)

    for i in range(len(model.layers)):
        model.solve_update(Xs, Ys_wish, n=i + 1) #n=random.randint(1, len(model.layers)))
        forw = model.forward(Xs)
        #print(forw.shape)
        mse = np.mean((forw[0,:] - Ys_wish[0,:])**2)
        print(mse)


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

import matplotlib.pyplot as plt
plt.plot(Xs[0,:], np.polyfit(Xs[0,:], Ys_wish[0,:], 1) @ Xs, label='simple line')

plt.plot(Xs[0,:], model.forward(Xs)[0,:], label='relu with training (custom)')

plt.plot(Xs[0,:], Ys_wish[0,:], label='real')

plt.legend()
plt.show()