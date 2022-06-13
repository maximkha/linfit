from typing import List
import numpy as np
from funcy import print_durations

# approximates the solution for A x = y, e.g. linear regression
def appx_solve(x, y):
    print(x.shape)
    print(y.shape)
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

        print(f"===============")
        print(f"SOLVING {n=}")
        print(f"{backw_val=}")
        print(f"{forw_val=}")
        print(f"{appx_solve(forw_val, backw_val)=}")

        return appx_solve(forw_val, backw_val)

    def solve_update(self, Xs, Ys, n):
        if n == 0: raise ValueError("Can't solve for 0th layer!")
        solution = self.solve_lay(Xs, Ys, n)
        self.layers[n - 1].update(solution)