from typing import List
import torch

USE_CUDA = True

# approximates the solution for A x = y, e.g. linear regression
def appx_solve(x, y) -> torch.Tensor:
    return (torch.linalg.pinv(x.T) @ y.T).T

class lay:
    def __init__(self, x_dim, y_dim, relu=True) -> None:
        #self.f = np.random.randn(y_dim + 1, x_dim + 1)
        self.f = torch.FloatTensor(y_dim + 1, x_dim + 1).uniform_() #torch.rand(y_dim + 1, x_dim + 1)# * (np.sqrt(y_dim))
        if USE_CUDA: self.f = self.f.cuda()
        # self.f = np.abs(self.f)

        self.f[-1] = torch.cat((torch.zeros(self.f.shape[-1] - 1), torch.ones(1)), 0)
        self.do_relu = relu
    
    def forward(self, x) -> torch.Tensor: #include bias column on first layer
        outp = self.f @ x
        if self.do_relu: outp = torch.relu(outp)
        return outp

    def update(self, new_f, correct = True) -> None:
        self.f = new_f
        if correct: self.f[-1] = torch.cat((torch.zeros(self.f.shape[-1] - 1), torch.ones(1)), 0)

class lay_holder:
    def __init__(self, lays:List[lay]) -> None:
        if lays == None: raise ValueError("Invalid set of layers")
        self.layers = lays

    def forward(self, x:torch.Tensor, n = None) -> torch.Tensor:
        if n == None: n = len(self.layers)
        elif n == -1: return x
        x = x.cuda()
        result = x
        for layer in self.layers[:n]:
            result = layer.forward(result)
        return result

    def backward_mult(self, n = -1) -> torch.Tensor:
        if n == -1: n = len(self.layers)
        # it is more useful if n == 0 we return the identity matrix!
        elif n == 0:
            #create an identity matrix with the shape of the output.
            if USE_CUDA: return torch.eye(self.layers[-1].f.shape[0]).cuda()
            else: return torch.eye(self.layers[-1].f.shape[0])

        if len(self.layers) - n < 0:
            raise ValueError("Can't backward that many layers!")
        
        prod = self.layers[-1].f
        if n == 1: return prod
        for i in range(n - 1):
            layer = self.layers[len(self.layers) - i - 2]
            prod = prod @ layer.f
        
        return prod
    
    def solve_lay(self, Xs, Ys, n=-1) -> torch.Tensor:
        if n == -1: n = len(self.layers)
        backw = self.backward_mult(len(self.layers) - n)
        backw_val = torch.linalg.pinv(backw) @ Ys
        forw_val = self.forward(Xs, n - 1)
        return appx_solve(forw_val, backw_val)

    def solve_update(self, Xs, Ys, n) -> None:
        if n == 0: raise ValueError("Can't solve for 0th layer!")
        solution = self.solve_lay(Xs, Ys, n)
        self.layers[n - 1].update(solution)