from typing import List
import torch
from torch import Tensor
import torch.nn as nn
from enum import Enum

# def solvelreg(x: Tensor, y: Tensor) -> Tensor:
#     return (torch.linalg.pinv(x) @ y).T

# natively solve a torch sequential model.
# NOTE: Conv implementation will be slow due to python and a my lack of a clever way to go 'backwards' throught them
# TODO: Linear, Conv

def flattenseq(model:nn.Sequential) -> List[nn.Module]:
    flat_layers:List[nn.Module] = []
    for module in model.children():
        if not isinstance(module, nn.Sequential):
            flat_layers.append(module)
        else:
            flat_layers.extend(flattenseq(module))
    return flat_layers

VALID_OPS = [nn.Linear, nn.ReLU, nn.ELU, nn.Softmax, nn.Identity, nn.LeakyReLU] # for now, we'll only support these
TRANSPARENT_OPS = [nn.ReLU, nn.Softmax, nn.Identity]
SINGLEBACK_OPS = [nn.ELU, nn.LeakyReLU]
SOLVEABLE = [nn.Linear]

# maybe make a wrapper class for seq???
def checkvalidops(modules: List[nn.Module]):
    global VALID_OPS
    # modules = flattenseq(model)
    for i, module in enumerate(modules):

        if (i == 0) and (type(module) == nn.Flatten):
            continue #if the model starts with a flatten that's ok.
        if not type(module) in VALID_OPS:
            raise ValueError(f"Unsupported torch op {type(module).__name__}")

def biasweight(linlay:nn.Linear):
    appendzeros = nn.ConstantPad1d((0,1), 0.)
    # print(f"{linlay.weight=}")
    weightwpass = appendzeros(linlay.weight)
    ncol = weightwpass.size(1)
    weightwpass = torch.vstack((weightwpass, torch.zeros(ncol, device=weightwpass.device)))
    weightwpass[-1, -1] = 1

    # print(weightwpass)
    if linlay.bias is None:
        return weightwpass
    
    weightwpass[:-1, -1] = linlay.bias
    return weightwpass

class WeightState(Enum):
    NO = 0
    ACCUM = 1

def backsingle(module: nn.Module, Ys: torch.Tensor) -> torch.Tensor:
    if type(module) == nn.ELU:
        # tmod:nn.ELU = module
        assert module.alpha == 1.
        x = torch.where(Ys < 0, torch.log(Ys+1), Ys)
        x[x==-float('inf')] = -36.7368 #empirical
        return x
    if type(module) == nn.LeakyReLU:
        x = torch.where(Ys < 0, Ys, Ys/module.negative_slope)
        return x

    raise NotImplementedError()

def getsolveable(modules: List[nn.Module]) -> List[int]:
    global SOLVEABLE, VALID_OPS
    outp = []
    for i, module in enumerate(modules):
        modtype = type(module)
        if (modtype in SOLVEABLE) and (modtype in VALID_OPS):
            outp.append(i)
    return outp

APPENDONE = nn.ConstantPad1d((0,1), 1.)

#TODO: refractor so it supports the bias
def backwards(modules: List[nn.Module], layern, Ys: torch.Tensor) -> torch.Tensor:
    # techincally we should call check valid ops here!!
    # modules = flattenseq(model)    
    global APPENDONE

    result = Ys
    weightstate = WeightState.NO
    cuweight = None
    for module in reversed(modules[layern:]):
        if type(module) in TRANSPARENT_OPS: continue
        if type(module) == nn.Linear:
            weight = biasweight(module)
            # print(weight)
            # print(f"{weight=}")
            if weightstate == WeightState.NO:
                result = APPENDONE(result)
                # print(result)
                weightstate = WeightState.ACCUM
                cuweight = weight
                continue
            # print(f"{cuweight=}")
            cuweight = cuweight @ weight
            # print(f"{cuweight=}")

        elif type(module) in SINGLEBACK_OPS:
            if weightstate == WeightState.ACCUM:
                result = result @ torch.linalg.pinv(cuweight, 1e-15).T
                
                result = result[:,:-1] #drop constant column
                #TODO: pop bias off of the reversed result
            weightstate = WeightState.NO
            result = backsingle(module, result)
        else:
            raise ValueError(f"Unsupported reverse for op {type(module).__name__}")

    if weightstate == WeightState.ACCUM: 
        #TODO: pop bias off of the reversed result
        # print(cuweight)
        # print(result.T)
        # print(torch.linalg.pinv(cuweight))
        # print(f"{torch.linalg.pinv(cuweight, 1e-15).T=}")
        # print(f"{cuweight=}")

        result = result @ torch.linalg.pinv(cuweight, 1e-15).T # result @ torch.linalg.pinv(cuweight)
        result = result[:,:-1]
        # print(f"{result=}")
    return result

def forwardsto(modules: List[nn.Module], n: int, Xs: torch.Tensor) -> torch.Tensor:
    for module in modules[:n]:
        Xs = module.forward(Xs)
    return Xs

def solvemodule(module: nn.Module, forwX: torch.Tensor, backY: torch.Tensor) -> nn.Module:
    global APPENDONE
    if type(module) == nn.Linear:
        weight = None
        if module.bias is not None:
            # backY = APPENDONE(backY)
            forwX = APPENDONE(forwX)

            solved = (torch.linalg.pinv(forwX) @ backY).T # solvelreg(forwX, backY)

            # weight = solved[:-1, :-1]
            weight = solved[:, :-1]
            # bias = solved[:-1, -1]
            bias = solved[:, -1]
            # print("new!")

            module.bias = nn.Parameter(bias)
        else:
            solved = (torch.linalg.pinv(forwX) @ backY).T # solvelreg(forwX, backY)
            weight = solved

        module.weight = nn.Parameter(weight)
        return module
    else:
        raise NotImplementedError(f"Not implemented for {type(module).__name__}")

def solve(modules: List[nn.Module], Xs: torch.Tensor, Ys: torch.Tensor) -> List[nn.Module]:
    solveable = getsolveable(modules)
    for _ in range(1):
        for i in reversed(solveable):
            # print(f"{i=}")
            forwx = forwardsto(modules, i, Xs)
            backy = backwards(modules, i + 1, Ys)
            modules[i] = solvemodule(modules[i], forwx, backy)
        
        for i in solveable:
            # print(f"{i=}")
            forwx = forwardsto(modules, i, Xs)
            backy = backwards(modules, i + 1, Ys)
            modules[i] = solvemodule(modules[i], forwx, backy)

    # for i in reversed(solveable):
    #     # print(f"{i=}")
    #     forwx = forwardsto(modules, i, Xs)
    #     backy = backwards(modules, i + 1, Ys)
    #     modules[i] = solvemodule(modules[i], forwx, backy)

    # forwx = forwardsto(modules, solveable[-1], Xs)
    # backy = backwards(modules, solveable[-1] + 1, Ys)
    # modules[solveable[-1]] = solvemodule(modules[solveable[-1]], forwx, backy)

    return modules

def solvemodel(model: nn.Sequential, Xs: torch.Tensor, Ys: torch.Tensor, errorcheck:bool = True) -> nn.Sequential:
    mod = None
    with torch.no_grad():
        modules = flattenseq(model)
        if errorcheck: checkvalidops(modules)
        mod = nn.Sequential(*solve(modules, Xs, Ys))
    return mod

if __name__ == '__main__':
    mod = nn.Sequential(
        nn.Linear(1, 1, True),
        nn.ReLU(),
    )
    xs = torch.Tensor([[1.],[2.]])
    ys = torch.Tensor([[2.],[3.]])

    mod = solvemodel(mod, xs, ys)

    # print(mod(xs))