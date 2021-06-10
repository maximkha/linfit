from typing import List
import torch
from torch import Tensor
from torch._C import Module, Value
import torch.nn as nn
from torch.nn.modules.activation import ReLU6
from torch.nn.modules.linear import Linear
from enum import Enum

def solvelreg(x: Tensor, y: Tensor) -> Tensor:
    return (torch.linalg.pinv(x.T) @ y.T).T

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

VALID_OPS = [nn.Linear, nn.ReLU] # for now, we'll only support these
TRANSPARENT_OPS = [nn.ReLU]
SINGLEBACK_OPS = []

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
    if linlay.bias is None: return linlay.weight
    raise NotImplementedError("I'll do this later")

class WeightState(Enum):
    NO = 0
    ACCUM = 1

def backsingle(module: nn.Module, Ys: torch.Tensor) -> torch.Tensor:
    #TODO: implement cnn reverse here
    pass

#TODO: refractor so it supports the bias
def backwards(modules: List[nn.Module], layern, Ys: torch.Tensor):
    # techincally we should call check valid ops here!!
    # modules = flattenseq(model)

    result = Ys
    weightstate = WeightState.NO
    cuweight = None
    for module in reversed(modules[layern:]):
        if type(module) in TRANSPARENT_OPS: continue
        if type(module) == nn.Linear:
            weight = biasweight(module)
            if weightstate == WeightState.NO:
                weightstate = WeightState.ACCUM
                cuweight = weight
                continue
            cuweight = cuweight @ weight
        elif type(module) in SINGLEBACK_OPS:
            if weightstate == WeightState.ACCUM:
                result = torch.linalg.pinv(cuweight) @ result
                #TODO: pop bias off of the reversed result
            weightstate = WeightState.NO
            result = backsingle(module, result)
        else:
            raise ValueError(f"Unsupported reverse for op {type(module).__name__}")

    if weightstate == WeightState.ACCUM: result = torch.linalg.pinv(cuweight) @ result #REFRACTOR
    return result

def forwardsto(modules: List[nn.Module], n: int, Xs: torch.Tensor) -> torch.Tensor:
    for module in modules[:n]:
        Xs = module.forward(Xs)
    return Xs

def appendone(t: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError()

if __name__ == '__main__':
    print("yo")
    # mod = nn.Sequential(
    #     nn.Sequential(
    #         nn.Linear(5, 10, False),
    #         nn.ReLU(),
    #     ),
    #     nn.Linear(10, 5, False),
    #     nn.ReLU(),
    #     nn.Linear(5, 2, False),
    #     nn.ReLU(),
    # )

    mod = nn.Sequential(
        nn.Sequential(
            nn.Linear(3, 1, False),
            nn.ReLU(),
        ),
    )

    modules = flattenseq(mod)
    print(modules)
    checkvalidops(modules)
    with torch.no_grad():
        outp = backwards(modules, 0, torch.Tensor([24]))
        print(outp)
        print(outp.shape)

        print(mod(outp))

        # outp = backwards(modules, 3, torch.Tensor([1, 2]))
        # print(outp)
        # print(outp.shape)