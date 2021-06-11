import pandas as pd
import numpy as np
import sys
import torch
from torch.nn.modules.activation import ReLU
import torch.nn as nn
import linstorch

df_train = pd.read_csv(r"C:\Users\maxim\Desktop\js\linfit\test\archive\mnist_train.csv")
df_test = pd.read_csv(r"C:\Users\maxim\Desktop\js\linfit\test\archive\mnist_test.csv")

xcolnams = df_train.columns.tolist()
print("check0")
df_train[xcolnams[1:]] /= 255. #convert the pixel values to 0..1
print("check1")

Xs_train = torch.tensor(df_train[xcolnams[1:]].values.T).float().cuda()
print("check2")

ycolnams = []
for i in range(10):
    coln = "is"+str(i)
    ycolnams.append(coln)
    df_train[coln] = (df_train["label"] == i).astype(float)

xcolnams = df_test.columns.tolist()
print("check0")
df_test[xcolnams[1:]] /= 255. #convert the pixel values to 0..1
print("check1")

Xs_test = torch.tensor(df_test[xcolnams[1:]].values.T).float().cuda()
print("check2")

ycolnams = []
for i in range(10):
    coln = "is"+str(i)
    ycolnams.append(coln)
    df_test[coln] = (df_test["label"] == i).astype(float)

Ys_train = torch.tensor(df_train[ycolnams].values).float().cuda()
Ys_test = torch.tensor(df_test[ycolnams].values).float().cuda()

print(Xs_test.shape)
print(Ys_train[:,:10])

mod = nn.Sequential(
    nn.Linear(28**2, 400),
    nn.ReLU(),
    nn.Linear(400, 10),
    nn.ReLU()
)

mod = mod.cuda()

mod = linstorch.solvemodel(mod, Xs_train, Ys_train)

forw = mod.forward(Xs_train)
guessed = torch.argmax(forw, axis=0).cpu().numpy()
print(sum(guessed == df_train["label"].values) / len(guessed))