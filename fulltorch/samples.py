import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import linstorch
import time
from collections import defaultdict

df_train = pd.read_csv(r"C:\Users\maxim\Desktop\js\linfit\test\archive\mnist_train.csv")
df_test = pd.read_csv(r"C:\Users\maxim\Desktop\js\linfit\test\archive\mnist_test.csv")

xcolnams = df_train.columns.tolist()
print("check0")
df_train[xcolnams[1:]] /= 255. #convert the pixel values to 0..1

print("check1")

Xs_train = torch.tensor(df_train[xcolnams[1:]].values).float().cuda()
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

Xs_test = torch.tensor(df_test[xcolnams[1:]].values).float().cuda()
print("check2")

ycolnams = []
for i in range(10):
    coln = "is"+str(i)
    ycolnams.append(coln)
    df_test[coln] = (df_test["label"] == i).astype(float)

Ys_train = torch.tensor(df_train[ycolnams].values).float().cuda()
Ys_test = torch.tensor(df_test[ycolnams].values).float().cuda()

trainloader=torch.utils.data.DataLoader(list(zip(Xs_train, torch.tensor(df_train["label"].values).cuda())), batch_size=32, shuffle=True)
testloader=torch.utils.data.DataLoader(list(zip(Xs_test, torch.tensor(df_test["label"].values).cuda())), batch_size=32, shuffle=False)

from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score
from tqdm import tqdm

def genmodel(use_softmax: bool = True) -> torch.nn.Sequential:
    mod = nn.Sequential(
        nn.Linear(28**2, 400),
        nn.ReLU(),
        nn.Linear(400, 10),
        nn.Softmax() if use_softmax else nn.Identity() #nn.ReLU()
    )

    mod = mod.cuda()
    return mod

import torch.optim as optim

def train(ep, opt):
    criterion = nn.CrossEntropyLoss()
    for _ in range(ep):  # loop over the dataset multiple times
        # running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # print(labels.shape)
            #labels = torch.argmax(labels, axis=1)
            # print(labels)
            # print(labels.shape)
            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = mod(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #         (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

    # print('Finished Training')

MODE = "weighted"

perf_data = defaultdict(lambda: [])

print("linstorch")

for i in tqdm(list(range(100))):
    mod = genmodel(False)
    t0 = time.time()
    # mod = linstorch.solvemodel(mod, Xs_train, (2*Ys_train)-1)
    mod = linstorch.solvemodel(mod, Xs_train, (2*Ys_train)-1)
    elapsedsecs = time.time() - t0
    # print(f'{elapsedsecs} seconds')

    # forw = mod.forward(Xs_test)
    forw = mod.forward((2*Xs_test)-1)
    guessed = torch.argmax(forw, axis=1).cpu().numpy()
    labels = df_test["label"].values

    perf_data["model"].append("linstorch")
    perf_data["elapsed"].append(elapsedsecs)
    perf_data["accuracy"].append(balanced_accuracy_score(labels, guessed))
    perf_data["recall"].append(recall_score(labels, guessed, average=MODE))
    perf_data["precision"].append(precision_score(labels, guessed, average=MODE))
    perf_data["f1"].append(f1_score(labels, guessed, average=MODE))

print("SGD")

for j in list(range(1, 4)):
    for i in tqdm(list(range(100))):
        mod = genmodel()
        t0 = time.time()
        #optimizer = optim.Adam(mod.parameters())
        optimizer = optim.SGD(mod.parameters(), lr=0.001, momentum=0.9)
        #optimizer = optim.Adagrad(mod.parameters()) #optim.Adagrad(mod.parameters(), lr=0.001)
        train(j, optimizer)
        elapsedsecs = time.time() - t0
        # print(f'{elapsedsecs} seconds')

        forw = mod.forward(Xs_test)
        guessed = torch.argmax(forw, axis=1).cpu().numpy()
        labels = df_test["label"].values

        perf_data["model"].append(f"SGD{j}")
        perf_data["elapsed"].append(elapsedsecs)
        perf_data["accuracy"].append(balanced_accuracy_score(labels, guessed))
        perf_data["recall"].append(recall_score(labels, guessed, average=MODE))
        perf_data["precision"].append(precision_score(labels, guessed, average=MODE))
        perf_data["f1"].append(f1_score(labels, guessed, average=MODE))

print("Adagrad")

for j in list(range(1, 4)):
    for i in tqdm(list(range(100))):
        mod = genmodel()
        t0 = time.time()
        #optimizer = optim.Adam(mod.parameters())
        optimizer = optim.Adagrad(mod.parameters())
        #optimizer = optim.Adagrad(mod.parameters()) #optim.Adagrad(mod.parameters(), lr=0.001)
        train(j, optimizer)
        elapsedsecs = time.time() - t0
        # print(f'{elapsedsecs} seconds')

        forw = mod.forward(Xs_test)
        guessed = torch.argmax(forw, axis=1).cpu().numpy()
        labels = df_test["label"].values

        perf_data["model"].append(f"Adagrad{j}")
        perf_data["elapsed"].append(elapsedsecs)
        perf_data["accuracy"].append(balanced_accuracy_score(labels, guessed))
        perf_data["recall"].append(recall_score(labels, guessed, average=MODE))
        perf_data["precision"].append(precision_score(labels, guessed, average=MODE))
        perf_data["f1"].append(f1_score(labels, guessed, average=MODE))

print("Adam")

for j in list(range(1, 4)):
    for i in tqdm(list(range(100))):
        mod = genmodel()
        t0 = time.time()
        #optimizer = optim.Adam(mod.parameters())
        optimizer = optim.Adam(mod.parameters())
        #optimizer = optim.Adagrad(mod.parameters()) #optim.Adagrad(mod.parameters(), lr=0.001)
        train(j, optimizer)
        elapsedsecs = time.time() - t0
        # print(f'{elapsedsecs} seconds')

        forw = mod.forward(Xs_test)
        guessed = torch.argmax(forw, axis=1).cpu().numpy()
        labels = df_test["label"].values

        perf_data["model"].append(f"Adam{j}")
        perf_data["elapsed"].append(elapsedsecs)
        perf_data["accuracy"].append(balanced_accuracy_score(labels, guessed))
        perf_data["recall"].append(recall_score(labels, guessed, average=MODE))
        perf_data["precision"].append(precision_score(labels, guessed, average=MODE))
        perf_data["f1"].append(f1_score(labels, guessed, average=MODE))


# import json
# jsonFile = open("data.json", "w")
# jsonFile.write(json.dumps(perf_data))
# jsonFile.close()

import pandas as pd

df = pd.DataFrame(perf_data)
# print(df)
df.to_csv("out.csv")