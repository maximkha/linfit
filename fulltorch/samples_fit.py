from math import sqrt
import pandas as pd
import numpy as np
import sys
import torch
import torch.nn as nn
import linstorch
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
from gen_data import gen_random_data_nd

df = pd.read_csv(r"fulltorch//data//auto-mpg.csv")
# df = pd.read_csv(r"data//super_trainb.csv")
df = df.dropna()

Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(df[["mpg", "acceleration", "displacement"]].values, df[["horsepower"]].values, test_size=0.33, random_state=42)
# print(f"{Ys_train=}")

# Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(df[["wtd_entropy_FusionHeat"]].values, df[["critical_temp"]].values, test_size=0.25, random_state=42
# x, y = gen_random_data_nd(np.linspace(-2, 2, 100), 1, 1)
# x = np.stack(np.meshgrid(np.linspace(0., 1., 100), np.linspace(0., 1., 100), indexing='xy')).reshape(2, -1).T
# y = np.cos(x[:,0] + x[:,1])
# x = np.linspace(0, 10, 1001) * 2 * np.pi * (1/10)
# # y = np.sin(x) + 1
# y = np.cos(x) + 1
# y = y.reshape(-1,1)
# x = x.reshape(-1,1)
# y = y.T
# print(f"{y.shape=}")
import matplotlib.pyplot as plt
# for i in range(y.shape[1]):
#     plt.scatter(x, y[:, i])
#     plt.show()
# Xs_train, Xs_test, Ys_train, Ys_test = train_test_split(x, y, test_size=0.33, random_state=42)

x_mean = np.mean(Xs_train)
x_std = np.std(Xs_train)

Xs_train -= x_mean
Xs_train /= x_std

Xs_test -= x_mean
Xs_test /= x_std

y_mean = np.mean(Ys_train)
y_std = np.std(Ys_train)

Ys_train -= y_mean
Ys_train /= y_std

Ys_test -= y_mean
Ys_test /= y_std

Ys_train = torch.tensor(Ys_train).float()#.cuda()
Ys_test = torch.tensor(Ys_test).float()#.cuda()

Xs_train = torch.tensor(Xs_train).float()#.cuda()
Xs_test = torch.tensor(Xs_test).float()#.cuda()

# plt.scatter(Xs_train[:, 0], Ys_train[:, 0])
# plt.show()

trainloader=torch.utils.data.DataLoader(list(zip(Xs_train, Ys_train)), batch_size=32, shuffle=True)
testloader=torch.utils.data.DataLoader(list(zip(Xs_test, Ys_test)), batch_size=32, shuffle=False)

from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, accuracy_score
from tqdm import tqdm

def genmodel() -> torch.nn.Sequential:
    mod = nn.Sequential(
        nn.Linear(3, 10),
        nn.ReLU(),
        # nn.Linear(10, 10),
        # nn.ReLU(),
        nn.Linear(10, 1),
    )

    return mod

import torch.optim as optim

def train(ep, opt):
    criterion = nn.MSELoss()
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

# #WARM UP
# mod = genmodel()
# mod = linstorch.solvemodel(mod, Xs_train, Ys_train)

# A = np.vstack([Xs_train.cpu().numpy()[:,0], np.ones(len(Xs_train.cpu().numpy()))]).T
# m, c = np.linalg.lstsq(A, Ys_train.cpu().numpy(), rcond=None)[0]
# print(f"{np.mean((Ys_train.cpu().numpy() - ((Xs_train.cpu().numpy() * m) + c))**2)=}")
# exit()

# for i in tqdm(list(range(100))):
#     mod = genmodel()
#     t0 = time.time()
#     cmse = 1
#     while cmse > .4:
#         mod = genmodel()
#         mod = linstorch.solvemodel(mod, Xs_train, Ys_train)
#         cmse = np.mean(((mod(Xs_train) - Ys_train)**2).detach().cpu().numpy())

#     elapsedsecs = time.time() - t0
#     forw = mod.forward(Xs_test)

#     perf_data["model"].append(f"linstorch")
#     perf_data["type"].append(f"linstorch")
#     perf_data["elapsed"].append(elapsedsecs)
#     perf_data["mse"].append(np.mean(((forw - Ys_test)**2).detach().cpu().numpy()))

# .7, .45, 20

print("lin reg")
with torch.no_grad():
    mod = nn.Sequential(nn.Linear(3, 1))
    t0 = time.time()
    mod = linstorch.solvemodel(mod, Xs_train, Ys_train)
    elapsedsecs = time.time() - t0

    perf_data["model"].append(f"linear_regression")
    perf_data["type"].append(f"linear_regression")
    perf_data["elapsed"].append(elapsedsecs)
    forw = mod.forward(Xs_test)
    perf_data["mse"].append(np.mean(((forw - Ys_test)**2).detach().cpu().numpy()))
    print(f"{torch.mean(((mod(Xs_train) - Ys_train)**2))=}")

print("linstorch")
with torch.no_grad():
    # for goal in np.linspace(1., .07, 20): # for goal in np.linspace(1., .82, 20): #np.linspace(.7, .13, 20):#np.linspace(.7, .3, 20): #[7.,.65,.6,.55,.5,.45]: #,.4,.35,.3]:
    # for goal in np.linspace(1., .3, 20): # for goal in np.linspace(1., .82, 20): #np.linspace(.7, .13, 20):#np.linspace(.7, .3, 20): #[7.,.65,.6,.55,.5,.45]: #,.4,.35,.3]:
    # for goal in np.linspace(1., .9, 40): # for goal in np.linspace(1., .82, 20): #np.linspace(.7, .13, 20):#np.linspace(.7, .3, 20): #[7.,.65,.6,.55,.5,.45]: #,.4,.35,.3]:
    for goal in np.linspace(.14, .11, 50): # for goal in np.linspace(1., .82, 20): #np.linspace(.7, .13, 20):#np.linspace(.7, .3, 20): #[7.,.65,.6,.55,.5,.45]: #,.4,.35,.3]:
        print(f"goal:{goal}")
        for i in tqdm(list(range(100))):
            mod = genmodel()
            t0 = time.time()
            while True:
                mod = genmodel()
                mod = linstorch.solvemodel(mod, Xs_train, Ys_train)
                if torch.mean(((mod(Xs_train) - Ys_train)**2)) <= goal: break

            elapsedsecs = time.time() - t0
            forw = mod.forward(Xs_test)

            perf_data["model"].append(f"linstorch({goal})")
            perf_data["type"].append(f"linstorch")
            perf_data["elapsed"].append(elapsedsecs)
            perf_data["mse"].append(np.mean(((forw - Ys_test)**2).detach().cpu().numpy()))

print("RANDOM")
with torch.no_grad():
    for goal in np.linspace(1., 0.56, 20): # for goal in np.linspace(1., .82, 20): #np.linspace(.7, .13, 20):#np.linspace(.7, .3, 20): #[7.,.65,.6,.55,.5,.45]: #,.4,.35,.3]:
        print(f"goal:{goal}")
        for i in tqdm(list(range(100))):
            mod = genmodel()
            t0 = time.time()
            while True:
                mod = genmodel()
                # mod = linstorch.solvemodel(mod, Xs_train, Ys_train)
                if torch.mean(((mod(Xs_train) - Ys_train)**2)) <= goal: break

            elapsedsecs = time.time() - t0
            forw = mod.forward(Xs_test)

            perf_data["model"].append(f"RANDOM({goal})")
            perf_data["type"].append(f"RANDOM")
            perf_data["elapsed"].append(elapsedsecs)
            perf_data["mse"].append(np.mean(((forw - Ys_test)**2).detach().cpu().numpy()))

print("SGD")

for j in list(range(1, 20)):
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

        perf_data["model"].append(f"SGD{j}")
        perf_data["type"].append(f"SGD")
        perf_data["elapsed"].append(elapsedsecs)
        perf_data["mse"].append(np.mean(((forw - Ys_test)**2).detach().cpu().numpy()))

print("Adagrad")

for j in list(range(1, 20)):
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

        perf_data["model"].append(f"Adagrad{j}")
        perf_data["type"].append(f"Adagrad")
        perf_data["elapsed"].append(elapsedsecs)
        perf_data["mse"].append(np.mean(((forw - Ys_test)**2).detach().cpu().numpy()))

print("Adam")

for j in list(range(1, 20)):
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

        perf_data["model"].append(f"Adam{j}")
        perf_data["type"].append(f"Adam")
        perf_data["elapsed"].append(elapsedsecs)
        perf_data["mse"].append(np.mean(((forw - Ys_test)**2).detach().cpu().numpy()))

import pandas as pd

df = pd.DataFrame(perf_data)
df.to_csv("out.csv")