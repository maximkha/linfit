import numpy as np
 
def gen(degree=15, taylor_like=True):
    coefs = np.random.normal(0, 1, size=degree)
    if taylor_like: coefs /= np.array([np.math.factorial(x) for x in range(degree)])
    linearity = coefs[1] / np.sum(np.abs(coefs[1:]))
    print(f"linearity={linearity}")
    return np.polynomial.Polynomial(coefs)
 
def gen_random_data(xs, sampling=1):
    f = gen(taylor_like=True)
    nxs = xs.repeat(sampling)
    nxs += np.random.normal(scale=.25, size=nxs.shape[0])
    ys = f(nxs)
    ys += np.random.normal(scale=.25, size=ys.shape[0])
    idx = np.arange(ys.shape[0])
    np.random.shuffle(idx)
    return xs[idx], ys[idx]

def gen_random_data_nd(xs, sampling=1, n=2):
    nxs = xs.repeat(sampling)
    nxs += np.random.normal(scale=.25, size=nxs.shape[0])
    ys = np.concatenate([(gen(taylor_like=True)(nxs))[np.newaxis,:] for _ in range(n)], axis=0)
    ys += np.random.normal(scale=.25, size=(n, ys.shape[1]))
    idx = np.arange(ys.shape[1])
    np.random.shuffle(idx)
    return xs[idx], ys[:, idx]