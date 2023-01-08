import torch
import math
import argparse
import faclin_cuda

p = argparse.ArgumentParser()
p.add_argument('batch_size', type=int)
p.add_argument('factor', type=int, nargs='+')
p.add_argument('--implementation', type=str, choices=['v1', 'v2', 'v3'], default='v3')

args = p.parse_args()

B = args.batch_size
factors = args.factor
impl = args.implementation

numX = B * math.prod(factors)
numW = sum([f*f for f in factors])
torch.manual_seed(0)
x = torch.randn(numX, device='cuda')
W1 = torch.randn(numW, device='cuda')
W2 = torch.randn(numW, device='cuda')

ws1 = []
ws2 = []
acc = 0
for f in factors:
    ws1.append(W1[acc:acc+f*f].reshape(f,f))
    ws2.append(W2[acc:acc+f*f].reshape(f,f))
    acc += f*f

class L1(torch.nn.Module):
    def __init__(self, factors, W):
        super().__init__()
        self.factors = factors
        self.ws = torch.nn.ParameterList()
        acc = 0
        for f in factors:
            self.ws.append(torch.nn.Parameter(W[acc:acc+f*f].reshape(f, f).clone()))
            acc += f*f

    def forward(self, x):
        x = x.reshape(-1, *self.factors)
        for w in self.ws:
            x = x.movedim(1, -1) @ w
        return x.flatten()

class L2(torch.nn.Module):
    def __init__(self, factors, W):
        self.factors = factors
        self.ws = torch.nn.ParameterList()
        acc = 0
        for f in factors:
            self.ws.append(torch.nn.Parameter(W[acc:acc+f*f].reshape(f, f).clone()))
            acc += f*f

    def forward(self, x):
        for w in self.ws:
            x = x.reshape(-1, w.shape[0])
            x = w.t() @ x.t()
        x = x.reshape(-1, B).t()
        return x.flatten()

class L3(object):
    def __init__(self, factors, W):
        self.factors = factors
        self.ws = WS

    def run(self, x):
        x = faclin_cuda.forward(x, self.ws, self.factors)
        return x.flatten()
IMP = {
        'v1': L1,
        'v2': L2,
        'v3': L3,
        }[impl]

class Faclin(torch.nn.Module):
    def __init__(self, inner, factors, *WS):
        super().__init__()
        self.factors = factors
        self.m = torch.nn.Sequential(
                *[inner(factors, W) for W in WS]
                )

    def forward(self, x):
        return self.m(x)

m = Faclin(IMP, factors, W1, W2)

with torch.cuda.profiler.profile():
    m(x)
    with torch.autograd.profiler.emit_nvtx():
        opt = torch.optim.SGD(m.parameters(), lr=1e-3)
        for _ in range(10):
            opt.zero_grad()
            loss = m(x).mean()
            loss.backward()
            opt.step()

