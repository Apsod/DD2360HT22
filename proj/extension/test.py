import torch
import math
import argparse
import faclin_cuda

class L1(torch.nn.Module):
    def __init__(self, factors, W):
        super().__init__()
        self.state_size = math.prod(factors)
        self.factors = list(reversed(factors))
        self.ws = torch.nn.ParameterList()
        acc = 0
        for f in factors:
            self.ws.append(torch.nn.Parameter(W[acc:acc+f*f].reshape(f, f).clone()))
            acc += f*f

    def forward(self, x):
        B = x.numel() // self.state_size
        x = x.reshape(B, *self.factors)
        for w in self.ws:
            x = (x @ w).movedim(-1, 1)
        return x

class L2(torch.nn.Module):
    def __init__(self, factors, W):
        super().__init__()
        self.state_size = math.prod(factors)
        self.factors = factors
        self.ws = torch.nn.ParameterList()
        acc = 0
        for f in factors:
            self.ws.append(torch.nn.Parameter(W[acc:acc+f*f].reshape(f, f).clone()))
            acc += f*f

    def forward(self, x):
        B = x.numel() // self.state_size
        for w in self.ws:
            x = x.reshape(-1, w.shape[0])
            x = w.t() @ x.t()
        x = x.reshape(-1, B).t()
        return x

class L3(torch.nn.Module):
    def __init__(self, factors, W):
        super().__init__()
        self.factors = factors
        self.ws = W

    def forward(self, x):
        x = faclin_cuda.forward(x, self.ws, self.factors)
        return x
IMPMAP = {
        'v1': L1,
        'v2': L2,
        'v3': L3,
        }

class Faclin(torch.nn.Module):
    def __init__(self, inner, factors, *WS):
        super().__init__()
        self.factors = factors
        self.m = torch.nn.Sequential(
                *[inner(factors, W) for W in WS]
                )

    def forward(self, x):
        return self.m(x)

def init(args):
    B = args.batch_size
    factors = args.factor

    numX = B * math.prod(factors)
    numW = sum([f*f for f in factors])
    torch.manual_seed(0)
    x = torch.randn(numX, device='cuda')
    W1 = torch.randn(numW, device='cuda')
    W2 = torch.randn(numW, device='cuda')
    acc = 0
    for f in factors:
        W1[acc:acc + f*f].mul_(f ** -.5)
        W2[acc:acc + f*f].mul_(f ** -.5)
        acc += f * f
    return factors, x, W1, W2


def check(args):
    factors, x, W1, W2 = init(args)
    
    ys = []
    for v, i in IMPMAP.items():
        m = Faclin(i, factors, W1, W2)
        y = m(x).flatten().to('cpu')
        print(f'||{v}|| = {y.norm()}')
        for w, y2 in ys:
            close = torch.allclose(y, y2)
            error = (y - y2).norm()
            print(f'{v} == {w}: {close} ({error:.1e})')
        ys.append((v, y))

def profile(args):
    factors, x, W1, W2 = init(args)
    impl = args.implementation

    m = Faclin(IMPMAP[impl], factors, W1, W2)

    run_name = '{}_{}'.format(impl, 'x'.join([str(f) for f in factors]))

    N = args.wait + args.warmup + args.active

    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))
        prof.export_chrome_trace(f'{run_name}.{prof.step_num}.json')

    with torch.no_grad(), torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=args.wait,
            warmup=args.warmup,
            active=args.active,
            ),
        on_trace_ready=trace_handler,
        ) as p:
        for _ in range(N):
            x = m(x)
            p.step()

def check_parser(sp):
    p = sp.add_parser('check')
    p.set_defaults(go=check)

def profile_parser(sp):
    p = sp.add_parser('profile')
    p.add_argument('--implementation', type=str, choices=IMPMAP)
    p.add_argument('--wait', type=int, default=5)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--active', type=int, default=10)
    p.set_defaults(go=profile)
    

def main():
    p = argparse.ArgumentParser()
    p.add_argument('batch_size', type=int)
    p.add_argument('factor', type=int, nargs='+')
    subparser = p.add_subparsers()
    check_parser(subparser)
    profile_parser(subparser)
    args = p.parse_args()
    args.go(args)

if __name__ == '__main__':
    main()


