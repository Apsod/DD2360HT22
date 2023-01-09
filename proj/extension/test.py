import torch
import math
import argparse
import faclin_cuda
import itertools

class FaclinF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, *weights):
        *hidden, out = faclin_cuda.forward(x, weights)
        ctx.save_for_backward(x, *weights, *hidden)
        ctx.N = len(weights)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        N = ctx.N
        ts = ctx.saved_tensors
        x = ts[0]
        weights = ts[1:N+1]
        hidden = ts[N+1:]
        
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        #d_x,  = faclin_cuda.backward(x, weights, factors, grad_output)
        grads = [x, *weights]
        ret = []
        for g, f in zip(grads, ctx.needs_input_grad):
            ret.append(g if f else None)
        return x, *weights

faclin = FaclinF.apply

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
        self.ws = torch.nn.ParameterList()
        acc = 0
        for f in factors:
            self.ws.append(torch.nn.Parameter(W[acc:acc+f*f].reshape(f, f).clone()))
            acc += f*f

    def forward(self, x):
        x = faclin(x, *self.ws)
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

    dtype = {'f':torch.float32, 'd':torch.float64}[args.dtype]

    numX = B * math.prod(factors)
    numW = sum([f*f for f in factors])
    torch.manual_seed(0)
    x = torch.randn(numX, device='cuda', dtype=dtype)
    x = x.reshape(B, *list(reversed(factors)))
    W1 = torch.randn(numW, device='cuda', dtype=dtype)
    W2 = torch.randn(numW, device='cuda', dtype=dtype)
    acc = 0
    for f in factors:
        W1[acc:acc + f*f].mul_(f ** -.5)
        W2[acc:acc + f*f].mul_(f ** -.5)
        acc += f * f
    return factors, x, W1, W2


def check(args):
    factors, x, W1, W2 = init(args)

    fake_loss = torch.randn(x.numel(), device=x.device, dtype=x.dtype)
    
    x.requires_grad = True
    old = []
    for v, i in IMPMAP.items():
        m = Faclin(i, factors, W1, W2)
        y = m(x).flatten()
        l = (y @ fake_loss)
        l.backward()
        y = y.detach().to('cpu')
        g = x.grad.detach().to('cpu')
        old.append((v, y, g))
        x.grad.zero_()
        print(f'||{v}_y|| = {y.norm()}')
        print(f'||{v}_g|| = {g.norm()}')

    for a, b in itertools.combinations(old, 2):
        v1, y1, g1 = a
        v2, y2, g2 = b
        close = torch.allclose(y1, y2)
        error = (y1 - y2).norm()
        print(f'{v1}_y == {v2}_y: {close} ({error:.1e})')
        close = torch.allclose(g1, g2)
        error = (g1 - g2).norm()
        print(f'{v1}_g == {v2}_g: {close} ({error:.1e})')
    
    

def profile(args):
    factors, x, W1, W2 = init(args)
    impl = args.implementation

    m = Faclin(IMPMAP[impl], factors, W1, W2)

    run_name = '{}_{}_{}'.format(impl, args.dtype,'x'.join([str(f) for f in factors]))

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
    p.add_argument('--wait', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--active', type=int, default=4)
    p.set_defaults(go=profile)
    

def main():
    p = argparse.ArgumentParser()
    p.add_argument('batch_size', type=int)
    p.add_argument('factor', type=int, nargs='+')
    p.add_argument('--dtype', choices=['f', 'd'])
    subparser = p.add_subparsers()
    check_parser(subparser)
    profile_parser(subparser)
    args = p.parse_args()
    args.go(args)

if __name__ == '__main__':
    main()


