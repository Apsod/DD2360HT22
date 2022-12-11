import os
import itertools

files = """
rho_net_10.vtk
rhoe_10.vtk
rhoi_10.vtk
""".strip().split()

dirs = ['data', 'data_gt']

for f in files:
    f1 = open(os.path.join('data', f), 'rt')
    f2 = open(os.path.join('data_gt', f), 'rt')
    mean_diff = 0.0
    max_diff = 0.0
    N = 0.0
    mean_relative = 0.0
    max_relative = 0.0
    for l1, l2 in itertools.islice(zip(f1, f2), 10, None):
        v1 = float(l1)
        v2 = float(l2)
        diff = abs(v1-v2)
        relative = 2 * diff / (v1 + v2)

        max_diff = max(diff, max_diff)
        max_relative = max(relative, max_relative)
        N += 1; 
        mean_diff += (diff - mean_diff) / (N + 1)
        mean_relative += (relative - mean_relative) / (N + 1)
    print(f'{f}:')
    print(f'  max diff : {max_diff:.3E}, max relative : {max_relative}')
    print(f'  mean diff: {mean_diff:.3E}, mean relative: {mean_relative}')
    
