"""Heavy cycle search (CP-SAT + annealer hints) for the most expensive cases.

Usage: python3 heavy.py input/in_level-6_3-large.txt --top N [--time T] [--workers W]
"""
import os
import sys
import time

from hexfab import read_input, write_output, simulate
from solver import Problem
from cycle import CycleProblem, simulate_prefix, greedy_delete
from cpsat import cycle_model


def main():
    args = sys.argv[1:]
    f = args[0]
    top = int(args[args.index('--top') + 1]) if '--top' in args else 40
    tlim = float(args[args.index('--time') + 1]) if '--time' in args else 120
    workers = int(args[args.index('--workers') + 1]) if '--workers' in args else 4
    here = os.path.dirname(os.path.abspath(__file__))
    cases = read_input(f)
    out = os.path.join(here, 'output', os.path.basename(f).replace('in_', 'out_'))
    progs = [[l.split() for l in b.strip().split('\n')] for b in open(out).read().strip().split('\n\n')]
    order = sorted(range(len(cases)), key=lambda k: -len(progs[k]))[:top]
    t0 = time.time()
    for k in order:
        rad, rows, cols, cnt, grid = cases[k]
        pb = Problem(rad, grid, cnt)
        cp = CycleProblem(pb)
        cp.load_c()
        before = len(progs[k])
        L = 1
        while 1 + cnt * L < len(progs[k]):
            c, hint = cp.anneal_c(L, 3000000, L)
            if c < 100:
                status, traj = 'anneal', hint
                st2, tr2 = cycle_model(cp, L, tlim / 4, hint, workers=workers)
                if tr2 is not None:
                    traj = tr2
            else:
                status, traj = cycle_model(cp, L, tlim, None, workers=workers)
            if traj is not None:
                prog = cp.program(traj, L, cnt + 2)
                n = simulate_prefix(simulate, rad, grid, prog, cnt)
                if n is not None and n < len(progs[k]):
                    p = greedy_delete(cp, prog[:n], simulate, rad, grid)
                    assert simulate(rad, grid, p) >= cnt
                    progs[k] = p
                    write_output(out, progs)
            L += 1
        print(f'case {k}: {before} -> {len(progs[k])}  total {sum(len(p) for p in progs)} '
              f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
