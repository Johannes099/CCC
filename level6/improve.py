"""Improve existing outputs with the cycle annealer; keeps the shorter program per case.

Usage: python3 improve.py input/in_level-6_2-medium.txt [...] [--iters N] [--restarts R]
"""
import os
import sys
import time
from multiprocessing import Pool

from hexfab import read_input, write_output, simulate
from solver import Problem
from cycle import solve_cycle, polish, CycleProblem

ITERS, RESTARTS = 2000000, 4
SEEDS, SEED0, POLISH = 1, 0, 0


def work(args):
    i, radius, grid, count, old = args
    pb = Problem(radius, grid, count)
    best = old
    for seed in range(SEEDS):
        prog = solve_cycle(pb, simulate, radius, grid, max_L=32, iters=ITERS, restarts=RESTARTS,
                           seed=seed + SEED0, budget_best=len(best) if best else None)
        if prog and (best is None or len(prog) < len(best)):
            assert simulate(radius, grid, prog) >= count
            best = prog
    if POLISH:
        cp = CycleProblem(pb)
        cp.load_c()
        best = polish(cp, best, simulate, radius, grid, iters=POLISH)
    assert simulate(radius, grid, best) >= count
    return i, best


def main():
    global ITERS, RESTARTS, SEEDS, SEED0, POLISH
    args = sys.argv[1:]
    files = []
    i = 0
    while i < len(args):
        if args[i] == '--iters':
            ITERS = int(args[i + 1]); i += 2
        elif args[i] == '--seeds':
            SEEDS = int(args[i + 1]); i += 2
        elif args[i] == '--seed0':
            SEED0 = int(args[i + 1]); i += 2
        elif args[i] == '--polish':
            POLISH = int(args[i + 1]); i += 2
        elif args[i] == '--restarts':
            RESTARTS = int(args[i + 1]); i += 2
        else:
            files.append(args[i]); i += 1
    here = os.path.dirname(os.path.abspath(__file__))
    for f in files:
        t0 = time.time()
        cases = read_input(f)
        out = os.path.join(here, 'output', os.path.basename(f).replace('in_', 'out_'))
        olds = [None] * len(cases)
        if os.path.exists(out):
            blocks = open(out).read().strip().split('\n\n')
            olds = [[l.split() for l in b.strip().split('\n')] for b in blocks]
        jobs = [(k, rad, grid, cnt, olds[k]) for k, (rad, rows, cols, cnt, grid) in enumerate(cases)]
        progs = list(olds)
        with Pool(os.cpu_count()) as pool:
            for k, prog in pool.imap_unordered(work, jobs):
                progs[k] = prog
        write_output(out, progs)
        old_score = sum(len(p) for p in olds if p)
        print(f'{os.path.basename(f)}: {old_score} -> {sum(len(p) for p in progs)} '
              f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
