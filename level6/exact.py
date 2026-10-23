"""Shorten every program with the exact CP-SAT whole-program model (cpsat.full_model).

Usage: python3 exact.py input/in_level-6_3-large.txt [--time T] [--maxT N]
"""
import os
import sys
import time
from multiprocessing import Pool

from hexfab import read_input, write_output, simulate
from solver import Problem
from cycle import CycleProblem, prog_to_states
from cpsat import full_model

TLIM, MAXT = 60.0, 80


def run(args):
    k, radius, grid, count, prog = args
    if len(prog) > MAXT:
        return k, prog
    pb = Problem(radius, grid, count)
    cp = CycleProblem(pb)
    while True:
        st, p = full_model(cp, len(prog) - 1, TLIM, prog_to_states(cp, prog))
        if p is None or simulate(radius, grid, p) < count:
            return k, prog
        prog = p


def main():
    global TLIM, MAXT
    args = sys.argv[1:]
    files = []
    i = 0
    while i < len(args):
        if args[i] == '--time':
            TLIM = float(args[i + 1]); i += 2
        elif args[i] == '--maxT':
            MAXT = int(args[i + 1]); i += 2
        else:
            files.append(args[i]); i += 1
    here = os.path.dirname(os.path.abspath(__file__))
    for f in files:
        t0 = time.time()
        cases = read_input(f)
        out = os.path.join(here, 'output', os.path.basename(f).replace('in_', 'out_'))
        progs = [[l.split() for l in b.strip().split('\n')] for b in open(out).read().strip().split('\n\n')]
        start = sum(len(p) for p in progs)
        work = [(k, rad, grid, cnt, progs[k]) for k, (rad, rows, cols, cnt, grid) in enumerate(cases)]
        work.sort(key=lambda w: len(w[4]))
        done = 0
        with Pool(os.cpu_count()) as pool:
            for k, prog in pool.imap_unordered(run, work):
                if len(prog) < len(progs[k]):
                    progs[k] = prog
                    write_output(out, progs)
                done += 1
                if done % 10 == 0:
                    print(f'{done}/{len(cases)} score {sum(len(p) for p in progs)} ({time.time() - t0:.0f}s)', flush=True)
        write_output(out, progs)
        print(f'{os.path.basename(f)}: {start} -> {sum(len(p) for p in progs)}', flush=True)


if __name__ == '__main__':
    main()
