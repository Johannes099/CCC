"""Exact cycle search with OR-Tools CP-SAT.

For a cycle length L every track has a one-hot state (position, active) per step,
consecutive states (cyclically) must be one legal instruction apart.  A cell's hit
count at a step is the sum of at most 6 booleans ("track t is active at the one
position whose ray crosses the cell").  Matter is propagated along the cycle:
  * every pattern cell must end the cycle as matter (last event = exactly 2 hits),
  * every skirting cell must end the cycle empty.
The objective is the setup length (moves from the home corners to the cycle).

Usage: python3 cpsat.py input/in_level-6_3-large.txt [--time T] [--jobs J]
"""
import os
import sys
import time
from multiprocessing import Pool

from ortools.sat.python import cp_model

from hexfab import read_input, write_output, simulate
from solver import Problem
from cycle import CycleProblem, simulate_prefix, polish

NT = 6


def cycle_model(cp, L, time_limit, hint=None, accepts=None, workers=1):
    """accepts: sorted step indices where a pattern is completed (default: only the last)."""
    if accepts is None:
        accepts = [L - 1]
    acc_set = set(accepts)
    pb = cp.pb
    plen = cp.plen
    m = cp_model.CpModel()
    states = [(p, a) for p in range(plen) for a in (0, 1)]
    x = [[{s: m.NewBoolVar('') for s in states} for _ in range(L)] for _ in range(NT)]
    for t in range(NT):
        for i in range(L):
            m.AddExactlyOne(x[t][i].values())
        if L > 1:
            for i in range(L):
                nxt = (i + 1) % L
                for (p, a) in states:
                    preds = [(p, a), ((p - 1) % plen, a), ((p + 1) % plen, a), (p, 1 - a)]
                    m.AddBoolOr([x[t][i][s] for s in preds]).OnlyEnforceIf(x[t][nxt][(p, a)])

    def hits(c, i):
        bs = []
        for t in range(NT):
            p = int(pb.target[t, c])
            if p >= 0:
                bs.append(x[t][i][(p, 1)])
        return bs

    for group, is_s in ((pb.S, True), (pb.K, False)):
        for c in group:
            c = int(c)
            prev = None
            for i in range(L):
                bs = hits(c, i)
                h = sum(bs)
                e2 = m.NewBoolVar('')
                e3 = m.NewBoolVar('')
                m.Add(h == 2).OnlyEnforceIf(e2)
                m.Add(h != 2).OnlyEnforceIf(e2.Not())
                m.Add(h >= 3).OnlyEnforceIf(e3)
                m.Add(h <= 2).OnlyEnforceIf(e3.Not())
                cur = m.NewBoolVar('')
                # cur == e2 or (prev and not e3); pattern cells restart empty after an accept
                if prev is None:
                    m.Add(cur == e2)
                else:
                    keep = m.NewBoolVar('')
                    m.AddBoolAnd([prev, e3.Not()]).OnlyEnforceIf(keep)
                    m.AddBoolOr([prev.Not(), e3, keep])
                    m.AddMaxEquality(cur, [e2, keep])
                if i in acc_set:
                    m.Add(cur == (1 if is_s else 0))
                    if is_s:
                        prev = None
                        continue
                prev = cur

    # setup: distance from home to the last state (+1 if it is active)
    setup = m.NewIntVar(0, plen, '')
    for t in range(NT):
        m.Add(setup >= sum(x[t][L - 1][(p, a)] * (cp.circ(cp.home[t], p) + a) for (p, a) in states))
    m.Minimize(setup)
    if hint:
        for t in range(NT):
            for i in range(L):
                for s in states:
                    m.AddHint(x[t][i][s], hint[t][i] == s)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    st = solver.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        traj = [[next(s for s in states if solver.Value(x[t][i][s])) for i in range(L)] for t in range(NT)]
        return 'ok', traj
    return ('infeasible' if st == cp_model.INFEASIBLE else 'unknown'), None


def run_case(args):
    k, radius, grid, count, old, tlim = args
    pb = Problem(radius, grid, count)
    cp = CycleProblem(pb)
    cp.load_c()
    best = old
    L = 1
    while L <= 32 and (best is None or 1 + count * L < len(best)):
        # warm start from the C annealer
        c, traj = cp.anneal_c(L, 1000000, L)
        hint = traj if c < 100 else None
        status, traj2 = cycle_model(cp, L, tlim, hint)
        if traj2 is None and hint is not None:
            traj2 = hint
        if traj2 is not None:
            prog = cp.program(traj2, L, count + 2)
            n = simulate_prefix(simulate, radius, grid, prog, count)
            if n is not None and (best is None or n < len(best)):
                best = prog[:n]
        L += 1
    best = polish(cp, best, simulate, radius, grid, iters=1000000, tries=1)
    assert simulate(radius, grid, best) >= count
    return k, best


def main():
    args = sys.argv[1:]
    tlim, jobs = 20.0, os.cpu_count()
    files = []
    i = 0
    while i < len(args):
        if args[i] == '--time':
            tlim = float(args[i + 1]); i += 2
        elif args[i] == '--jobs':
            jobs = int(args[i + 1]); i += 2
        else:
            files.append(args[i]); i += 1
    here = os.path.dirname(os.path.abspath(__file__))
    for f in files:
        t0 = time.time()
        cases = read_input(f)
        out = os.path.join(here, 'output', os.path.basename(f).replace('in_', 'out_'))
        olds = [None] * len(cases)
        if os.path.exists(out):
            olds = [[l.split() for l in b.strip().split('\n')] for b in open(out).read().strip().split('\n\n')]
        progs = list(olds)
        work = [(k, rad, grid, cnt, olds[k], tlim) for k, (rad, rows, cols, cnt, grid) in enumerate(cases)]
        # biggest programs first so the slow ones start early
        work.sort(key=lambda w: -(len(w[4]) if w[4] else 10 ** 6))
        done = 0
        with Pool(jobs) as pool:
            for k, prog in pool.imap_unordered(run_case, work):
                progs[k] = prog
                done += 1
                if done % 10 == 0:
                    write_output(out, progs)
                    print(f'{done}/{len(cases)} score {sum(len(p) for p in progs)}', flush=True)
        write_output(out, progs)
        print(f'{os.path.basename(f)}: {sum(len(p) for p in olds if p)} -> '
              f'{sum(len(p) for p in progs)} ({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()


def full_model(cp, T, time_limit, hint=None, workers=1):
    """Exact model of a whole program of length T (real acceptance semantics).
    Returns a list of program lines or None."""
    pb = cp.pb
    plen = cp.plen
    m = cp_model.CpModel()
    states = [(p, a) for p in range(plen) for a in (0, 1)]
    x = [[{s: m.NewBoolVar('') for s in states} for _ in range(T)] for _ in range(NT)]
    for t in range(NT):
        home = (cp.home[t], 0)
        for i in range(T):
            m.AddExactlyOne(x[t][i].values())
            for (p, a) in states:
                preds = [(p, a), ((p - 1) % plen, a), ((p + 1) % plen, a), (p, 1 - a)]
                if i == 0:
                    if home not in preds:
                        m.Add(x[t][0][(p, a)] == 0)
                else:
                    m.AddBoolOr([x[t][i - 1][s] for s in preds]).OnlyEnforceIf(x[t][i][(p, a)])

    def ev(c, i):
        bs = [x[t][i][(int(pb.target[t, c]), 1)] for t in range(NT) if pb.target[t, c] >= 0]
        h = sum(bs)
        e2 = m.NewBoolVar('')
        e3 = m.NewBoolVar('')
        m.Add(h == 2).OnlyEnforceIf(e2)
        m.Add(h != 2).OnlyEnforceIf(e2.Not())
        m.Add(h >= 3).OnlyEnforceIf(e3)
        m.Add(h <= 2).OnlyEnforceIf(e3.Not())
        return e2, e3

    acc = [m.NewBoolVar('') for _ in range(T)]
    mat = {}
    for c in list(pb.S) + list(pb.K):
        c = int(c)
        is_s = c in set(int(v) for v in pb.S)
        prev = None
        for i in range(T):
            e2, e3 = ev(c, i)
            cur = m.NewBoolVar('')
            if prev is None:
                m.Add(cur == e2)
            else:
                keep = m.NewBoolVar('')
                # keep = prev and not e3 (and, for pattern cells, no accept at i-1)
                lits = [prev, e3.Not()] + ([acc[i - 1].Not()] if is_s else [])
                m.AddBoolAnd(lits).OnlyEnforceIf(keep)
                m.AddBoolOr([l.Not() for l in lits] + [keep])
                m.AddMaxEquality(cur, [e2, keep])
            mat[c, i] = cur
            prev = cur
    for i in range(T):
        lits = [mat[int(c), i] for c in pb.S] + [mat[int(c), i].Not() for c in pb.K]
        m.AddBoolAnd(lits).OnlyEnforceIf(acc[i])
        m.AddBoolOr([l.Not() for l in lits] + [acc[i]])
    m.Add(sum(acc) >= cp.count)
    if hint:
        for t in range(NT):
            for i in range(min(T, len(hint[t]))):
                for s in states:
                    m.AddHint(x[t][i][s], hint[t][i] == s)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    st = solver.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        from cycle import states_to_prog
        traj = [[next(s for s in states if solver.Value(x[t][i][s])) for i in range(T)] for t in range(NT)]
        return 'ok', states_to_prog(cp, traj, T)
    return ('infeasible' if st == cp_model.INFEASIBLE else 'unknown'), None
